"""
calibration/comparer.py

Compares a Leapfrog trajectory against an IAS15 reference trajectory and
assembles a PER-BODY calibration table: one row per (timestep, body),
identified by a `body_id` column. This replaced the old per-system (one
row per timestep, system-wide metrics only) table, because per-system
rows hide which body actually did what (e.g. `min_distance` alone can't
tell you that the closest pair changed identity, or that one specific
body -- not "the system" -- is the one being ejected).

calibration/logger.py is responsible for splitting this long-format
DataFrame into one CSV file per body (`<simulation_id>_body<NN>.csv`)
so logs are generated per body, not per system.

Every row carries:
  - identifying columns (simulation_id, seed, timestep, body_count, time,
    body_id, body_name, mass)
  - this body's own error/dynamics metrics (position/velocity error,
    acceleration, jerk)
  - this body's nearest-neighbor info
  - this body's online-correction bookkeeping (corrections reset the
    WHOLE system state at once, so the system-wide columns here --
    correction_id, error_at_trigger, correction_interval, etc. -- are
    identical across bodies for a given timestep and are included in
    every body's log so each per-body file is self-contained; but a new
    `body_triggered_correction` flag records whether THIS body's own
    error also crossed the threshold that caused the correction, which
    is what calibration/logger.py uses to decide whether a given body's
    file actually gets a row for that event -- see its module docstring)
  - this body's TRUE pre-correction dynamical state on a correction row
    (`pre_correction_acceleration`, `pre_correction_jerk`,
    `pre_correction_nearest_neighbor_distance`,
    `pre_correction_relative_velocity`): unlike this row's own
    `acceleration`/`jerk`/`nearest_neighbor_distance`/`relative_velocity`
    columns (computed by calibration/events.py from the sampled
    trajectory, which has ALREADY been snapped to the IAS15 reference by
    the time this row exists), these four are read straight off
    `CorrectionEvent.pre_correction_positions/velocities/acceleration/
    jerk` -- the actual erroneous state that triggered the correction,
    one internal integrator step before the snap. This is the intended
    input for fitting a real-time safety/accuracy index, since a
    predictor can never see the post-snap state before deciding whether
    to trust Leapfrog. Populated for EVERY row, not just corrections:
    on a non-correction row they describe the same pre-check quantities
    at that sample (the names keep the `pre_correction_` prefix for
    compatibility with existing CSVs and analysis scripts).
  - which kind of sample the row is (`sample_role`: correction,
    negative_pre_failure, negative_hard, negative_baseline; see the
    sampling policy at the top of this module and
    calibration/logger.py), `sampling_weight` (roughly how many real
    samples the row stands for), and `system_position_error_pre_check`
    (system RMS error at the pre-check state -- a label-side quantity for
    severity analysis, NOT something a real-time predictor could know).
  - NOTE: every state-derived column (position_error, velocity_error,
    energy drift, acceleration, jerk, nearest-neighbor info, and all
    situational flags) is computed from the PRE-CHECK Leapfrog state, so
    correction rows show the state that failed rather than the
    snapped-to-IAS15 state that replaced it.
  - every situational-event flag from calibration/events.py that is
    either intrinsically per-body, or a per-body "involved_in_*"
    projection of a system-wide situation (closest-pair switch,
    hierarchical configuration, near-symmetric configuration, chaotic
    scattering), plus the underlying system-wide context columns
    (e.g. system_min_distance) broadcast onto every body's row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibration.events import compute_body_events
from calibration.generator import ExperimentConfig
from core.leapfrog import CorrectionEvent, Trajectory
from core.metrics import (
    angular_momentum_drift,
    energy_drift,
    mass_ratio,
    nearest_neighbor_per_body,
    position_error,
)
from core.physics import compute_angular_momentum, compute_total_energy

# ---------------------------------------------------------------------------
# Negative-example sampling policy (which non-correction samples get logged)
# ---------------------------------------------------------------------------
# Logging every sample is too much storage, but logging ONLY corrections
# leaves a safety-index model with no examples of "Leapfrog was fine", so it
# can rank how bad failures were but can never learn where safe ends and
# unsafe begins. Each sampled timestep therefore gets a `sample_role`:
#
#   "correction"            a correction fired (positive example)
#   "negative_pre_failure"  one of the PRE_FAILURE_SAMPLES samples right
#                           before a correction: the last states that
#                           still looked fine, i.e. the hardest negatives
#   "negative_hard"         a near-miss (system RMS error >= NEAR_MISS_FRACTION
#                           * correction_threshold), thinned to every
#                           NEG_HARD_STRIDE-th such sample; OR a risky-
#                           looking moment (close encounter / rapid approach
#                           / high relative-velocity encounter / near
#                           collision) that is NOT a near-miss, thinned much
#                           harder (every NEG_RISKY_STRIDE-th) because in
#                           crowded systems almost every sample looks risky
#                           and would otherwise swamp the log. These teach a
#                           model that "looks risky" is not the same as
#                           "will fail".
#   "negative_baseline"     every NEG_BASELINE_STRIDE-th sample regardless of
#                           what is happening: an unbiased view of normal
#                           operation
#   ""                      not logged
#
# Priority when a sample qualifies for several: correction > pre_failure >
# hard > baseline. `sampling_weight` approximates how many real samples a
# logged row stands for (1 for correction/pre_failure, the stride for hard
# and baseline), so analysis can undo the deliberate over-representation of
# hard negatives when it needs honest base rates.
NEG_BASELINE_STRIDE: int = 20
NEG_HARD_STRIDE: int = 8
NEG_RISKY_STRIDE: int = 10
NEAR_MISS_FRACTION: float = 0.5
PRE_FAILURE_SAMPLES: int = 2


def _assign_sample_roles(
    n_samples: int,
    system_error: np.ndarray,
    correction_threshold: float,
    correction_sample_indices: set[int],
    risky_moment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decide, for every sample index, whether and why it gets logged (see the
    sampling-policy comment above).

    Args:
        n_samples: Number of samples in the trajectory.
        system_error: shape-(T,) system RMS position error at each sample's
            pre-check state.
        correction_threshold: The experiment's correction threshold.
        correction_sample_indices: Sample indices where a correction fired.
        risky_moment: shape-(T,) bool, True where any body shows a
            risky-looking situational flag.

    Returns:
        (roles, weights): shape-(T,) object array of role strings ("" =
        not logged) and shape-(T,) float array of sampling weights.
    """
    roles = np.full(n_samples, "", dtype=object)
    weights = np.zeros(n_samples, dtype=float)

    pre_failure: set[int] = set()
    for c in correction_sample_indices:
        for k in range(1, PRE_FAILURE_SAMPLES + 1):
            if c - k >= 1:
                pre_failure.add(c - k)

    near_miss_counter = 0
    risky_counter = 0
    for idx in range(1, n_samples):  # sample 0 has no dynamical history
        if idx in correction_sample_indices:
            roles[idx], weights[idx] = "correction", 1.0
            continue
        if idx in pre_failure:
            roles[idx], weights[idx] = "negative_pre_failure", 1.0
            continue
        if system_error[idx] >= NEAR_MISS_FRACTION * correction_threshold:
            near_miss_counter += 1
            if near_miss_counter % NEG_HARD_STRIDE == 0:
                roles[idx], weights[idx] = "negative_hard", float(NEG_HARD_STRIDE)
                continue
        elif risky_moment[idx]:
            risky_counter += 1
            if risky_counter % NEG_RISKY_STRIDE == 0:
                roles[idx], weights[idx] = "negative_hard", float(NEG_RISKY_STRIDE)
                continue
        if idx % NEG_BASELINE_STRIDE == 0:
            roles[idx], weights[idx] = "negative_baseline", float(NEG_BASELINE_STRIDE)
    return roles, weights


def compare_trajectories(
    config: ExperimentConfig,
    leapfrog_traj: Trajectory,
    ias15_traj: Trajectory,
    corrections: list[CorrectionEvent] | None = None,
) -> pd.DataFrame:
    """
    Compute the full calibration comparison table between a Leapfrog
    trajectory and an IAS15 reference trajectory for one experiment, in
    PER-BODY long format (one row per timestep per body).

    Args:
        config: The ExperimentConfig that produced both trajectories
            (used for simulation_id, seed, dt, body_count, g, softening).
        leapfrog_traj: Trajectory returned by core.leapfrog.run_leapfrog or
            core.leapfrog.run_leapfrog_with_correction.
        ias15_traj: Trajectory returned by core.rebound_reference.run_ias15.
        corrections: Correction events returned alongside `leapfrog_traj`
            by core.leapfrog.run_leapfrog_with_correction, if the
            corrected integrator was used. Pass None (default) or an
            empty list if no online correction was applied.

    Returns:
        A Pandas DataFrame with one row per (sampled timestep, body); see
        calibration/logger.py::REQUIRED_COLUMNS for the exact schema.

    Raises:
        ValueError: If the two trajectories do not share identical
            timestamps (within floating point tolerance), which would
            indicate a sampling misconfiguration upstream.
    """
    if corrections is None:
        corrections = []
    corrections_by_sample_idx = {event.sample_index: event for event in corrections}
    if leapfrog_traj.times.shape != ias15_traj.times.shape or not np.allclose(
        leapfrog_traj.times, ias15_traj.times, atol=1e-9
    ):
        raise ValueError(
            "Leapfrog and IAS15 trajectories must share identical sample "
            "timestamps to be compared. Got mismatched `times` arrays."
        )

    times = leapfrog_traj.times
    n_samples = times.shape[0]
    masses = leapfrog_traj.masses
    n_bodies = masses.shape[0]
    g = config.g
    softening = config.softening

    body_names = [f"body_{i}" for i in range(n_bodies)]

    # Initial energy / angular momentum, used as the drift reference point
    # (system-wide quantities; identical across bodies, broadcast below).
    initial_energy = compute_total_energy(
        leapfrog_traj.positions[0], leapfrog_traj.velocities[0], masses, g=g, softening=softening
    )
    initial_l = compute_angular_momentum(
        leapfrog_traj.positions[0], leapfrog_traj.velocities[0], masses
    )
    m_ratio = mass_ratio(masses)

    # Per-body / per-system situational event detection over the WHOLE
    # trajectory at once (see calibration/events.py).
    #
    # Everything state-derived below (events/flags, position_error,
    # energy drift, nearest-neighbor geometry, ...) is computed from the
    # PRE-CHECK state -- the raw Leapfrog state before any snap-to-reference
    # -- not from `leapfrog_traj.positions`, which at correction samples
    # holds the already-snapped IAS15 state. Otherwise a correction row's
    # position_error would read ~0 and its finite-difference jerk/flags
    # would be polluted by the snap discontinuity, which would both hide
    # the real state that failed AND hand a classifier a spurious "snap
    # artifact" cue for telling corrections from non-corrections.
    # (Plain run_leapfrog trajectories carry no pre-check arrays and are
    # never snapped, so `positions` is already the raw state there.)
    lf_positions = (
        leapfrog_traj.pre_check_positions
        if leapfrog_traj.pre_check_positions is not None
        else leapfrog_traj.positions
    )
    lf_velocities = (
        leapfrog_traj.pre_check_velocities
        if leapfrog_traj.pre_check_velocities is not None
        else leapfrog_traj.velocities
    )

    events = compute_body_events(
        times=times,
        positions=lf_positions,
        velocities=lf_velocities,
        masses=masses,
        g=g,
        softening=softening,
    )

    # Per-sample, per-body pre-check acceleration/jerk magnitudes at the
    # integrator's own dt resolution. Trajectories from run_leapfrog_with_
    # correction carry these for every sample; for any other trajectory,
    # fall back to the correction events' copies (correction samples only)
    # and NaN elsewhere.
    if leapfrog_traj.pre_check_acceleration is not None and leapfrog_traj.pre_check_jerk is not None:
        pre_accel_mag = np.linalg.norm(leapfrog_traj.pre_check_acceleration, axis=-1)
        pre_jerk_mag = np.linalg.norm(leapfrog_traj.pre_check_jerk, axis=-1)
    else:
        pre_accel_mag = np.full((n_samples, n_bodies), np.nan)
        pre_jerk_mag = np.full((n_samples, n_bodies), np.nan)
        for c_idx, c_event in corrections_by_sample_idx.items():
            pre_accel_mag[c_idx] = np.linalg.norm(c_event.pre_correction_acceleration, axis=-1)
            pre_jerk_mag[c_idx] = np.linalg.norm(c_event.pre_correction_jerk, axis=-1)

    # Which samples get logged, and why (see the sampling-policy comment at
    # the top of this module).
    system_error = np.array(
        [position_error(lf_positions[i], ias15_traj.positions[i]) for i in range(n_samples)]
    )
    risky_moment = (
        events.close_encounter
        | events.rapid_approach
        | events.high_relative_velocity_encounter
        | events.near_collision
    ).any(axis=1)
    sample_roles, sample_weights = _assign_sample_roles(
        n_samples,
        system_error,
        config.correction_threshold,
        set(corrections_by_sample_idx.keys()),
        risky_moment,
    )

    # Running correction-tracking state (system-wide: a correction resets
    # every body's state simultaneously), walked forward sample by sample.
    last_correction_id = 0
    last_correction_time = 0.0
    last_correction_step = 0

    rows: list[dict] = []
    for idx in range(n_samples):
        lf_pos = lf_positions[idx]
        lf_vel = lf_velocities[idx]
        ref_pos = ias15_traj.positions[idx]
        ref_vel = ias15_traj.velocities[idx]

        total_energy = compute_total_energy(lf_pos, lf_vel, masses, g=g, softening=softening)
        e_drift = energy_drift(total_energy, initial_energy)
        l_vec = compute_angular_momentum(lf_pos, lf_vel, masses)
        l_mag = float(np.linalg.norm(l_vec))
        l_drift = angular_momentum_drift(l_vec, initial_l)

        current_step = int(round(float(times[idx]) / config.dt))

        correction_event = corrections_by_sample_idx.get(idx)
        if correction_event is not None:
            correction_id = correction_event.correction_id
            error_at_trigger = correction_event.error_at_trigger
            correction_interval = correction_event.correction_interval
            time_since_last_correction = 0.0
            steps_since_last_correction = 0

            # This body's OWN contribution to the system-wide RMS error
            # that triggered the correction. error_at_trigger above is a
            # single system-wide number and is identical for every body;
            # these two arrays are what actually differ per body, since
            # each body deviated from the IAS15 reference by a different
            # amount at the moment of the snap.
            body_position_error_at_trigger_arr = np.linalg.norm(
                correction_event.pre_correction_positions - correction_event.corrected_positions,
                axis=-1,
            )
            body_velocity_error_at_trigger_arr = np.linalg.norm(
                correction_event.pre_correction_velocities
                - correction_event.corrected_velocities,
                axis=-1,
            )

            # A correction is triggered by the system-wide RMS position
            # error exceeding config.correction_threshold, and when it
            # fires EVERY body's state is snapped simultaneously -- but
            # that does not mean every body's OWN deviation was actually
            # large. Since RMS = sqrt(mean(x_i^2)) <= max(|x_i|) always,
            # at least one body's own error is guaranteed to exceed the
            # threshold whenever the RMS does, but the others may not.
            # This per-body flag marks which bodies actually "own" this
            # correction (their own error also crossed the threshold);
            # calibration/logger.py uses it to keep a body's log limited
            # to corrections it was actually responsible for, rather than
            # every system-wide correction event regardless of whether
            # this body needed it.
            body_triggered_correction_arr = (
                body_position_error_at_trigger_arr > config.correction_threshold
            )

            last_correction_id = correction_event.correction_id
            last_correction_time = float(times[idx])
            last_correction_step = current_step
        else:
            correction_id = last_correction_id
            error_at_trigger = np.nan
            correction_interval = np.nan
            time_since_last_correction = float(times[idx]) - last_correction_time
            steps_since_last_correction = current_step - last_correction_step
            body_position_error_at_trigger_arr = np.full(n_bodies, np.nan)
            body_velocity_error_at_trigger_arr = np.full(n_bodies, np.nan)
            body_triggered_correction_arr = np.zeros(n_bodies, dtype=bool)

        # Pre-check nearest-neighbor geometry at this sample's raw (pre-snap)
        # state. Together with pre_accel_mag / pre_jerk_mag (precomputed
        # above from the integrator's own per-step values) these are the
        # four inputs a real-time safety index would have had, computed the
        # SAME way for correction and non-correction samples so a model
        # fit on them cannot key on how the row was produced.
        (
            pre_nn_distance_arr,
            pre_nn_relative_velocity_arr,
            _pre_nn_id_arr,
        ) = nearest_neighbor_per_body(lf_pos, lf_vel)
        pre_accel_mag_arr = pre_accel_mag[idx]
        pre_jerk_mag_arr = pre_jerk_mag[idx]

        sys_i = int(events.system_min_distance_pair_i[idx])
        sys_j = int(events.system_min_distance_pair_j[idx])

        for body_id in range(n_bodies):
            pos_err_body = float(np.linalg.norm(lf_pos[body_id] - ref_pos[body_id]))
            vel_err_body = float(np.linalg.norm(lf_vel[body_id] - ref_vel[body_id]))

            rows.append(
                {
                    # --- identity ---
                    "simulation_id": config.simulation_id,
                    "seed": config.seed,
                    "timestep": config.dt,
                    "body_count": config.body_count,
                    "time": float(times[idx]),
                    "body_id": body_id,
                    "body_name": body_names[body_id],
                    "mass": float(masses[body_id]),
                    "mass_ratio": m_ratio,
                    # --- this body's own accuracy metrics ---
                    "position_error": pos_err_body,
                    "velocity_error": vel_err_body,
                    # --- this body's own dynamics ---
                    "acceleration": float(events.acceleration_magnitude[idx, body_id]),
                    "jerk": float(events.jerk_magnitude[idx, body_id]),
                    "nearest_neighbor_id": int(events.nearest_neighbor_id[idx, body_id]),
                    "nearest_neighbor_distance": float(
                        events.nearest_neighbor_distance[idx, body_id]
                    ),
                    "relative_velocity": float(
                        events.nearest_neighbor_relative_velocity[idx, body_id]
                    ),
                    # --- system-wide state (broadcast; whole-system quantities) ---
                    "total_energy": total_energy,
                    "energy_drift": e_drift,
                    "angular_momentum": l_mag,
                    "angular_momentum_drift": l_drift,
                    "system_min_distance": float(events.system_min_distance[idx]),
                    "system_min_distance_pair": f"{sys_i}-{sys_j}" if sys_i >= 0 else "",
                    # --- online-correction bookkeeping (system-wide) ---
                    "correction_id": correction_id,
                    "time_since_last_correction": time_since_last_correction,
                    "steps_since_last_correction": steps_since_last_correction,
                    "error_at_trigger": error_at_trigger,
                    "correction_interval": correction_interval,
                    "body_position_error_at_trigger": float(
                        body_position_error_at_trigger_arr[body_id]
                    ),
                    "body_velocity_error_at_trigger": float(
                        body_velocity_error_at_trigger_arr[body_id]
                    ),
                    "body_triggered_correction": bool(
                        body_triggered_correction_arr[body_id]
                    ),
                    "pre_correction_acceleration": float(pre_accel_mag_arr[body_id]),
                    "pre_correction_jerk": float(pre_jerk_mag_arr[body_id]),
                    "pre_correction_nearest_neighbor_distance": float(
                        pre_nn_distance_arr[body_id]
                    ),
                    "pre_correction_relative_velocity": float(
                        pre_nn_relative_velocity_arr[body_id]
                    ),
                    # --- sample selection (see module-level sampling policy) ---
                    "system_position_error_pre_check": float(system_error[idx]),
                    "sample_role": str(sample_roles[idx]),
                    "sampling_weight": float(sample_weights[idx]),
                    # --- situational events: per-body ---
                    "strong_acceleration": bool(events.strong_acceleration[idx, body_id]),
                    "high_jerk": bool(events.high_jerk[idx, body_id]),
                    "acceleration_spike": bool(events.acceleration_spike[idx, body_id]),
                    "close_encounter": bool(events.close_encounter[idx, body_id]),
                    "rapid_approach": bool(events.rapid_approach[idx, body_id]),
                    "rapid_separation": bool(events.rapid_separation[idx, body_id]),
                    "high_relative_velocity_encounter": bool(
                        events.high_relative_velocity_encounter[idx, body_id]
                    ),
                    "near_collision": bool(events.near_collision[idx, body_id]),
                    "ejection_escape": bool(events.ejection_escape[idx, body_id]),
                    "temporary_capture": bool(events.temporary_capture[idx, body_id]),
                    "temporary_capture_partner_id": int(
                        events.temporary_capture_partner_id[idx, body_id]
                    ),
                    "strong_mass_ratio_interaction": bool(
                        events.strong_mass_ratio_interaction[idx, body_id]
                    ),
                    # --- situational events: system-wide, projected per-body ---
                    "closest_pair_switch": bool(events.closest_pair_switch[idx]),
                    "involved_in_closest_pair_switch": bool(
                        events.involved_in_closest_pair_switch[idx, body_id]
                    ),
                    "hierarchical_configuration": bool(events.hierarchical_configuration[idx]),
                    "involved_in_hierarchical_tight_pair": bool(
                        events.involved_in_hierarchical_tight_pair[idx, body_id]
                    ),
                    "near_symmetric_configuration": bool(
                        events.near_symmetric_configuration[idx]
                    ),
                    "chaotic_scattering": bool(events.chaotic_scattering[idx]),
                    # --- baseline: the complement of every anomaly flag above ---
                    "stable": bool(events.stable[idx, body_id]),
                    "quiescent_regime": bool(events.quiescent_regime[idx, body_id]),
                }
            )

    return pd.DataFrame(rows)
