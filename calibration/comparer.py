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
from core.metrics import angular_momentum_drift, energy_drift, mass_ratio
from core.physics import compute_angular_momentum, compute_total_energy


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
    events = compute_body_events(
        times=times,
        positions=leapfrog_traj.positions,
        velocities=leapfrog_traj.velocities,
        masses=masses,
        g=g,
        softening=softening,
    )

    # Running correction-tracking state (system-wide: a correction resets
    # every body's state simultaneously), walked forward sample by sample.
    last_correction_id = 0
    last_correction_time = 0.0
    last_correction_step = 0

    rows: list[dict] = []
    for idx in range(n_samples):
        lf_pos = leapfrog_traj.positions[idx]
        lf_vel = leapfrog_traj.velocities[idx]
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
