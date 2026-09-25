"""
calibration/comparer.py

Compares a Leapfrog trajectory against an IAS15 reference trajectory,
computing the full set of per-timestep calibration metrics required by
the project spec, and assembles them into a single Pandas DataFrame
ready for CSV export (calibration/logger.py).

This module assumes both trajectories were sampled at identical
timestamps (guaranteed by core/leapfrog.py and core/rebound_reference.py
both being driven from the same ExperimentConfig).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibration.generator import ExperimentConfig
from core.leapfrog import CorrectionEvent, Trajectory
from core.metrics import (
    angular_momentum_drift,
    closest_pair_relative_velocity,
    energy_drift,
    jerk_finite_difference,
    mass_ratio,
    max_acceleration_magnitude,
    min_pairwise_distance,
    position_error,
    velocity_error,
)
from core.physics import compute_accelerations, compute_angular_momentum, compute_total_energy


def compare_trajectories(
    config: ExperimentConfig,
    leapfrog_traj: Trajectory,
    ias15_traj: Trajectory,
    corrections: list[CorrectionEvent] | None = None,
) -> pd.DataFrame:
    """
    Compute the full calibration comparison table between a Leapfrog
    trajectory and an IAS15 reference trajectory for one experiment.

    Args:
        config: The ExperimentConfig that produced both trajectories
            (used for simulation_id, seed, dt, body_count, g, softening).
        leapfrog_traj: Trajectory returned by core.leapfrog.run_leapfrog or
            core.leapfrog.run_leapfrog_with_correction.
        ias15_traj: Trajectory returned by core.rebound_reference.run_ias15.
        corrections: Correction events returned alongside `leapfrog_traj`
            by core.leapfrog.run_leapfrog_with_correction, if the
            corrected integrator was used. Pass None (default) or an
            empty list if no online correction was applied -- in that
            case the correction-related columns below are filled with
            correction_id=0 and NaN/0 as appropriate on every row.

    Returns:
        A Pandas DataFrame with one row per sampled timestep, containing
        every column required by calibration/logger.py:
        simulation_id, seed, timestep, body_count, time, mass_ratio,
        min_distance, relative_velocity, acceleration, jerk,
        position_error, velocity_error, total_energy, energy_drift,
        angular_momentum, angular_momentum_drift, correction_id,
        time_since_last_correction, steps_since_last_correction,
        error_at_trigger, correction_interval.

        For the correction columns: `correction_id` on each row is the ID
        of the most recent correction applied at or before that row (0 if
        none yet); `time_since_last_correction` / `steps_since_last_correction`
        are running counters that reset to 0 at each correction; and
        `error_at_trigger` / `correction_interval` are only populated
        (non-NaN) on the exact row where a correction was triggered.

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
    g = config.g
    softening = config.softening

    # Precompute per-sample accelerations for BOTH the jerk finite
    # difference and the max-acceleration-magnitude metric, evaluated on
    # the Leapfrog trajectory (the "fast" integrator under calibration).
    accelerations = np.zeros_like(leapfrog_traj.positions)
    for idx in range(n_samples):
        accelerations[idx] = compute_accelerations(
            leapfrog_traj.positions[idx], masses, g=g, softening=softening
        )

    # Initial energy / angular momentum (from Leapfrog's own initial state,
    # which is identical to IAS15's initial state by construction) used as
    # the drift reference point.
    initial_energy = compute_total_energy(
        leapfrog_traj.positions[0], leapfrog_traj.velocities[0], masses, g=g, softening=softening
    )
    initial_l = compute_angular_momentum(
        leapfrog_traj.positions[0], leapfrog_traj.velocities[0], masses
    )

    m_ratio = mass_ratio(masses)

    # Running correction-tracking state, updated as we walk forward through
    # the samples in order. `dt` (config.dt) is needed to translate a
    # sample's elapsed time back into a Leapfrog step count.
    last_correction_id = 0
    last_correction_time = 0.0
    last_correction_step = 0

    rows: list[dict] = []
    for idx in range(n_samples):
        lf_pos = leapfrog_traj.positions[idx]
        lf_vel = leapfrog_traj.velocities[idx]
        ref_pos = ias15_traj.positions[idx]
        ref_vel = ias15_traj.velocities[idx]

        pos_err = position_error(lf_pos, ref_pos)
        vel_err = velocity_error(lf_vel, ref_vel)

        min_dist = min_pairwise_distance(lf_pos)
        rel_vel = closest_pair_relative_velocity(lf_pos, lf_vel)
        max_acc = max_acceleration_magnitude(lf_pos, masses, g=g, softening=softening)

        # Jerk via central finite difference using neighboring samples
        # where available; boundary samples fall back to a one-sided
        # difference (jerk_finite_difference returns 0.0 only if dt <= 0).
        if 0 < idx < n_samples - 1:
            dt_local = times[idx + 1] - times[idx - 1]
            jerk = jerk_finite_difference(accelerations[idx - 1], accelerations[idx + 1], dt_local)
        elif idx == 0 and n_samples > 1:
            dt_local = times[idx + 1] - times[idx]
            jerk = jerk_finite_difference(accelerations[idx], accelerations[idx + 1], dt_local)
        elif idx == n_samples - 1 and n_samples > 1:
            dt_local = times[idx] - times[idx - 1]
            jerk = jerk_finite_difference(accelerations[idx - 1], accelerations[idx], dt_local)
        else:
            jerk = 0.0

        total_energy = compute_total_energy(lf_pos, lf_vel, masses, g=g, softening=softening)
        e_drift = energy_drift(total_energy, initial_energy)

        l_vec = compute_angular_momentum(lf_pos, lf_vel, masses)
        l_mag = float(np.linalg.norm(l_vec))
        l_drift = angular_momentum_drift(l_vec, initial_l)

        current_step = int(round(float(times[idx]) / config.dt))

        correction_event = corrections_by_sample_idx.get(idx)
        if correction_event is not None:
            # This row is the exact sample where a correction fired: the
            # trigger-specific columns are populated, and the running
            # "since last correction" counters reset to zero here.
            correction_id = correction_event.correction_id
            error_at_trigger = correction_event.error_at_trigger
            correction_interval = correction_event.correction_interval
            time_since_last_correction = 0.0
            steps_since_last_correction = 0

            last_correction_id = correction_event.correction_id
            last_correction_time = float(times[idx])
            last_correction_step = current_step
        else:
            correction_id = last_correction_id
            error_at_trigger = np.nan
            correction_interval = np.nan
            time_since_last_correction = float(times[idx]) - last_correction_time
            steps_since_last_correction = current_step - last_correction_step

        rows.append(
            {
                "simulation_id": config.simulation_id,
                "seed": config.seed,
                "timestep": config.dt,
                "body_count": config.body_count,
                "time": float(times[idx]),
                "mass_ratio": m_ratio,
                "min_distance": min_dist,
                "relative_velocity": rel_vel,
                "acceleration": max_acc,
                "jerk": jerk,
                "position_error": pos_err,
                "velocity_error": vel_err,
                "total_energy": total_energy,
                "energy_drift": e_drift,
                "angular_momentum": l_mag,
                "angular_momentum_drift": l_drift,
                "correction_id": correction_id,
                "time_since_last_correction": time_since_last_correction,
                "steps_since_last_correction": steps_since_last_correction,
                "error_at_trigger": error_at_trigger,
                "correction_interval": correction_interval,
            }
        )

    return pd.DataFrame(rows)
