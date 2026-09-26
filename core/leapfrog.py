"""
core/leapfrog.py

A from-scratch implementation of the Velocity Verlet (Leapfrog) symplectic
integrator for N-body gravitational systems. No external integration
libraries are used; this module only depends on core/physics.py for
force/acceleration evaluation.

Velocity Verlet update scheme (per timestep dt):

    a(t)        = acceleration at current positions
    x(t + dt)   = x(t) + v(t) * dt + 0.5 * a(t) * dt^2
    a(t + dt)   = acceleration at new positions x(t + dt)
    v(t + dt)   = v(t) + 0.5 * (a(t) + a(t + dt)) * dt

This scheme is second-order accurate and symplectic (good long-term energy
behavior), which makes it a reasonable "fast" integrator to calibrate
against the high-accuracy adaptive IAS15 integrator from REBOUND.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.physics import System, compute_accelerations


@dataclass
class Trajectory:
    """
    Container for a full simulated trajectory.

    Attributes:
        times: shape-(T,) float64 array of sample times.
        positions: shape-(T, N, 3) float64 array of positions at each sample time.
        velocities: shape-(T, N, 3) float64 array of velocities at each sample time.
        masses: shape-(N,) float64 array of body masses (constant over time).
    """

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray


@dataclass
class CorrectionEvent:
    """
    Record of a single correction event during a run of
    `run_leapfrog_with_correction`: the moment the Leapfrog state was
    snapped to the IAS15 reference state because the position error
    between them exceeded `error_threshold`.

    Attributes:
        correction_id: 1-indexed sequential identifier for this event
            within its run (the 1st correction is 1, the 2nd is 2, ...).
        sample_index: Index into the trajectory's `times`/`positions`/
            `velocities` arrays at which the correction was applied.
        time: Simulated time at which the correction occurred.
        step_index: Leapfrog integrator step count (in units of `dt`)
            at which the correction occurred.
        error_at_trigger: The position_error magnitude that exceeded
            `error_threshold` and triggered this correction.
        steps_since_last_correction: Number of Leapfrog steps since the
            previous correction (or since the start of the run, if this
            is the first correction).
        time_since_last_correction: Simulated time elapsed since the
            previous correction (or since t=0, if this is the first
            correction).
        correction_interval: The ground-truth interval between this
            correction and the previous one. Identical in value to
            `time_since_last_correction`; kept as a separate, clearly
            named field so that downstream ML pipelines can treat it as
            the label/target being predicted, distinct from any
            incrementally-computed feature of the same name.
        corrected_positions: shape-(N, 3) copy of the IAS15 positions the
            Leapfrog state was reset to.
        corrected_velocities: shape-(N, 3) copy of the IAS15 velocities
            the Leapfrog state was reset to.
        pre_correction_positions: shape-(N, 3) copy of the Leapfrog
            positions immediately BEFORE the snap-to-reference, i.e. the
            erroneous state that actually triggered this correction. This
            is what lets downstream analysis compute each body's own
            contribution to the trigger (`corrected_positions -
            pre_correction_positions`, per body), since `error_at_trigger`
            itself is a single system-wide RMS value and is identical for
            every body.
        pre_correction_velocities: shape-(N, 3) copy of the Leapfrog
            velocities immediately BEFORE the snap-to-reference.
    """

    correction_id: int
    sample_index: int
    time: float
    step_index: int
    error_at_trigger: float
    steps_since_last_correction: int
    time_since_last_correction: float
    correction_interval: float
    corrected_positions: np.ndarray
    corrected_velocities: np.ndarray
    pre_correction_positions: np.ndarray
    pre_correction_velocities: np.ndarray


def run_leapfrog(
    system: System,
    dt: float,
    total_time: float,
    sample_interval: float | None = None,
) -> Trajectory:
    """
    Integrate a gravitational N-body system forward in time using the
    Velocity Verlet (Leapfrog) method, implemented from scratch.

    Args:
        system: The initial System (bodies + G + softening). This object
            is NOT mutated; an internal working copy is used.
        dt: Integration timestep. Must be > 0.
        total_time: Total simulated duration. The integrator advances in
            steps of `dt` until it has covered at least `total_time`.
        sample_interval: If provided, trajectory samples are recorded every
            `sample_interval` of simulated time (must be a multiple of dt
            for exact alignment with the IAS15 reference sampling). If None,
            every integrator step is recorded (sample_interval = dt).

    Returns:
        A Trajectory object containing sampled times, positions, and
        velocities, plus the (constant) masses array.

    Raises:
        ValueError: If dt <= 0, total_time <= 0, or sample_interval is not
            a positive multiple of dt (within floating point tolerance).
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if total_time <= 0:
        raise ValueError(f"total_time must be positive, got {total_time}")

    if sample_interval is None:
        sample_interval = dt

    # Validate that sample_interval is (approximately) an integer multiple of dt,
    # so that Leapfrog samples land on exactly the same timestamps as IAS15.
    steps_per_sample = sample_interval / dt
    if abs(steps_per_sample - round(steps_per_sample)) > 1e-9:
        raise ValueError(
            f"sample_interval ({sample_interval}) must be an integer multiple "
            f"of dt ({dt}); got steps_per_sample={steps_per_sample}"
        )
    steps_per_sample = int(round(steps_per_sample))

    n_steps = int(round(total_time / dt))
    working = system.copy()

    masses = working.masses()
    g = working.g
    softening = working.softening

    positions = working.positions()
    velocities = working.velocities()

    times_out = [0.0]
    positions_out = [positions.copy()]
    velocities_out = [velocities.copy()]

    # Initial acceleration, reused as a(t) for the first half-kick.
    acc = compute_accelerations(positions, masses, g=g, softening=softening)

    current_time = 0.0
    for step in range(1, n_steps + 1):
        # Position update: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        positions = positions + velocities * dt + 0.5 * acc * dt * dt

        # Acceleration at the new positions: a(t+dt)
        new_acc = compute_accelerations(positions, masses, g=g, softening=softening)

        # Velocity update: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        velocities = velocities + 0.5 * (acc + new_acc) * dt

        acc = new_acc
        current_time = step * dt

        if step % steps_per_sample == 0:
            times_out.append(current_time)
            positions_out.append(positions.copy())
            velocities_out.append(velocities.copy())

    return Trajectory(
        times=np.array(times_out, dtype=np.float64),
        positions=np.array(positions_out, dtype=np.float64),
        velocities=np.array(velocities_out, dtype=np.float64),
        masses=masses,
    )


def run_leapfrog_with_correction(
    system: System,
    dt: float,
    total_time: float,
    reference_positions: np.ndarray,
    reference_velocities: np.ndarray,
    reference_times: np.ndarray,
    error_threshold: float,
) -> tuple[Trajectory, list[CorrectionEvent]]:
    """
    Integrate a gravitational N-body system forward in time using Velocity
    Verlet (Leapfrog), exactly like `run_leapfrog`, but with online error
    correction: at every sampled timestep, the current Leapfrog position
    is compared against the corresponding IAS15 reference position
    (`reference_positions`), and if the RMS position error exceeds
    `error_threshold`, the Leapfrog state (positions AND velocities of
    every body) is immediately overwritten with the IAS15 reference state
    before integration continues.

    This requires the IAS15 reference trajectory to already exist (see
    core/rebound_reference.py::run_ias15) and to have been sampled at
    exactly the same timestamps this function will produce -- i.e. every
    `sample_interval` implied by `reference_times`, which must itself be
    an integer multiple of `dt`.

    Args:
        system: The initial System (bodies + G + softening). Not mutated;
            an internal working copy is used.
        dt: Integration timestep. Must be > 0.
        total_time: Total simulated duration.
        reference_positions: shape-(T, N, 3) IAS15 positions, one row per
            sample, aligned with `reference_times`.
        reference_velocities: shape-(T, N, 3) IAS15 velocities, aligned
            with `reference_times`.
        reference_times: shape-(T,) sample timestamps the reference
            trajectory was recorded at. Must start at 0.0 and be spaced
            by a constant `sample_interval` that is an integer multiple
            of `dt`.
        error_threshold: RMS position error (see core.metrics.position_error)
            above which a correction is triggered. Must be > 0.

    Returns:
        A tuple of:
            Trajectory: the (possibly corrected) Leapfrog trajectory,
                sampled at the same timestamps as `reference_times`.
            list[CorrectionEvent]: every correction that was applied,
                in chronological order (possibly empty).

    Raises:
        ValueError: If dt <= 0, total_time <= 0, error_threshold <= 0, or
            `reference_times` is not a valid, dt-aligned sample grid.
    """
    # Local import to avoid a core.leapfrog <-> core.metrics import cycle at
    # module load time (core.metrics does not import core.leapfrog, but the
    # import is kept local here for clarity about the dependency direction).
    from core.metrics import position_error

    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if total_time <= 0:
        raise ValueError(f"total_time must be positive, got {total_time}")
    if error_threshold <= 0:
        raise ValueError(f"error_threshold must be positive, got {error_threshold}")
    if reference_times.shape[0] < 1 or abs(float(reference_times[0])) > 1e-9:
        raise ValueError("reference_times must be non-empty and start at 0.0")

    if reference_times.shape[0] < 2:
        sample_interval = dt
    else:
        sample_interval = float(reference_times[1] - reference_times[0])

    steps_per_sample = sample_interval / dt
    if abs(steps_per_sample - round(steps_per_sample)) > 1e-9:
        raise ValueError(
            f"reference sample_interval ({sample_interval}) must be an integer "
            f"multiple of dt ({dt}); got steps_per_sample={steps_per_sample}"
        )
    steps_per_sample = int(round(steps_per_sample))

    n_steps = int(round(total_time / dt))
    working = system.copy()

    masses = working.masses()
    g = working.g
    softening = working.softening

    positions = working.positions()
    velocities = working.velocities()

    times_out = [0.0]
    positions_out = [positions.copy()]
    velocities_out = [velocities.copy()]

    corrections: list[CorrectionEvent] = []
    next_correction_id = 1
    last_correction_time = 0.0
    last_correction_step = 0

    # Initial acceleration, reused as a(t) for the first half-kick.
    acc = compute_accelerations(positions, masses, g=g, softening=softening)

    sample_idx = 0  # index 0 corresponds to t = 0.0, already recorded above

    current_time = 0.0
    for step in range(1, n_steps + 1):
        # Position update: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        positions = positions + velocities * dt + 0.5 * acc * dt * dt

        # Acceleration at the new positions: a(t+dt)
        new_acc = compute_accelerations(positions, masses, g=g, softening=softening)

        # Velocity update: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        velocities = velocities + 0.5 * (acc + new_acc) * dt

        acc = new_acc
        current_time = step * dt

        if step % steps_per_sample == 0:
            sample_idx += 1

            # Compare against the IAS15 reference state at this same
            # sample index, and correct if the error exceeds threshold.
            ref_pos = reference_positions[sample_idx]
            error = position_error(positions, ref_pos)

            if error > error_threshold:
                steps_since_last = step - last_correction_step
                time_since_last = current_time - last_correction_time

                corrections.append(
                    CorrectionEvent(
                        correction_id=next_correction_id,
                        sample_index=sample_idx,
                        time=current_time,
                        step_index=step,
                        error_at_trigger=error,
                        steps_since_last_correction=steps_since_last,
                        time_since_last_correction=time_since_last,
                        correction_interval=time_since_last,
                        corrected_positions=reference_positions[sample_idx].copy(),
                        corrected_velocities=reference_velocities[sample_idx].copy(),
                        pre_correction_positions=positions.copy(),
                        pre_correction_velocities=velocities.copy(),
                    )
                )

                # Snap the Leapfrog state to the IAS15 reference state.
                positions = reference_positions[sample_idx].copy()
                velocities = reference_velocities[sample_idx].copy()
                # Recompute acceleration at the corrected positions so the
                # next Velocity Verlet half-kick is consistent with the
                # new state (acc must always equal a(current_time)).
                acc = compute_accelerations(positions, masses, g=g, softening=softening)

                next_correction_id += 1
                last_correction_time = current_time
                last_correction_step = step

            times_out.append(current_time)
            positions_out.append(positions.copy())
            velocities_out.append(velocities.copy())

    trajectory = Trajectory(
        times=np.array(times_out, dtype=np.float64),
        positions=np.array(positions_out, dtype=np.float64),
        velocities=np.array(velocities_out, dtype=np.float64),
        masses=masses,
    )
    return trajectory, corrections
