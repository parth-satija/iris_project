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
