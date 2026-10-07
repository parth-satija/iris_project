"""
core/rebound_reference.py

Wraps the REBOUND library's IAS15 integrator to serve as the
high-accuracy reference trajectory against which the custom Leapfrog
integrator (core/leapfrog.py) is calibrated.

IAS15 (Rein & Spiegel 2015) is a 15th-order adaptive Gauss-Radau
integrator and is effectively "ground truth" for non-relativistic
N-body gravity at machine precision.

This module deliberately mirrors the Trajectory data structure used by
core/leapfrog.py so the two integrators can be compared timestep-for-
timestep in core/metrics.py.
"""

from __future__ import annotations

import numpy as np
import rebound

from core.leapfrog import Trajectory
from core.physics import System


def run_ias15(
    system: System,
    total_time: float,
    sample_interval: float,
) -> Trajectory:
    """
    Integrate a gravitational N-body system forward in time using
    REBOUND's IAS15 integrator, sampling at the same interval used by
    the Leapfrog reference run so the two trajectories are directly
    comparable timestamp-for-timestamp.

    Args:
        system: The initial System (bodies + G + softening). REBOUND's
            own softening (`sim.softening` via `rebound.Simulation`'s
            gravity is Newtonian by default) is matched to `system.softening`.
        total_time: Total simulated duration, matching the Leapfrog run.
        sample_interval: Time between recorded samples. Must match the
            Leapfrog run's sample_interval exactly for a valid comparison.

    Returns:
        A Trajectory object with times, positions, velocities sampled at
        multiples of `sample_interval` from 0 to total_time (inclusive).

    Raises:
        ValueError: If sample_interval <= 0 or total_time <= 0.
    """
    if sample_interval <= 0:
        raise ValueError(f"sample_interval must be positive, got {sample_interval}")
    if total_time <= 0:
        raise ValueError(f"total_time must be positive, got {total_time}")

    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = system.g

    # REBOUND's IAS15 supports a softening length applied uniformly to all
    # particles via sim.softening (added in quadrature to separations,
    # identical convention to core/physics.py's `softening` parameter).
    sim.softening = system.softening

    for body in system.bodies:
        sim.add(
            m=body.mass,
            x=body.position[0],
            y=body.position[1],
            z=body.position[2],
            vx=body.velocity[0],
            vy=body.velocity[1],
            vz=body.velocity[2],
        )

    # Deliberately do NOT call sim.move_to_com() here: we keep the system in
    # its original frame so results are directly comparable to the Leapfrog
    # run, which also does not recenter on the center of mass.

    n = system.n
    n_samples = int(round(total_time / sample_interval)) + 1

    times_out = np.zeros(n_samples, dtype=np.float64)
    positions_out = np.zeros((n_samples, n, 3), dtype=np.float64)
    velocities_out = np.zeros((n_samples, n, 3), dtype=np.float64)

    for sample_idx in range(n_samples):
        t = sample_idx * sample_interval
        if t > 0.0:
            sim.integrate(t, exact_finish_time=1)
        times_out[sample_idx] = t
        for i, particle in enumerate(sim.particles):
            positions_out[sample_idx, i] = (particle.x, particle.y, particle.z)
            velocities_out[sample_idx, i] = (particle.vx, particle.vy, particle.vz)

    masses = system.masses()

    return Trajectory(
        times=times_out,
        positions=positions_out,
        velocities=velocities_out,
        masses=masses,
    )


def run_ias15_stepped(
    system: System,
    total_time: float,
    sample_interval: float,
    dt: float,
) -> Trajectory:
    """
    Pure IAS15, but forced to stop at EVERY multiple of `dt`, exactly the way the
    adaptive integrator steps its IAS15 stretches (integrate to step*dt with
    exact_finish_time), sampled on the same grid as run_ias15.

    This is the CHAOS-FLOOR CONTROL: run_ias15 stops only at sample times, so it
    and this run are two equally valid numerical solutions that differ only by
    rounding-level noise (~1e-15). In a chaotic system that noise grows
    exponentially, so the error of THIS run against run_ias15 is the best any
    integrator that matches IAS15 to rounding can score. An adaptive error that
    occurs where this control is still accurate is attributable to the algorithm;
    one that occurs where the control has failed too is not fixable.

    Args:
        system: The initial System.
        total_time: Total simulated duration.
        sample_interval: Time between recorded samples (a multiple of dt).
        dt: The step the adaptive run uses.

    Raises:
        ValueError: If an argument is non-positive or sample_interval is not a
            multiple of dt.
    """
    if sample_interval <= 0 or total_time <= 0 or dt <= 0:
        raise ValueError(
            f"total_time, sample_interval and dt must be positive, got {total_time}, {sample_interval}, {dt}"
        )
    steps_per_sample = int(round(sample_interval / dt))
    if steps_per_sample < 1 or abs(steps_per_sample * dt - sample_interval) > 1e-9 * sample_interval:
        raise ValueError(f"sample_interval ({sample_interval}) must be a multiple of dt ({dt})")
    n_steps = int(round(total_time / dt))
    n_samples = n_steps // steps_per_sample + 1

    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = system.g
    sim.softening = system.softening
    for body in system.bodies:
        sim.add(
            m=body.mass,
            x=body.position[0], y=body.position[1], z=body.position[2],
            vx=body.velocity[0], vy=body.velocity[1], vz=body.velocity[2],
        )

    n = system.n
    times_out = np.zeros(n_samples, dtype=np.float64)
    positions_out = np.zeros((n_samples, n, 3), dtype=np.float64)
    velocities_out = np.zeros((n_samples, n, 3), dtype=np.float64)

    def record(k: int, t: float) -> None:
        times_out[k] = t
        for i, particle in enumerate(sim.particles):
            positions_out[k, i] = (particle.x, particle.y, particle.z)
            velocities_out[k, i] = (particle.vx, particle.vy, particle.vz)

    record(0, 0.0)
    for step in range(1, n_steps + 1):
        sim.integrate(step * dt, exact_finish_time=1)
        if step % steps_per_sample == 0:
            k = step // steps_per_sample
            record(k, k * sample_interval)

    return Trajectory(
        times=times_out,
        positions=positions_out,
        velocities=velocities_out,
        masses=system.masses(),
    )
