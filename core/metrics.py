"""
core/metrics.py

Low-level, per-timestep physical metrics used by calibration/comparer.py
to build the full comparison DataFrame. Every function here operates on
a single timestep's worth of state (positions/velocities for all bodies)
and returns either a scalar or a small array. comparer.py is responsible
for looping over timesteps and assembling results into a table.

Kept separate from core/physics.py because physics.py implements the
*laws* (forces, energy, momentum) used by the integrators themselves,
while this module implements *diagnostics* derived from trajectories
(errors between two trajectories, finite-difference jerk, drift, etc.).
"""

from __future__ import annotations

import numpy as np

from core.physics import (
    compute_accelerations,
    compute_angular_momentum,
    compute_total_energy,
)


def position_error(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """
    Compute the RMS position error between two integrators' states at a
    single timestep, defined as the root-mean-square of per-body position
    differences:

        err = sqrt( mean_i( |pos_a[i] - pos_b[i]|^2 ) )

    Args:
        pos_a: shape-(N, 3) array of positions (e.g. Leapfrog).
        pos_b: shape-(N, 3) array of positions (e.g. IAS15 reference).

    Returns:
        Scalar RMS position error.
    """
    diff = pos_a - pos_b
    per_body_sq = np.sum(diff**2, axis=1)
    return float(np.sqrt(np.mean(per_body_sq)))


def velocity_error(vel_a: np.ndarray, vel_b: np.ndarray) -> float:
    """
    Compute the RMS velocity error between two integrators' states at a
    single timestep (same convention as position_error).

    Args:
        vel_a: shape-(N, 3) array of velocities (e.g. Leapfrog).
        vel_b: shape-(N, 3) array of velocities (e.g. IAS15 reference).

    Returns:
        Scalar RMS velocity error magnitude.
    """
    diff = vel_a - vel_b
    per_body_sq = np.sum(diff**2, axis=1)
    return float(np.sqrt(np.mean(per_body_sq)))


def min_pairwise_distance(positions: np.ndarray) -> float:
    """
    Compute the minimum pairwise distance between any two distinct bodies
    in the system at a single timestep.

    Args:
        positions: shape-(N, 3) array of positions.

    Returns:
        Scalar minimum distance (float). For N < 2, returns np.inf.
    """
    n = positions.shape[0]
    if n < 2:
        return float("inf")
    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    return float(np.min(dist[iu]))


def closest_pair_relative_velocity(positions: np.ndarray, velocities: np.ndarray) -> float:
    """
    Find the closest pair of bodies by position, and return the magnitude
    of their relative velocity.

    Args:
        positions: shape-(N, 3) array of positions.
        velocities: shape-(N, 3) array of velocities.

    Returns:
        Scalar relative velocity magnitude between the closest-approaching
        pair of bodies. For N < 2, returns 0.0.
    """
    n = positions.shape[0]
    if n < 2:
        return 0.0
    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    flat_dists = dist[iu]
    min_idx = int(np.argmin(flat_dists))
    i, j = iu[0][min_idx], iu[1][min_idx]
    rel_v = velocities[i] - velocities[j]
    return float(np.linalg.norm(rel_v))


def max_acceleration_magnitude(
    positions: np.ndarray, masses: np.ndarray, g: float, softening: float = 0.0
) -> float:
    """
    Compute the maximum per-body gravitational acceleration magnitude in
    the system at a single timestep.

    Args:
        positions: shape-(N, 3) array of positions.
        masses: shape-(N,) array of masses.
        g: Gravitational constant.
        softening: Softening length epsilon.

    Returns:
        Scalar maximum acceleration magnitude across all bodies.
    """
    acc = compute_accelerations(positions, masses, g=g, softening=softening)
    mags = np.linalg.norm(acc, axis=1)
    return float(np.max(mags))


def jerk_finite_difference(acc_prev: np.ndarray, acc_next: np.ndarray, dt: float) -> float:
    """
    Estimate the maximum jerk (time derivative of acceleration) magnitude
    across all bodies via a central finite difference:

        jerk_i ~= (acc_next[i] - acc_prev[i]) / dt

    Args:
        acc_prev: shape-(N, 3) array of accelerations at the previous sample.
        acc_next: shape-(N, 3) array of accelerations at the next sample.
        dt: Time separating acc_prev and acc_next.

    Returns:
        Scalar maximum jerk magnitude across all bodies. Returns 0.0 if
        dt <= 0 (e.g. at a boundary sample with no defined neighbor).
    """
    if dt <= 0:
        return 0.0
    jerk = (acc_next - acc_prev) / dt
    mags = np.linalg.norm(jerk, axis=1)
    return float(np.max(mags))


def energy_drift(current_energy: float, initial_energy: float) -> float:
    """
    Compute relative energy drift from the initial total energy:

        drift = (E(t) - E(0)) / |E(0)|

    Args:
        current_energy: Total energy at the current timestep.
        initial_energy: Total energy at t = 0.

    Returns:
        Scalar relative drift. If initial_energy is ~0, returns the raw
        (absolute) difference instead to avoid division by zero.
    """
    if abs(initial_energy) < 1e-300:
        return float(current_energy - initial_energy)
    return float((current_energy - initial_energy) / abs(initial_energy))


def angular_momentum_drift(current_l: np.ndarray, initial_l: np.ndarray) -> float:
    """
    Compute relative drift in the *magnitude* of total angular momentum:

        drift = (|L(t)| - |L(0)|) / |L(0)|

    Args:
        current_l: shape-(3,) angular momentum vector at current timestep.
        initial_l: shape-(3,) angular momentum vector at t = 0.

    Returns:
        Scalar relative drift. If |L(0)| is ~0, returns the raw (absolute)
        difference in magnitude instead to avoid division by zero.
    """
    mag_current = float(np.linalg.norm(current_l))
    mag_initial = float(np.linalg.norm(initial_l))
    if abs(mag_initial) < 1e-300:
        return mag_current - mag_initial
    return (mag_current - mag_initial) / abs(mag_initial)


def nearest_neighbor_per_body(
    positions: np.ndarray, velocities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute, for every body at a single timestep, the distance to its
    nearest other body and the relative velocity magnitude to that same
    neighbor. Same underlying pairwise-distance logic as
    calibration/events.py's per-trajectory nearest-neighbor computation,
    but for a single (positions, velocities) snapshot rather than a full
    (T, N, 3) trajectory -- meant for one-off snapshots such as the
    pre-correction state captured by core.leapfrog.CorrectionEvent, where
    running the full events pipeline would be overkill.

    Args:
        positions: shape-(N, 3) array of positions.
        velocities: shape-(N, 3) array of velocities.

    Returns:
        Tuple of (distance, relative_velocity, neighbor_id), each shape-(N,).
        For N < 2, distance is np.inf, relative_velocity is 0.0, and
        neighbor_id is -1 for every body.
    """
    n = positions.shape[0]
    if n < 2:
        return (
            np.full(n, np.inf),
            np.zeros(n),
            np.full(n, -1, dtype=int),
        )
    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    neighbor_id = np.argmin(dist, axis=1)
    distance = dist[np.arange(n), neighbor_id]
    vel_diff = velocities[np.newaxis, :, :] - velocities[:, np.newaxis, :]
    rel_speed = np.linalg.norm(vel_diff, axis=-1)
    relative_velocity = rel_speed[np.arange(n), neighbor_id]
    return distance, relative_velocity, neighbor_id.astype(int)


def mass_ratio(masses: np.ndarray) -> float:
    """
    Compute a single scalar summarizing the mass ratio of the system,
    defined as max(mass) / min(mass).

    Args:
        masses: shape-(N,) array of masses.

    Returns:
        Scalar ratio of largest to smallest mass. Returns 1.0 for N < 2.
    """
    if masses.size < 2:
        return 1.0
    return float(np.max(masses) / np.min(masses))
