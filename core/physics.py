"""
core/physics.py

Newtonian N-body gravitational physics implemented from scratch using
vectorized NumPy operations. This module is integrator-agnostic: it only
knows how to compute accelerations from masses and positions. It is used
by both the custom Leapfrog integrator (core/leapfrog.py) and the
comparison/metrics code (core/metrics.py).

All state is stored as float64 NumPy arrays for numerical reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Default gravitational constant in SI-like "simulation units".
# Callers may override this via the `g` parameter of the functions below,
# or by constructing a System with a different `g` value.
DEFAULT_G: float = 1.0

# A small softening length (epsilon) squared, used to avoid singularities
# when two bodies pass arbitrarily close to one another. Set to 0.0 for
# exact Newtonian gravity (default), override for softened gravity.
DEFAULT_SOFTENING: float = 0.0


@dataclass
class Body:
    """
    A single point-mass body in the simulation.

    Attributes:
        mass: Scalar mass of the body (float64).
        position: shape-(3,) float64 array, [x, y, z].
        velocity: shape-(3,) float64 array, [vx, vy, vz].
        name: Optional human-readable identifier for logging/debugging.
    """

    mass: float
    position: np.ndarray
    velocity: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        self.mass = float(self.mass)
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(3)


@dataclass
class System:
    """
    A collection of bodies plus the physical constants that govern them.

    Attributes:
        bodies: List of Body instances.
        g: Gravitational constant used for force calculations.
        softening: Softening length (epsilon), used as epsilon^2 inside
            the denominator of the force law to avoid singularities at
            r -> 0. Defaults to 0.0 (unsoftened, exact Newtonian gravity).
    """

    bodies: list[Body]
    g: float = DEFAULT_G
    softening: float = DEFAULT_SOFTENING

    @property
    def n(self) -> int:
        """Number of bodies in the system."""
        return len(self.bodies)

    def masses(self) -> np.ndarray:
        """Return shape-(N,) float64 array of body masses."""
        return np.array([b.mass for b in self.bodies], dtype=np.float64)

    def positions(self) -> np.ndarray:
        """Return shape-(N, 3) float64 array of body positions."""
        return np.array([b.position for b in self.bodies], dtype=np.float64)

    def velocities(self) -> np.ndarray:
        """Return shape-(N, 3) float64 array of body velocities."""
        return np.array([b.velocity for b in self.bodies], dtype=np.float64)

    def set_state(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """
        Overwrite the positions and velocities of every body in place.

        Args:
            positions: shape-(N, 3) array of new positions.
            velocities: shape-(N, 3) array of new velocities.
        """
        for i, body in enumerate(self.bodies):
            body.position = np.asarray(positions[i], dtype=np.float64).reshape(3)
            body.velocity = np.asarray(velocities[i], dtype=np.float64).reshape(3)

    def copy(self) -> "System":
        """Return a deep-enough copy of the system (new Body/array objects)."""
        new_bodies = [
            Body(mass=b.mass, position=b.position.copy(), velocity=b.velocity.copy(), name=b.name)
            for b in self.bodies
        ]
        return System(bodies=new_bodies, g=self.g, softening=self.softening)


def pairwise_separations(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute all pairwise displacement vectors and distances between bodies.

    Args:
        positions: shape-(N, 3) array of body positions.

    Returns:
        Tuple of:
            diff: shape-(N, N, 3) array where diff[i, j] = positions[j] - positions[i]
                  (vector pointing FROM body i TOWARD body j).
            dist: shape-(N, N) array of Euclidean distances, dist[i, i] = 0.
    """
    # Broadcasting: positions[None, :, :] - positions[:, None, :]
    # gives diff[i, j] = positions[j] - positions[i]
    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dist = np.linalg.norm(diff, axis=2)
    return diff, dist


def compute_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    g: float = DEFAULT_G,
    softening: float = DEFAULT_SOFTENING,
) -> np.ndarray:
    """
    Compute the gravitational acceleration on every body due to every
    other body, via vectorized pairwise Newtonian gravity:

        a_i = G * sum_{j != i} m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)

    Args:
        positions: shape-(N, 3) float64 array of positions.
        masses: shape-(N,) float64 array of masses.
        g: Gravitational constant.
        softening: Softening length epsilon (0.0 for exact Newtonian gravity).

    Returns:
        shape-(N, 3) float64 array of accelerations, one per body.
    """
    n = positions.shape[0]
    diff, dist = pairwise_separations(positions)  # diff[i,j] = r_j - r_i

    eps2 = softening * softening
    # denom[i,j] = (dist(i,j)^2 + eps^2)^{3/2}; set diagonal to 1 to avoid /0,
    # the self-term is zeroed out explicitly afterward via mass-weighted diff=0.
    denom = np.power(dist**2 + eps2, 1.5)
    np.fill_diagonal(denom, 1.0)  # placeholder, self-contribution is exactly zero below

    # inv_denom[i,j] = 1 / denom[i,j], with diagonal irrelevant since diff[i,i] = 0
    inv_denom = 1.0 / denom

    # acc_contrib[i,j,:] = m_j * diff[i,j,:] * inv_denom[i,j]
    acc_contrib = masses[np.newaxis, :, np.newaxis] * diff * inv_denom[:, :, np.newaxis]

    accelerations = g * np.sum(acc_contrib, axis=1)
    return accelerations


def compute_potential_energy(
    positions: np.ndarray,
    masses: np.ndarray,
    g: float = DEFAULT_G,
    softening: float = DEFAULT_SOFTENING,
) -> float:
    """
    Compute total gravitational potential energy of the system:

        U = -G * sum_{i < j} m_i * m_j / sqrt(|r_i - r_j|^2 + eps^2)

    Args:
        positions: shape-(N, 3) array of positions.
        masses: shape-(N,) array of masses.
        g: Gravitational constant.
        softening: Softening length epsilon.

    Returns:
        Scalar total potential energy (float).
    """
    n = positions.shape[0]
    _, dist = pairwise_separations(positions)
    eps2 = softening * softening
    inv_r = np.zeros_like(dist)
    # Only compute for i < j to avoid double counting and the i == j singularity.
    iu = np.triu_indices(n, k=1)
    inv_r[iu] = 1.0 / np.sqrt(dist[iu] ** 2 + eps2)
    mass_prod = np.outer(masses, masses)
    potential = -g * np.sum(mass_prod[iu] * inv_r[iu])
    return float(potential)


def compute_kinetic_energy(velocities: np.ndarray, masses: np.ndarray) -> float:
    """
    Compute total kinetic energy: KE = sum_i 0.5 * m_i * |v_i|^2

    Args:
        velocities: shape-(N, 3) array of velocities.
        masses: shape-(N,) array of masses.

    Returns:
        Scalar total kinetic energy (float).
    """
    speeds_sq = np.sum(velocities**2, axis=1)
    return float(0.5 * np.sum(masses * speeds_sq))


def compute_total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    g: float = DEFAULT_G,
    softening: float = DEFAULT_SOFTENING,
) -> float:
    """
    Compute total mechanical energy E = KE + U.

    Args:
        positions: shape-(N, 3) array of positions.
        velocities: shape-(N, 3) array of velocities.
        masses: shape-(N,) array of masses.
        g: Gravitational constant.
        softening: Softening length epsilon.

    Returns:
        Scalar total energy (float).
    """
    ke = compute_kinetic_energy(velocities, masses)
    pe = compute_potential_energy(positions, masses, g=g, softening=softening)
    return ke + pe


def compute_angular_momentum(
    positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray
) -> np.ndarray:
    """
    Compute total system angular momentum about the origin:

        L = sum_i m_i * (r_i x v_i)

    Args:
        positions: shape-(N, 3) array of positions.
        velocities: shape-(N, 3) array of velocities.
        masses: shape-(N,) array of masses.

    Returns:
        shape-(3,) float64 array, the total angular momentum vector.
    """
    per_body_l = masses[:, np.newaxis] * np.cross(positions, velocities)
    return np.sum(per_body_l, axis=0)


def compute_center_of_mass(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Compute the system's center of mass position.

    Args:
        positions: shape-(N, 3) array of positions.
        masses: shape-(N,) array of masses.

    Returns:
        shape-(3,) float64 array, the center-of-mass position.
    """
    total_mass = np.sum(masses)
    return np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass


def compute_total_momentum(velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Compute total linear momentum of the system: P = sum_i m_i * v_i

    Args:
        velocities: shape-(N, 3) array of velocities.
        masses: shape-(N,) array of masses.

    Returns:
        shape-(3,) float64 array, the total momentum vector.
    """
    return np.sum(velocities * masses[:, np.newaxis], axis=0)
