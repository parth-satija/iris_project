"""
calibration/generator.py

Generates randomized, reproducible N-body initial conditions for
calibration experiments and serializes them to JSON configuration files
under experiments/configs/.

Every generated configuration carries the random seed used to create it,
so any experiment can be regenerated exactly (determinism requirement).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

# Supported body counts for randomized systems.
SUPPORTED_BODY_COUNTS: tuple[int, ...] = (2, 3, 4, 5)

# Default physical ranges used when randomizing systems. These are in
# arbitrary "simulation units" (consistent with DEFAULT_G in core/physics.py).
MASS_RANGE: tuple[float, float] = (0.5, 10.0)
RADIUS_RANGE: tuple[float, float] = (1.0, 8.0)
ECCENTRICITY_RANGE: tuple[float, float] = (0.0, 0.6)

# Default RMS position-error threshold (core.metrics.position_error units,
# i.e. same units as position) above which the online correction mechanism
# in core.leapfrog.run_leapfrog_with_correction snaps the Leapfrog state
# back to the IAS15 reference state. Configurable per experiment.
DEFAULT_CORRECTION_THRESHOLD: float = 1.0e-2


@dataclass
class ExperimentConfig:
    """
    Serializable description of a single calibration experiment's initial
    conditions and run parameters.

    Attributes:
        simulation_id: Unique string identifier for this experiment.
        seed: Random seed used to generate this configuration.
        body_count: Number of bodies (N) in the system.
        g: Gravitational constant to use for both integrators.
        softening: Softening length to use for both integrators.
        dt: Leapfrog timestep.
        total_time: Total simulated duration.
        sample_interval: Time between recorded samples (must be a multiple
            of dt so Leapfrog and IAS15 samples align exactly).
        correction_threshold: RMS position-error threshold above which the
            online correction mechanism snaps the Leapfrog state back to
            the IAS15 reference state (see core.leapfrog.run_leapfrog_with_correction).
        masses: List of body masses, length body_count.
        positions: List of [x, y, z] positions, length body_count.
        velocities: List of [vx, vy, vz] velocities, length body_count.
    """

    simulation_id: str
    seed: int
    body_count: int
    g: float
    softening: float
    dt: float
    total_time: float
    sample_interval: float
    correction_threshold: float
    masses: list[float]
    positions: list[list[float]]
    velocities: list[list[float]]

    def to_dict(self) -> dict:
        """Convert this configuration to a plain JSON-serializable dict."""
        return asdict(self)


def _random_orbital_system(
    rng: np.random.Generator,
    body_count: int,
    g: float,
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    """
    Build a randomized N-body system consisting of one central "anchor"
    body plus (body_count - 1) orbiting bodies placed on randomized,
    eccentricity-like Keplerian-inspired orbits with random 3D orientation.

    This is not a full Keplerian orbit solver -- it approximates plausible
    bound orbits by placing each orbiting body at a random radius and
    giving it a velocity close to the local circular velocity, scaled by
    a randomized "eccentricity-like" factor and rotated into a random
    orbital plane. This produces varied, physically reasonable dynamics
    (not exact conics) suitable for stress-testing the integrators.

    Args:
        rng: NumPy random Generator (already seeded by caller).
        body_count: Number of bodies to generate (including the anchor).
        g: Gravitational constant to assume for velocity scaling.

    Returns:
        Tuple of (masses, positions, velocities), each a plain Python list
        matching ExperimentConfig's field shapes.
    """
    masses: list[float] = []
    positions: list[list[float]] = []
    velocities: list[list[float]] = []

    # Central anchor body: heavier than the rest, placed at rest at origin.
    anchor_mass = float(rng.uniform(*MASS_RANGE) * 5.0)
    masses.append(anchor_mass)
    positions.append([0.0, 0.0, 0.0])
    velocities.append([0.0, 0.0, 0.0])

    orbiting_momentum = np.zeros(3, dtype=np.float64)

    for _ in range(body_count - 1):
        mass = float(rng.uniform(*MASS_RANGE))
        radius = float(rng.uniform(*RADIUS_RANGE))
        eccentricity = float(rng.uniform(*ECCENTRICITY_RANGE))

        # Random orientation: pick a random orbital plane via random
        # inclination/longitude-of-ascending-node-like Euler angles, plus
        # a random phase (true-anomaly-like angle) within the plane.
        inclination = float(rng.uniform(0.0, np.pi))
        raan = float(rng.uniform(0.0, 2.0 * np.pi))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))

        # Position in the orbital plane (local 2D coordinates).
        local_pos = np.array([radius * np.cos(phase), radius * np.sin(phase), 0.0])

        # Circular velocity magnitude for this radius around the anchor,
        # then reduced/increased by (1 - e) to emulate an eccentricity-like
        # perturbation away from a perfectly circular orbit.
        v_circ = np.sqrt(g * anchor_mass / radius)
        v_mag = v_circ * np.sqrt(max(1e-6, 1.0 - eccentricity))
        # Velocity direction perpendicular to the radius vector, in-plane.
        local_vel = np.array([-v_mag * np.sin(phase), v_mag * np.cos(phase), 0.0])

        # Rotation matrices: first by inclination about the x-axis, then by
        # raan (longitude of ascending node) about the z-axis. This gives a
        # uniformly randomized orbital plane orientation in 3D.
        rot_inc = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(inclination), -np.sin(inclination)],
                [0.0, np.sin(inclination), np.cos(inclination)],
            ]
        )
        rot_raan = np.array(
            [
                [np.cos(raan), -np.sin(raan), 0.0],
                [np.sin(raan), np.cos(raan), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotation = rot_raan @ rot_inc

        world_pos = rotation @ local_pos
        world_vel = rotation @ local_vel

        masses.append(mass)
        positions.append(world_pos.tolist())
        velocities.append(world_vel.tolist())

        orbiting_momentum += mass * world_vel

    # Give the anchor a small recoil velocity so total system momentum is
    # (very close to) zero, keeping the system's center of mass stationary.
    anchor_velocity = (-orbiting_momentum / anchor_mass).tolist()
    velocities[0] = anchor_velocity

    return masses, positions, velocities


def generate_experiment_config(
    simulation_id: str,
    seed: int,
    body_count: int,
    g: float = 1.0,
    softening: float = 0.0,
    dt: float = 1.0e-3,
    total_time: float = 5.0,
    sample_interval: float | None = None,
    correction_threshold: float = DEFAULT_CORRECTION_THRESHOLD,
) -> ExperimentConfig:
    """
    Generate a single randomized, reproducible experiment configuration.

    Args:
        simulation_id: Unique identifier for this experiment.
        seed: Random seed; the same seed always yields the same system.
        body_count: Number of bodies (must be in SUPPORTED_BODY_COUNTS).
        g: Gravitational constant for both integrators.
        softening: Softening length for both integrators.
        dt: Leapfrog integration timestep.
        total_time: Total simulated duration.
        sample_interval: Sampling interval; defaults to 50 * dt if omitted.
        correction_threshold: RMS position-error threshold that triggers
            an online correction (Leapfrog state snapped to IAS15 state).

    Returns:
        A populated ExperimentConfig instance.

    Raises:
        ValueError: If body_count is not in SUPPORTED_BODY_COUNTS.
    """
    if body_count not in SUPPORTED_BODY_COUNTS:
        raise ValueError(
            f"body_count must be one of {SUPPORTED_BODY_COUNTS}, got {body_count}"
        )

    if sample_interval is None:
        sample_interval = 50.0 * dt

    rng = np.random.default_rng(seed)
    masses, positions, velocities = _random_orbital_system(rng, body_count, g)

    return ExperimentConfig(
        simulation_id=simulation_id,
        seed=seed,
        body_count=body_count,
        g=g,
        softening=softening,
        dt=dt,
        total_time=total_time,
        sample_interval=sample_interval,
        correction_threshold=correction_threshold,
        masses=masses,
        positions=positions,
        velocities=velocities,
    )


def save_experiment_config(config: ExperimentConfig, configs_dir: str) -> str:
    """
    Serialize an ExperimentConfig to a JSON file inside `configs_dir`.

    Args:
        config: The ExperimentConfig to save.
        configs_dir: Directory to write the JSON file into (created if
            it does not exist).

    Returns:
        The full path to the written JSON file.
    """
    os.makedirs(configs_dir, exist_ok=True)
    path = os.path.join(configs_dir, f"{config.simulation_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
    return path


def load_experiment_config(path: str) -> ExperimentConfig:
    """
    Load an ExperimentConfig from a JSON file previously written by
    save_experiment_config.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        A reconstructed ExperimentConfig instance.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ExperimentConfig(**data)


def generate_calibration_batch(
    n_experiments: int,
    configs_dir: str,
    base_seed: int = 0,
    dt: float = 1.0e-3,
    total_time: float = 5.0,
    correction_threshold: float = DEFAULT_CORRECTION_THRESHOLD,
    body_counts: tuple[int, ...] | None = None,
) -> list[str]:
    """
    Generate a batch of `n_experiments` randomized configurations, cycling
    through the supported body counts, and save each to `configs_dir`.

    Args:
        n_experiments: Number of experiment configurations to generate.
        configs_dir: Directory to write JSON configuration files into.
        base_seed: Base random seed; experiment i uses seed = base_seed + i,
            guaranteeing each experiment is independently reproducible.
        dt: Leapfrog timestep shared by all generated experiments.
        total_time: Total simulated duration shared by all experiments.
        correction_threshold: RMS position-error threshold shared by all
            generated experiments; see ExperimentConfig.correction_threshold.
        body_counts: Optional subset of SUPPORTED_BODY_COUNTS to cycle through
            instead of all of them, e.g. (3,) for three-body systems only.
            Defaults to every supported body count.

    Returns:
        List of file paths to the generated JSON configuration files, in
        generation order.

    Raises:
        ValueError: If body_counts is empty or contains an unsupported count.
    """
    counts = SUPPORTED_BODY_COUNTS if body_counts is None else tuple(body_counts)
    if not counts or any(c not in SUPPORTED_BODY_COUNTS for c in counts):
        raise ValueError(
            f"body_counts must be a non-empty subset of {SUPPORTED_BODY_COUNTS}, got {body_counts}"
        )
    paths: list[str] = []
    for i in range(n_experiments):
        seed = base_seed + i
        body_count = counts[i % len(counts)]
        simulation_id = f"sim_{i:04d}_n{body_count}_seed{seed}"
        config = generate_experiment_config(
            simulation_id=simulation_id,
            seed=seed,
            body_count=body_count,
            dt=dt,
            total_time=total_time,
            correction_threshold=correction_threshold,
        )
        path = save_experiment_config(config, configs_dir)
        paths.append(path)
    return paths
