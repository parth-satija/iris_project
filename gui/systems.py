"""
gui/systems.py

Where the simulated systems come from: random ones (same generator and seeds as
validate_adaptive.py) or a custom one placed by hand in the Setup tab.

A custom system is a plain dict, so it can live in the settings JSON:

    {"g": 1.0, "softening": 0.0,
     "bodies": [{"name": "Star", "mass": 100.0, "pos": [0, 0, 0], "vel": [0, 0, 0]}, ...]}

Both sources are turned into ExperimentConfig JSON files (the format the rest of the project
already uses), which the worker processes read.
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict

import numpy as np

MAX_BODIES = 12
DEFAULT_NEW_MASS = 1.0


# --------------------------------------------------------------------------- helpers
def body(name: str, mass: float, pos, vel) -> dict:
    return {"name": name, "mass": float(mass), "pos": [float(x) for x in pos], "vel": [float(x) for x in vel]}


def system(bodies: list[dict], g: float = 1.0, softening: float = 0.0) -> dict:
    return {"g": float(g), "softening": float(softening), "bodies": bodies}


def center_system(cs: dict) -> dict:
    """Shift to the centre-of-mass frame (COM at the origin, total momentum zero). Returns a copy."""
    out = copy.deepcopy(cs)
    bodies = out["bodies"]
    m = np.array([b["mass"] for b in bodies], dtype=float)
    if m.sum() <= 0:
        return out
    pos = np.array([b["pos"] for b in bodies], dtype=float)
    vel = np.array([b["vel"] for b in bodies], dtype=float)
    pos -= (m[:, None] * pos).sum(axis=0) / m.sum()
    vel -= (m[:, None] * vel).sum(axis=0) / m.sum()
    for b, p, v in zip(bodies, pos, vel):
        b["pos"], b["vel"] = [float(x) for x in p], [float(x) for x in v]
    return out


def circular_speed(g: float, central_mass: float, r: float) -> float:
    return math.sqrt(g * central_mass / r)


# --------------------------------------------------------------------------- presets
def _figure_eight() -> dict:
    # Chenciner-Montgomery figure-eight choreography (G = 1, equal masses 1, period ~6.3259).
    p = (0.97000436, -0.24308753, 0.0)
    v3 = (-0.93240737, -0.86473146, 0.0)
    v12 = (-v3[0] / 2, -v3[1] / 2, 0.0)
    return system([
        body("A", 1.0, p, v12),
        body("B", 1.0, (-p[0], -p[1], 0.0), v12),
        body("C", 1.0, (0.0, 0.0, 0.0), v3),
    ])


def _pythagorean() -> dict:
    # Burrau's Pythagorean problem: masses 3, 4, 5 at rest at the corners of a 3-4-5 triangle.
    # Chaotic, with several very close encounters: a good stress test for the switching.
    return system([
        body("m=3", 3.0, (1.0, 3.0, 0.0), (0, 0, 0)),
        body("m=4", 4.0, (-2.0, -1.0, 0.0), (0, 0, 0)),
        body("m=5", 5.0, (1.0, -1.0, 0.0), (0, 0, 0)),
    ])


def _star_and_planets() -> dict:
    big = 100.0
    g = 1.0
    tilt = math.radians(25.0)
    planets = []
    for name, r, m, inc, sign in (("Inner", 8.0, 0.5, 0.0, 1.0), ("Middle", 14.0, 0.5, tilt, 1.0),
                                  ("Outer", 22.0, 0.5, -tilt, -1.0)):
        v = sign * circular_speed(g, big + m, r)
        planets.append(body(name, m, (r, 0.0, 0.0), (0.0, v * math.cos(inc), v * math.sin(inc))))
    return center_system(system([body("Star", big, (0, 0, 0), (0, 0, 0))] + planets, g=g))


def _binary_intruder() -> dict:
    g = 1.0
    m = 5.0
    d = 2.0
    v = 0.5 * math.sqrt(2 * g * m / d)
    return center_system(system([
        body("Binary 1", m, (-d / 2, 0.0, 0.0), (0.0, -v, 0.0)),
        body("Binary 2", m, (d / 2, 0.0, 0.0), (0.0, v, 0.0)),
        body("Intruder", 2.0, (18.0, 1.5, 3.0), (-3.0, 0.0, -0.15)),
    ], g=g))


PRESETS = {
    "Figure-eight (3 bodies)": _figure_eight,
    "Pythagorean 3-4-5 (chaotic)": _pythagorean,
    "Star + 3 tilted planets": _star_and_planets,
    "Tight binary + 3D intruder": _binary_intruder,
}


def random_system(seed: int, n_bodies: int) -> dict:
    """The random system validate_adaptive.py would generate for this seed and body count."""
    from calibration.generator import generate_experiment_config

    cfg = generate_experiment_config("preview", seed=seed, body_count=n_bodies)
    return system(
        [body(str(i + 1), cfg.masses[i], cfg.positions[i], cfg.velocities[i]) for i in range(cfg.body_count)],
        g=cfg.g, softening=cfg.softening,
    )


def default_custom_system() -> dict:
    return _figure_eight()


# --------------------------------------------------------------------------- validation
def validate_custom(cs: dict) -> list[str]:
    errs: list[str] = []
    bodies = cs.get("bodies", [])
    if len(bodies) < 2:
        errs.append("Place at least 2 bodies.")
    if len(bodies) > MAX_BODIES:
        errs.append(f"At most {MAX_BODIES} bodies are supported.")
    if not (cs.get("g", 0) > 0 and math.isfinite(cs.get("g", 0))):
        errs.append("G must be positive.")
    if not (cs.get("softening", 0) >= 0 and math.isfinite(cs.get("softening", 0))):
        errs.append("Softening must be >= 0.")
    for i, b in enumerate(bodies):
        label = f"Body {i + 1}"
        if not (b["mass"] > 0 and math.isfinite(b["mass"])):
            errs.append(f"{label}: mass must be positive.")
        if not all(math.isfinite(x) for x in list(b["pos"]) + list(b["vel"])):
            errs.append(f"{label}: position and velocity must be finite numbers.")
    pos = np.array([b["pos"] for b in bodies], dtype=float) if bodies else np.zeros((0, 3))
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if np.linalg.norm(pos[i] - pos[j]) < 1e-9:
                errs.append(f"Bodies {i + 1} and {j + 1} are at the same position.")
    return errs


# --------------------------------------------------------------------------- configs
def _write_config(cfg, configs_dir: str) -> str:
    os.makedirs(configs_dir, exist_ok=True)
    path = os.path.join(configs_dir, f"{cfg.simulation_id}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    return path


def build_configs(s: dict, run_dir: str) -> tuple[list[str], list[list[str]]]:
    """
    Write the ExperimentConfig JSON files for this run.
    Returns (config paths, body names per config).
    """
    from calibration.generator import ExperimentConfig, generate_calibration_batch, load_experiment_config

    configs_dir = os.path.join(run_dir, "configs")
    dt = s["dt"]
    sample_interval = s["sample_every"] * dt

    if s["system_source"] == "custom":
        cs = s["custom_system"]
        bodies = cs["bodies"]
        cfg = ExperimentConfig(
            simulation_id="custom", seed=0, body_count=len(bodies), g=cs["g"], softening=cs["softening"],
            dt=dt, total_time=s["total_time"], sample_interval=sample_interval,
            correction_threshold=s["correction_threshold"],
            masses=[b["mass"] for b in bodies], positions=[list(b["pos"]) for b in bodies],
            velocities=[list(b["vel"]) for b in bodies],
        )
        return [_write_config(cfg, configs_dir)], [[b["name"] for b in bodies]]

    paths = generate_calibration_batch(
        n_experiments=s["n_experiments"], configs_dir=configs_dir, base_seed=s["seed"], dt=dt,
        total_time=s["total_time"], correction_threshold=s["correction_threshold"],
        body_counts=(3,) if s["three_body"] else None,
    )
    names = []
    for p in paths:
        cfg = load_experiment_config(p)
        cfg.sample_interval = sample_interval  # same system, GUI-chosen frame spacing
        _write_config(cfg, configs_dir)
        names.append([str(i + 1) for i in range(cfg.body_count)])
    return paths, names
