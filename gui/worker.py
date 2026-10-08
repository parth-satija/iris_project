"""
gui/worker.py

Runs ONE experiment in ONE child process: only the adaptive integrator (core.adaptive.run_adaptive).
There is no IAS15-only or Leapfrog-only reference run, no oracle and no correction.

What it saves (in <run_dir>/traj/):
    exp_XXXX.npz   sampled states + what the integrator did at every sample
    exp_XXXX.json  summary numbers + metadata

Because there is no reference, accuracy is judged by conservation: the relative drift of total
energy and angular momentum at every sample.

The integrator switches for the WHOLE system (the system score is the worst body's score), so
"which integrator" is one value per sample. What differs per body is the risk: the per-body
scores are rebuilt after the run from the sampled states (jerk from a backward Taylor step), so
the viewer can show which body is driving each switch. That reconstruction is checked against the
score the integrator logged and the worst mismatch is stored.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PHASE_QUEUED, PHASE_RUNNING, PHASE_ANALYSING, PHASE_SAVING, PHASE_DONE = 0, 1, 2, 3, 4
PHASE_NAMES = {0: "queued", 1: "integrating", 2: "analysing", 3: "saving", 4: "done"}

PER_BODY_MAX_SAMPLES = 20000  # skip the per-body reconstruction for very long sample series


class _ProgressIndex:
    """
    Wraps the raw-index function: forwards every call and attribute, and reports the step number
    now and then. run_adaptive has no progress callback, but it calls this once per step.
    """

    def __init__(self, inner, report, n_steps: int):
        self._inner = inner
        self._report = report
        self._n = max(int(n_steps), 1)
        self._max = 0

    def __call__(self, features):
        step = features.step
        if step is not None and step > self._max:
            self._max = step
            if step % 500 == 0:
                self._report(min(step / self._n, 1.0))
        return self._inner(features)

    def __getattr__(self, name):  # configure / snapshot_state / restore_state of stateful indices
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


def _clean(obj):
    """Make a structure JSON-safe (NaN / inf -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def index_source_name(s: dict) -> str:
    return "smoke" if s["smoke_test_index"] else "situations" if s["situation_index"] else "formula"


def _per_body_scores(traj, cfg, raw_fn, scale, steps_per_sample: int):
    """Per-body 0-1 scores at every sample, rebuilt from the sampled states. Returns (T, N)."""
    from core.physics import compute_accelerations
    from core.safety_index import compute_safety_features, normalize_safety_index

    pos, vel, m = traj.positions, traj.velocities, traj.masses
    t_count, n = pos.shape[0], pos.shape[1]
    out = np.full((t_count, n), np.nan)
    dt = cfg.dt
    for k in range(1, t_count):
        p, v = pos[k], vel[k]
        a = compute_accelerations(p, m, cfg.g, cfg.softening)
        p_prev = p - v * dt + 0.5 * a * dt * dt  # state one dt earlier (2nd-order Taylor)
        a_prev = compute_accelerations(p_prev, m, cfg.g, cfg.softening)
        feats = compute_safety_features(p, v, a, a_prev, dt, m, step=k * steps_per_sample)
        out[k] = normalize_safety_index(raw_fn(feats), scale)
    return out


def _conservation(traj, cfg):
    """Relative drift of total energy and angular momentum at every sample."""
    from core.physics import compute_angular_momentum, compute_total_energy

    pos, vel, m = traj.positions, traj.velocities, traj.masses
    t_count = pos.shape[0]
    e = np.array([compute_total_energy(pos[k], vel[k], m, cfg.g, cfg.softening) for k in range(t_count)])
    lz = np.array([compute_angular_momentum(pos[k], vel[k], m) for k in range(t_count)])
    e_drift = np.abs(e - e[0]) / max(abs(e[0]), 1e-300)
    l0 = np.linalg.norm(lz[0])
    l_drift = np.linalg.norm(lz - lz[0], axis=1) / (l0 if l0 > 0 else 1.0)
    return e_drift, l_drift


def run_inline(config_path: str, run_dir: str, s: dict, idx: int, names: list[str], shared=None) -> dict:
    """Run one experiment in the calling process. Returns the result dict (raises on failure)."""
    from calibration.generator import load_experiment_config
    from core.adaptive import run_adaptive
    from core.physics import Body, System
    from core.safety_index import IndexScale, check_index_available, resolve_raw_index_fn

    def set_phase(p):
        if shared is not None:
            shared[2 * idx + 1] = p

    def report(frac):
        if shared is not None:
            shared[2 * idx] = frac

    cfg = load_experiment_config(config_path)
    source = index_source_name(s)
    ok, msg = check_index_available(resolve_raw_index_fn(source))
    if not ok:
        raise RuntimeError(f"safety index '{source}' is not usable: {msg}")
    raw_fn = resolve_raw_index_fn(source)
    scale = IndexScale(s["index_scale"], s["index_lo"], s["index_hi"])

    system = System(
        [Body(cfg.masses[i], cfg.positions[i], cfg.velocities[i], names[i] if i < len(names) else str(i + 1))
         for i in range(cfg.body_count)],
        g=cfg.g, softening=cfg.softening,
    )
    n_steps = int(round(cfg.total_time / cfg.dt))
    steps_per_sample = int(round(cfg.sample_interval / cfg.dt))
    drift_budget = s["drift_budget"]
    fp_threshold = None
    if s["detect_false_positives"]:
        fp_threshold = s["false_positive_threshold"] or cfg.correction_threshold

    set_phase(PHASE_RUNNING)
    report(0.0)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    traj, log = run_adaptive(
        system, cfg.dt, cfg.total_time, cfg.sample_interval,
        safety_threshold=s["safety_threshold"], index_scale=scale,
        raw_index_fn=_ProgressIndex(raw_fn, report, n_steps),
        release_fraction=s["release_fraction"],
        rewind=s["rewind"], checkpoint_count=s["checkpoint_count"],
        checkpoint_interval=s["checkpoint_interval"], ias15_hold_steps=s["ias15_hold_steps"],
        rewind_back=s["rewind_back"],
        detect_false_positives=s["detect_false_positives"], false_positive_threshold=fp_threshold,
        drift_budget=drift_budget, drift_error_threshold=(cfg.correction_threshold if drift_budget > 0 else None),
        rewind_to_anchor=s["rewind_to_anchor"], resync_on_switch=s["resync_on_switch"],
        richardson_levels=s["richardson_levels"], compensated_sums=s["compensated_sums"],
        # deliberately NOT passed: reference / error_threshold (correction), oracle (rollback analysis)
    )
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    report(1.0)

    set_phase(PHASE_ANALYSING)
    t_count = traj.positions.shape[0]
    e_drift, l_drift = _conservation(traj, cfg)

    body_score = None
    mismatch = None
    if source != "situations" and t_count <= PER_BODY_MAX_SAMPLES:
        body_score = _per_body_scores(traj, cfg, resolve_raw_index_fn(source), scale, steps_per_sample)
        worst = np.full(t_count, np.nan)
        valid = ~np.isnan(body_score).all(axis=1)  # sample 0 has no jerk, so no score
        worst[valid] = np.nanmax(body_score[valid], axis=1)
        diff = np.abs(worst[1:] - np.asarray(log.safety_score, dtype=float)[1:])
        mismatch = float(np.nanmax(diff)) if np.isfinite(diff).any() else None

    set_phase(PHASE_SAVING)
    active = np.array([a == "ias15" for a in log.active_integrator], dtype=bool)
    sim_id = cfg.simulation_id
    traj_dir = os.path.join(run_dir, "traj")
    os.makedirs(traj_dir, exist_ok=True)
    base = os.path.join(traj_dir, f"exp_{idx:04d}")

    arrays = dict(
        times=traj.times, masses=traj.masses, positions=traj.positions, velocities=traj.velocities,
        active=active, ias15_fraction=np.asarray(log.ias15_step_fraction, dtype=float),
        score=np.asarray(log.safety_score, dtype=float), raw_index=np.asarray(log.raw_index, dtype=float),
        energy_drift=e_drift, momentum_drift=l_drift,
        switches=np.array([[e.step, e.time, 1.0 if e.to_integrator == "ias15" else 0.0, e.score]
                           for e in log.switch_events], dtype=float).reshape(-1, 4),
        rewinds=np.array([[e.step, e.to_step, e.time, e.to_time, e.score, e.depth]
                          for e in log.rewind_events], dtype=float).reshape(-1, 6),
        resyncs=np.array([[e.step, e.time, e.estimated_error, 1.0 if e.kind == "budget" else 0.0]
                          for e in log.resync_events], dtype=float).reshape(-1, 4),
    )
    if body_score is not None:
        arrays["body_score"] = body_score
    np.savez_compressed(base + ".npz", **arrays)

    n_fp = sum(1 for st in log.stretches if st.false_positive) if log.stretches else 0
    summary = {
        "simulation_id": sim_id, "n_bodies": cfg.body_count, "n_samples": int(t_count),
        "n_steps": int(log.n_steps), "n_ias15_steps": int(log.n_ias15_steps),
        "ias15_fraction": log.n_ias15_steps / max(log.n_steps, 1),
        "n_switches": len(log.switch_events), "n_rewinds": int(log.n_rewinds),
        "n_rewound_steps": int(log.n_rewound_steps), "n_resyncs": int(log.n_resyncs),
        "n_stretches": len(log.stretches), "n_false_positive_stretches": n_fp,
        "max_energy_drift": float(np.nanmax(e_drift)), "final_energy_drift": float(e_drift[-1]),
        "max_momentum_drift": float(np.nanmax(l_drift)),
        "wall_s": wall, "cpu_s": cpu, "per_body_mismatch": mismatch,
    }
    meta = {
        "index": idx, "simulation_id": sim_id, "names": list(names), "summary": summary,
        "safety_threshold": s["safety_threshold"], "release_fraction": s["release_fraction"],
        "dt": cfg.dt, "total_time": cfg.total_time, "sample_interval": cfg.sample_interval,
        "g": cfg.g, "softening": cfg.softening, "index_source": source,
        "has_body_score": body_score is not None,
    }
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(_clean(meta), fh, indent=2)
    set_phase(PHASE_DONE)
    return {"idx": idx, "ok": True, "simulation_id": sim_id, "meta_path": base + ".json",
            "summary": _clean(summary)}


def run_experiment(config_path, run_dir, s, idx, names, shared, queue) -> None:
    """Child-process entry point: run one experiment and put the result on `queue`."""
    try:
        queue.put(run_inline(config_path, run_dir, s, idx, names, shared))
    except BaseException as exc:  # report everything, including KeyboardInterrupt in the child
        queue.put({"idx": idx, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                   "trace": traceback.format_exc()})
