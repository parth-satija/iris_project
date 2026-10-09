"""
tests/test_adaptive_cpp.py

Parity test: the C++ loop (core/adaptive_cpp.py -> cpp/) must reproduce core/adaptive.py::run_adaptive.

    python tests/test_adaptive_cpp.py

Asserts, for random 3/4/5-body systems with close encounters, with the formula and situations indices, with and
without rewinding / anchor rewinding / drift budget / resync-on-switch / Kahan / 6th-order Richardson:
  * identical integrator-switch sequence (step and direction), IAS15 step count, rewind count, resync count
  * trajectories equal to within TRAJ_TOL (they are not bit-identical: C++ evaluates pair-symmetric forces and
    compares the score in raw-index space, so results differ at the 1e-16 level, which chaos amplifies)
Also checks that FORMULA_COEF in core/adaptive_cpp.py still reproduces core.safety_index.raw_safety_index.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adaptive import run_adaptive
from core.adaptive_cpp import FORMULA_COEF, cpp_available, cpp_unavailable_reason, run_adaptive_fast
from core.metrics import position_error
from core.physics import Body, System
from core.safety_index import IndexScale, SafetyFeatures, raw_safety_index, resolve_raw_index_fn

TRAJ_TOL = 1e-9


def make_system(seed: int, n: int) -> System:
    rng = np.random.default_rng(seed)
    bodies = [Body(1.0, [0, 0, 0], [0, 0, 0])]
    for i in range(n - 1):
        r = 0.5 + 0.35 * i + 0.1 * rng.random()
        th = rng.random() * 6.28
        v = np.sqrt(1 / r) * (0.55 + 0.3 * rng.random())
        bodies.append(Body(2e-2 * (1 + rng.random()),
                           [r * np.cos(th), r * np.sin(th), 0.05 * rng.normal()],
                           [-v * np.sin(th), v * np.cos(th), 0.02 * rng.normal()]))
    return System(bodies)


def check_formula_coefficients() -> bool:
    rng = np.random.default_rng(0)
    f = SafetyFeatures(
        relative_velocity=10 ** rng.uniform(-3, 1, 4), nn_distance=10 ** rng.uniform(-3, 0, 4),
        acceleration=10 ** rng.uniform(-2, 3, 4), jerk=10 ** rng.uniform(-1, 4, 4),
        mass=10 ** rng.uniform(-4, 0, 4), mass_ratio=123.0)
    lg = f.logs()
    c = FORMULA_COEF
    mine = (c[0] + c[1] * lg["log_rel_velocity"] + c[2] * lg["log_jerk"] + c[3] * lg["log_acceleration"]
            + c[4] * lg["log_nn_distance"] + c[5] * lg["log_mass"] + c[6] * lg["log_mass_ratio"])
    ok = bool(np.allclose(mine, raw_safety_index(f), rtol=0, atol=1e-12))
    print(f"formula coefficients in adaptive_cpp.py match raw_safety_index: {'OK' if ok else 'STALE - update FORMULA_COEF'}")
    return ok


def main() -> int:
    if not cpp_available():
        print("C++ library not available:", cpp_unavailable_reason())
        return 2
    ok = check_formula_coefficients()
    cases = []
    for n in (3, 4, 5):
        for seed in (11, 12):
            cases += [
                (n, seed, "formula", dict(safety_threshold=0.9)),
                (n, seed, "formula", dict(safety_threshold=0.9, rewind=True, checkpoint_interval=7, rewind_back=2, ias15_hold_steps=3)),
                (n, seed, "formula", dict(safety_threshold=0.9, rewind=True, rewind_to_anchor=True, drift_budget=0.2,
                                          drift_error_threshold=1e-6, compensated_sums=True)),
                (n, seed, "formula", dict(safety_threshold=0.9, rewind=True, resync_on_switch=True, richardson_levels=3,
                                          compensated_sums=True)),
                (n, seed, "situations", dict(safety_threshold=0.0029, rewind=True)),
                (n, seed, "situations", dict(safety_threshold=0.011)),
            ]
    for n, seed, idx, kw in cases:
        s = make_system(seed, n)
        common = dict(dt=1e-3, total_time=4.0, sample_interval=1e-2, index_scale=IndexScale("sigmoid"), **kw)
        a, la = run_adaptive(s, raw_index_fn=resolve_raw_index_fn(idx), **common)
        b, lb = run_adaptive_fast(s, raw_index_fn=resolve_raw_index_fn(idx), **common)
        err = max(position_error(a.positions[k], b.positions[k]) for k in range(len(a.times)))
        same = (
            [(e.step, e.to_integrator) for e in la.switch_events] == [(e.step, e.to_integrator) for e in lb.switch_events]
            and la.n_ias15_steps == lb.n_ias15_steps and la.n_rewinds == lb.n_rewinds and la.n_resyncs == lb.n_resyncs
            and err < TRAJ_TOL
        )
        ok &= same
        print(f"N={n} seed={seed} {idx:10s} switches={len(la.switch_events):3d} ias15={la.n_ias15_steps:4d} "
              f"rewinds={la.n_rewinds} resyncs={la.n_resyncs:3d} maxdiff={err:.1e} {'OK' if same else 'MISMATCH'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
