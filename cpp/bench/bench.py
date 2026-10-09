"""
cpp/bench/bench.py -- microseconds per step of the C++ adaptive loop, split into its parts.

    python cpp/bench/bench.py [--steps 5000] [--bodies 3 5]

"us/step" = wall time of ONE iris_run call / nominal steps (replayed steps after a rewind are extra work and ARE
counted in the time). The adaptive average is  f_ias15 * t_ias15 + (1 - f_ias15) * t_leapfrog,  so the 2 us target
holds only while the IAS15 share f_ias15 stays below about (2 - t_lf) / (t_ias15 - t_lf).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.adaptive_cpp import cpp_available, cpp_unavailable_reason, run_adaptive_fast
from core.physics import Body, System
from core.safety_index import IndexScale, resolve_raw_index_fn


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


def run(n: int, index: str, thr: float, steps: int, reps: int = 5, **kw):
    s = make_system(11, n)
    best, log = 1e9, None
    for _ in range(reps):
        t0 = time.perf_counter()
        _, log = run_adaptive_fast(s, dt=1e-3, total_time=steps * 1e-3, sample_interval=1e-2, safety_threshold=thr,
                                   index_scale=IndexScale("sigmoid"), raw_index_fn=resolve_raw_index_fn(index), **kw)
        best = min(best, time.perf_counter() - t0)
    return best / steps * 1e6, log.n_ias15_steps / steps, log.n_rewound_steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bodies", type=int, nargs="+", default=[3, 5])
    a = ap.parse_args()
    if not cpp_available():
        sys.exit(f"C++ library not available: {cpp_unavailable_reason()}")
    for n in a.bodies:
        print(f"--- N={n}, {a.steps} steps (dt=1e-3) ---")
        for idx in ("formula", "situations"):
            us, _, _ = run(n, idx, 1.0, a.steps)
            print(f"  Leapfrog only, {idx:10s} index        : {us:7.3f} us/step")
        us, _, _ = run(n, "formula", 0.0, min(a.steps, 3000))
        print(f"  IAS15 stepped every dt (100% IAS15)       : {us:7.3f} us/step")
        for idx, thr in (("formula", 0.9), ("situations", 0.011), ("situations", 0.0029)):
            us, f, rw = run(n, idx, thr, a.steps, rewind=True)
            print(f"  adaptive {idx:10s} thr={thr:<7} rewind: {us:7.3f} us/step  (IAS15 share {100 * f:5.1f}%, {rw} replayed)")


if __name__ == "__main__":
    main()
