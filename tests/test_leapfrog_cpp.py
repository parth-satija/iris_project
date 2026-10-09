"""
tests/test_leapfrog_cpp.py

Parity test: core/leapfrog_cpp.py (C++) vs core/leapfrog.py, plain and with correction, including the pre_check_*
arrays and the CorrectionEvent list; plus the adaptive loop with a reference (--correct).

    python tests/test_leapfrog_cpp.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adaptive import run_adaptive
from core.adaptive_cpp import cpp_available, cpp_unavailable_reason, run_adaptive_fast
from core.leapfrog import run_leapfrog, run_leapfrog_with_correction
from core.leapfrog_cpp import run_leapfrog_fast, run_leapfrog_with_correction_fast
from core.metrics import position_error
from core.rebound_reference import run_ias15
from core.safety_index import IndexScale, resolve_raw_index_fn
from tests.test_adaptive_cpp import make_system


def main() -> int:
    if not cpp_available():
        print("C++ library not available:", cpp_unavailable_reason())
        return 2
    ok = True
    for n in (3, 4, 5):
        for seed in (11, 12):
            s = make_system(seed, n)
            T, dt, si = 4.0, 1e-3, 1e-2
            ref = run_ias15(s, total_time=T, sample_interval=si)
            a, b = run_leapfrog(s, dt, T, si), run_leapfrog_fast(s, dt, T, si)
            e1 = max(position_error(a.positions[k], b.positions[k]) for k in range(len(a.times)))
            ca, ea = run_leapfrog_with_correction(s, dt, T, ref.positions, ref.velocities, ref.times, 1e-6)
            cb, eb = run_leapfrog_with_correction_fast(s, dt, T, ref.positions, ref.velocities, ref.times, 1e-6)
            e2 = max(position_error(ca.positions[k], cb.positions[k]) for k in range(len(ca.times)))
            pre = max(np.nanmax(np.abs(getattr(ca, f) - getattr(cb, f)) / (1 + np.abs(getattr(ca, f))))
                      for f in ("pre_check_positions", "pre_check_velocities", "pre_check_acceleration", "pre_check_jerk"))
            key = lambda ev: [(x.correction_id, x.sample_index, x.step_index, x.steps_since_last_correction) for x in ev]
            same_events = key(ea) == key(eb) and np.array_equal(ca.times, cb.times)
            kw = dict(dt=dt, total_time=T, sample_interval=si, safety_threshold=0.9, index_scale=IndexScale("sigmoid"),
                      reference=ref, error_threshold=1e-6, rewind=True, resync_on_switch=True)
            x, lx = run_adaptive(s, raw_index_fn=resolve_raw_index_fn("formula"), **kw)
            y, ly = run_adaptive_fast(s, raw_index_fn=resolve_raw_index_fn("formula"), **kw)
            e3 = max(position_error(x.positions[k], y.positions[k]) for k in range(len(x.times)))
            adaptive_same = (lx.n_corrections, lx.n_ias15_steps, lx.n_rewinds) == (ly.n_corrections, ly.n_ias15_steps, ly.n_rewinds)
            good = e1 < 1e-9 and e2 < 1e-9 and pre < 1e-6 and same_events and adaptive_same and e3 < 1e-9
            ok &= good
            print(f"N={n} seed={seed} plain={e1:.1e} corrected={e2:.1e} pre_check={pre:.1e} corrections={len(ea)}/{len(eb)} "
                  f"| adaptive --correct err={e3:.1e} {'OK' if good else 'MISMATCH'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
