# C++ adaptive loop (REBOUND C API)

C++ port of the main loop of `core/adaptive.py::run_adaptive`. One call runs the whole simulation; nothing crosses
the Python/C++ boundary per step.

## Build

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --config Release
python tests/test_adaptive_cpp.py        # parity vs the Python loop, must print PASS
python cpp/bench/bench.py                # us/step breakdown
```

* REBOUND's C sources are downloaded and compiled in (tag `IRIS_REBOUND_TAG`, default **5.2.2**). Make it match
  `python -c "import rebound; print(rebound.__version__)"`, or pass `-DIRIS_REBOUND_DIR=<local checkout>`.
  The struct layout of IAS15 is read directly, so a different REBOUND major version may not compile.
* Windows: use MSVC (or clang-cl). MinGW is untested (REBOUND's WHFast512 assembly path).
* `-DIRIS_NATIVE=OFF` if you copy the binary to another CPU.
* Use from Python: `from core.adaptive_cpp import run_adaptive_fast` (same signature as `run_adaptive`). Library lookup:
  `$IRIS_CPP_LIB`, else `cpp/build/**`. If the library is missing, or a feature is not ported, it silently calls the
  Python `run_adaptive`.
* To switch the validation pipeline, in `validation/runner.py` import `run_adaptive_fast as run_adaptive`.

## What is ported

Switching with release hysteresis and hold steps; rewinding (`rewind_back`, `rewind_to_anchor`, checkpoint ring,
forced IAS15 replay); drift budget; `resync_on_switch`; Richardson levels 2/3; Kahan sums; snapping to a reference
(`--correct`); the formula index, the smoke index, and the stateful situations index (with snapshot/restore for rewinds).

**Not ported** (validation-only analysis, falls back to Python): `detect_false_positives`, `analyze_rollback`, custom
raw-index callables, N > 16. If you refit `raw_safety_index`, paste the new coefficients into `FORMULA_COEF` in
`core/adaptive_cpp.py` (the test fails if they go stale); a refit situations weight set is read automatically.

## Where the speed comes from

* Compile-time body count (N = 2..5 specialised, runtime fallback up to 16): unrolled force loops, state on the stack,
  no allocation in the step loop.
* Pair-symmetric forces; the nearest-neighbour search of the safety features is a by-product of the force pass.
* The safety score is compared in raw-logit space against a precomputed threshold (found by bisection on the exact
  Python score function): no sigmoid or `exp` per step. Squared magnitudes, so no `sqrt` for the features.
* IAS15: one persistent REBOUND simulation, reset in place at a switch (state, time, step guess, b/e predictors,
  compensated-sum carries: indistinguishable from a fresh sim), stepped through `integrator.callbacks.step` with a
  replica of `exact_finish_time` (`Ias15::advance_to`): no `signal()`/`gettimeofday`/heartbeat per dt. Verified
  bit-identical to `reb_simulation_integrate`.
* Situations index evaluated only on sample steps, as in Python; no per-sample allocation.

## Measured (sandbox, one core, N-body systems with close encounters, dt = 1e-3, 500 samples)

| | N=3 | N=5 |
|---|---|---|
| Leapfrog only + formula index | 0.2-0.3 us/step | 0.26-0.28 |
| Leapfrog only + situations index | 0.28 | 0.66 |
| IAS15 stepped at every dt | ~8.6 | ~14.5 |

IAS15 itself is REBOUND's algorithm and dominates: this port removes call overhead (about 8-12% per IAS15 step) but
does not make IAS15 faster. The adaptive average is `f * t_ias15 + (1 - f) * t_leapfrog`, so **sub-2 us/step holds only
while the IAS15 share `f` stays below roughly 15-25%** (it was met in the benchmark at ~2-10% share, missed at 28-100%).
`f` is set by the safety threshold and the system, not by this code. Your 6 us IAS15 figure leaves more room than this
sandbox's CPU; run `cpp/bench/bench.py` on your machine.

Parity: 36 random N=3..5 cases (both indices, rewind/anchor/drift/resync/Kahan/level 3) give identical switch steps and
IAS15/rewind/resync counts, trajectories within 1e-12 to 1e-16 (not bit-identical; chaos amplifies roundoff).

## Known limits / next steps

* Situations index percentile refresh is O(history) every 50 samples: negligible at 500 samples, but at ~5000 samples
  it adds ~2 us/step for N=5. An incremental order-statistics structure would fix it.
* `iris_run` per-call setup (REBOUND create, ring allocation) is ~tens of us; irrelevant per run, but not per step.
