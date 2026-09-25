# IRIS Calibration Framework (V1)

Research software for the IRIS project. This is **experiment #1**, not the
final orbital simulator. Its only job: run the same N-body system through
two integrators — a from-scratch **Velocity Verlet (Leapfrog)** integrator
and **IAS15** (via REBOUND, used as high-accuracy ground truth) — and
produce CSV calibration data comparing them.

This document is meant to get a new contributor from "cloned the repo" to
"comfortable modifying it" without having to read every file first.

---

## 1. Mental model

There are exactly three moving parts, and everything else is plumbing
around them:

1. **Physics** (`core/physics.py`) — the law of gravity. Given masses and
   positions, compute forces/energy/momentum. No notion of time or
   integration lives here.
2. **Two integrators** that both consume `core/physics.py`:
   - `core/leapfrog.py` — the "fast" integrator under test.
   - `core/rebound_reference.py` — the "ground truth" integrator (thin
     wrapper around the REBOUND library's IAS15).
3. **Calibration pipeline** (`calibration/`) — generates random systems,
   runs both integrators on each one, compares the two trajectories, and
   writes the result to CSV. `calibration/runner.py` fans this out across
   many OS processes in parallel.

`main.py` just wires steps 1–3 together with a CLI.

If you remember nothing else: **`System`/`Body` (physics) flows into
`Trajectory` (an integrator's output) flows into a **`DataFrame`** (the
comparer's output) flows into a **CSV file** (the logger's output).**

---

## 2. Folder structure

```text
iris_calibration/
│
├── core/                       # Physics + integrators. No parallelism, no I/O, no randomness.
│   ├── physics.py               # Body, System, forces, energy, momentum, angular momentum
│   ├── leapfrog.py               # Custom Velocity Verlet integrator + Trajectory dataclass
│   ├── rebound_reference.py      # IAS15 wrapper (REBOUND), returns the same Trajectory shape
│   └── metrics.py                # Per-timestep diagnostics (errors, drift, jerk, etc.)
│
├── calibration/                # Orchestration: randomize, run, compare, save, parallelize.
│   ├── generator.py              # Builds & serializes randomized ExperimentConfig -> JSON
│   ├── comparer.py               # Leapfrog Trajectory + IAS15 Trajectory -> comparison DataFrame
│   ├── logger.py                 # DataFrame -> CSV, enforces the required column schema
│   └── runner.py                 # ProcessPoolExecutor fan-out across many experiments
│
├── experiments/
│   ├── configs/                  # Generated ExperimentConfig JSON files (one per experiment)
│   └── calibration/               # Reserved for future experiment-specific artifacts
│
├── tests/
│   └── validate_integrator.py    # Standalone physical-correctness checks (no pytest needed)
│
├── outputs/
│   ├── csv/                      # One CSV per experiment, written by calibration/logger.py
│   └── figures/                  # Optional validation plot (main.py --plot)
│
├── main.py                      # CLI entry point: generate -> run -> summarize
└── requirements.txt
```

Rule of thumb for where new code goes: if it's a physical law, it goes in
`core/`; if it's about orchestrating *many* experiments (randomizing,
running in parallel, saving), it goes in `calibration/`.

---

## 3. Data flow, end to end

```text
generate_calibration_batch()          [calibration/generator.py]
        │  writes JSON files
        ▼
experiments/configs/*.json  ──load_experiment_config()──▶  ExperimentConfig
        │
        ▼
_config_to_system()                    [calibration/runner.py]
        │  builds a core.physics.System (list of Body + G + softening)
        ▼
   ┌────────────────────┬─────────────────────────┐
   ▼                    ▼                         │
run_leapfrog()      run_ias15()                   │
[core/leapfrog.py]  [core/rebound_reference.py]    │
   │                    │                         │
   ▼                    ▼                         │
Trajectory (Leapfrog)  Trajectory (IAS15)          │
   └─────────┬──────────┘                          │
             ▼                                     │
   compare_trajectories()          [calibration/comparer.py]
             │  same-timestep-by-timestep diff + physics diagnostics
             ▼
   pandas.DataFrame (one row per sampled timestep)
             │
             ▼
   save_calibration_csv()          [calibration/logger.py]
             │
             ▼
   outputs/csv/<simulation_id>.csv
```

`calibration/runner.py` runs the middle section (`_config_to_system` through
`save_calibration_csv`) once per experiment, in a separate OS process, for
every config path handed to it. `main.py` is just:
`generate_calibration_batch()` → `run_calibration_batch()` → print summary.

---

## 4. Core data structures

### `core.physics.Body`
One point mass: `mass` (float), `position` (3-vector), `velocity`
(3-vector), optional `name`. Always stored as `float64`.

### `core.physics.System`
A list of `Body` plus the constants that apply to all of them: `g`
(gravitational constant) and `softening` (softening length, 0.0 = exact
Newtonian gravity). Has convenience accessors (`.masses()`, `.positions()`,
`.velocities()`) that stack per-body arrays into `(N,)` / `(N, 3)` NumPy
arrays — this stacked form is what the vectorized physics functions
actually operate on.

### `core.leapfrog.Trajectory`
The common output shape for **both** integrators:
```python
times: (T,)        # sample timestamps
positions: (T, N, 3)
velocities: (T, N, 3)
masses: (N,)        # constant over time
```
Because both `run_leapfrog()` and `run_ias15()` return this exact shape,
`calibration/comparer.py` doesn't need to know or care which integrator
produced which trajectory.

### `calibration.generator.ExperimentConfig`
A plain, JSON-serializable dataclass holding everything needed to
reproduce one experiment: `simulation_id`, `seed`, `body_count`, `g`,
`softening`, `dt`, `total_time`, `sample_interval`, and the initial
`masses` / `positions` / `velocities`. This is the *only* thing that
crosses the process boundary in the parallel runner (workers load it from
disk rather than receiving live Python objects), which is what makes
`calibration/runner.py` safe to parallelize.

---

## 5. Why sample_interval has to be a multiple of dt

`run_leapfrog()` steps at a fixed `dt` and only *records* a sample every
`sample_interval` of simulated time. `run_ias15()` is asked to integrate
directly to each multiple of `sample_interval`. For the two trajectories
to be comparable timestep-for-timestep in `compare_trajectories()`, their
recorded timestamps must match exactly — so `sample_interval` must be an
exact integer multiple of `dt`. `run_leapfrog()` raises `ValueError` if you
violate this; if you're writing a script that calls it directly (rather
than going through `generator.py`, which already respects this), keep it
in mind.

---

## 6. How to extend common things

### Add a new integrator to calibrate
1. Write a function in `core/` (e.g. `core/rk4.py`) with the signature
   `run_rk4(system: System, dt: float, total_time: float, sample_interval: float | None) -> Trajectory`,
   matching `run_leapfrog`'s contract exactly (same `Trajectory` shape,
   same sampling rules).
2. In `calibration/runner.py::run_single_experiment`, call your new
   function alongside (or instead of) `run_leapfrog`.
3. `calibration/comparer.py::compare_trajectories` already works against
   any two `Trajectory` objects with matching timestamps — no changes
   needed there.

### Add a new metric/column to the CSV
1. Implement the calculation as a pure function in `core/metrics.py`
   (per-timestep in, scalar out — follow the existing style).
2. Call it inside the per-sample loop in
   `calibration/comparer.py::compare_trajectories` and add the result to
   the `rows.append({...})` dict.
3. Add the new column name to `REQUIRED_COLUMNS` in
   `calibration/logger.py` — `save_calibration_csv` will refuse to write
   a CSV that's missing anything in that tuple, so this is a deliberate
   guardrail against silently dropping columns.

### Change how initial conditions are randomized
Everything lives in `calibration/generator.py::_random_orbital_system`.
It currently places one heavy "anchor" body plus N-1 bodies on randomized,
randomly-oriented near-circular orbits. To change the distribution of
systems (e.g. add hierarchical binaries, different mass spectra, unbound
trajectories), edit this function — the rest of the pipeline only cares
that it gets back `(masses, positions, velocities)` lists of the right
shape. `MASS_RANGE`, `RADIUS_RANGE`, and `ECCENTRICITY_RANGE` at the top
of the file are the quickest knobs to turn.

### Add a new validation test
Add a `test_*() -> bool` function to `tests/validate_integrator.py`
following the existing pattern (build a system, integrate, assert some
physical invariant, call `_print_result`), then add it to the list in
`run_all_tests()`.

### Change parallelism (worker count, batch size)
- Worker count: `--workers` CLI flag on `main.py`, or the `n_workers`
  argument to `calibration.runner.run_calibration_batch`. Default is 16
  (`DEFAULT_WORKERS` in `calibration/runner.py`).
- Number of experiments: `--n-experiments` CLI flag on `main.py`, or the
  `n_experiments` argument to `calibration.generator.generate_calibration_batch`.
  Default is 16.
- Each worker is a fresh OS process (`ProcessPoolExecutor`), so anything
  passed into `run_single_experiment` must be picklable — that's why
  workers take a `config_path` (a string) rather than a live `System`
  object.

---

## 7. Running things

```bash
# Install dependencies
pip install -r requirements.txt

# Sanity-check the integrators before trusting any calibration data
python tests/validate_integrator.py

# Run the default pipeline: 16 experiments, 16 parallel workers
python main.py

# Common overrides
python main.py --n-experiments 4 --workers 4      # smaller/faster local run
python main.py --dt 5e-4 --total-time 10.0         # finer timestep, longer run
python main.py --seed 123                          # different (still reproducible) batch
python main.py --plot                              # also emit outputs/figures/*.png
```

Every run's `outputs/csv/<simulation_id>.csv` is uniquely named after its
experiment, so re-running `main.py` never silently overwrites unrelated
experiments' data — though it *will* overwrite a CSV if you reuse the same
`--seed` (same seed ⇒ same `simulation_id`s ⇒ same filenames).

---

## 8. Reproducibility guarantees

- Every `ExperimentConfig` carries the `seed` that generated it; the exact
  same seed through `generate_experiment_config` always yields the exact
  same masses/positions/velocities (uses `numpy.random.default_rng(seed)`,
  never global RNG state).
- Both integrators are deterministic given the same initial conditions —
  Leapfrog because it's a fixed-step explicit scheme, IAS15 because
  REBOUND's Gauss-Radau stepper is deterministic for a given system and
  target times (verified by Test 1 in `tests/validate_integrator.py`).
- Nothing in `core/` or `calibration/` touches wall-clock time, thread
  IDs, or other non-reproducible state for anything except progress
  logging / timing metadata (`WorkerResult.elapsed_seconds`).

---

## 9. Known sharp edges

- **Large `position_error` / `energy_drift` on some experiments is
  expected, not a bug.** Randomized systems occasionally produce close
  encounters; a fixed-step Leapfrog integrator can lose accuracy fast in
  those cases while IAS15 (adaptive) does not. That divergence *is* the
  calibration signal this framework exists to measure — don't "fix" it
  by, say, shrinking `dt` globally without understanding why.
- `core/physics.py::compute_accelerations` computes the full O(N²)
  pairwise interaction matrix every call. Fine at N ≤ 5 (the current
  scope); would need reworking (e.g. Barnes-Hut, or handing everything to
  REBOUND) before scaling to much larger N.
- `core/rebound_reference.py` calls `sim.integrate()` once per sample
  point rather than once for the whole run — this keeps its sampling
  aligned exactly with Leapfrog's, at some performance cost. If profiling
  shows this matters, look at REBOUND's `exact_finish_time` /
  scheduled-heartbeat APIs before trying to batch it.
