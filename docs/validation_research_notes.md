# IRIS Adaptive-Integrator Validation: Research Notes

Source material for a project report on the validation stage of `iris_calibration`
(`validate_adaptive.py` and everything it drives). Written from the code
(`core/adaptive.py`, `core/safety_index.py`, `core/situations.py`, `validation/runner.py`,
`validation/compare.py`, `fit_safety_models.py`, `fit_situation_weights.py`,
`analyze_safety_index.py`) and the result files under `outputs/`.

**Reading guide.** Every statement is tagged by how I know it:
- **[code]** read directly from source or docstrings.
- **[data]** computed from a CSV/JSON in `outputs/`.
- **[inferred]** deduced from indirect evidence. The summary CSVs do **not** record the
  command-line flags a run used, so run configurations below are reconstructions. Confirm them
  against your shell history before putting them in a paper.

---

## 1. Problem and goal

The calibration stage (`main.py`) runs a fast fixed-step **Velocity Verlet / Leapfrog**
integrator in lockstep with **IAS15** (REBOUND's 15th-order adaptive integrator, the ground
truth). At each sample, if the RMS position error against IAS15 exceeds the
`correction_threshold`, Leapfrog is snapped back onto IAS15. That needs the reference, so it
cannot be used in production.

The validation stage asks: **can a method that never sees IAS15 get a comparable error count?**
The adaptive integrator (`core/adaptive.py`) uses Leapfrog by default and hands over to IAS15
when a **safety score**, computed from its own state only, says Leapfrog is about to fail.
Everything in this document is a strategy to cut the number of samples whose error exceeds the
threshold, and the tooling to understand why the remaining ones fail.

## 2. Experimental design

**Three trajectories per experiment** [code: `validation/runner.py`, `validation/compare.py`]:

| Name | What it is | Role |
|---|---|---|
| IAS15 | `run_ias15`, REBOUND native | ground truth |
| baseline | pure Leapfrog (with `--correct`: `run_leapfrog_with_correction`, i.e. the calibration run) | cheap method to beat |
| adaptive | `run_adaptive` | method under test |

**Metric.** RMS position error vs IAS15 (`core.metrics.position_error`) at every sample.
An **error** is a sample where the *pre-check* (pre-snap) error exceeds
`correction_threshold` = 0.01 [data]. With `--correct`, an error equals one correction that had
to be applied, so error counts are directly comparable to the calibration procedure. Also
reported: mean/max/final error, velocity error, energy and angular-momentum drift, IAS15 step
share, switches, rewinds, wall time.

**Run geometry in the saved results** [data]: 32 three-body experiments per run, seeds 956–987,
200,000 steps at dt = 1e-3 (total time 200, sample interval 0.05, 4,001 samples each, 128,032
samples per run). `correction_enabled = True` in all runs. Note this differs from the CLI
default `--total-time 5.0`.

**Leakage control.** Calibration (training) data is `outputs/csv_3body`, seeds 42–89
(48 experiments, 3 bodies). Validation seeds 956–987 do not overlap. The CLI default seed is
10,000 for the same reason [code]. Experiment *i* uses `seed + i`.

**Baseline difficulty** [data]: pure Leapfrog with correction needed **53 corrections across
20 of 32 experiments** (of 128,032 samples). This is the number every adaptive variant tries to
reduce.

## 3. Deriving the safety index

### 3.1 Features (all causal, from Leapfrog's own state) [code: `core/safety_index.py`]

Per body, at the current step: relative speed to the nearest neighbour, nearest-neighbour
distance, |acceleration|, |jerk| (backward difference `(a(t) − a(t−dt))/dt`), mass, and a
system-level mass ratio. All are log10-transformed with a floor of 1e-15
(`log_rel_velocity`, `log_jerk`, `log_acceleration`, `log_nn_distance`, `log_mass`,
`log_mass_ratio`; also `log_v_over_r`, used only by tree models). The system score is the
**max over bodies**. A non-finite index maps to score 1.0 (fail-safe: unsafe).

Quantities needing IAS15 (`position_error`, `error_at_trigger`, ...) are excluded as **label
leakage**. Situational flags are excluded from the deployable index because they use central
differences and whole-run percentiles (look-ahead).

### 3.2 Step 1: exploratory power-law regression (`analyze_safety_index.py`)

- Data: correction-event rows only (every CSV under `outputs/csv` is already "just corrections").
- Predictors: the `pre_correction_*` columns (state one dt *before* the snap, i.e. what a
  real-time check would see), plus mass, mass ratio, timestep.
- Target: `body_position_error_at_trigger` (severity at detection).
- Model: OLS on `log10(error) = c0 + Σ coef_i · log10(feature_i)`, i.e. a power-law index
  `10^c0 · Π feature_i^coef_i`. 25% held-out test split for R².
- Diagnostics: single-feature log–log correlations; pairwise collinearity matrix with a warning
  at |r| ≥ 0.6. The script explains why: acceleration ~ 1/r² and jerk ~ v/r³ move together, so
  coefficients can flip sign between collinear features.
- Default `--drop mass mass_ratio timestep` because they carried near-zero or unstable
  coefficients in exploratory fits.
- Thresholds: the index value at percentiles 50/75/90/95/99 of observed corrections.
- **Limitation that motivated the next step:** it only sees corrections (positives). It cannot
  say how often a *quiet* step would also score high, so it cannot give a false-positive cost.

### 3.3 Step 2: classification with negatives (`fit_safety_models.py`)

This is where the ridge/lasso formula comes from.

**Data.** Calibration logging now also writes sampled *negative* rows. Label:
`sample_role == "correction"` → 1, any `negative_*` → 0. Negatives are deliberately enriched
(pre-failure samples, near-misses, every 20th baseline sample) and carry a `sampling_weight`
that undoes the enrichment. `--three-body` restricts to 3-body systems from `outputs/csv_3body`.
Rows are capped at 120,000 (all positives kept; negatives thinned, weights scaled up so weighted
metrics stay honest).

**Models compared** (each with grouped CV):

| Model | Features | Purpose |
|---|---|---|
| logistic | `rv_only` | the "relative velocity is the index" hypothesis |
| logistic | `jerk_only`, `jerk+rv` | compact candidates |
| ridge (L2 logistic) | all causal continuous | shrinks correlated features instead of letting them flip signs |
| lasso (L1 logistic) | all causal continuous | zeroes redundant features; a data-backed test of "only rv matters" |
| gradient-boosted trees | `rv_only`, `continuous` | nonlinearity/interaction check: if trees do not beat linear, a linear index is not leaving signal behind |
| ridge/GBT with flags (`*`) | + 18 situational flags | **offline reference only**; look-ahead, never deployed |

**Method safeguards worth stating in a paper:**
1. **Grouped K-fold by `simulation_id`** (default 5). Rows within an experiment are strongly
   correlated; random splits would leak near-duplicates and inflate AUC.
2. **Nested tuning:** C ∈ logspace(−3, 2, 6) is chosen by an *inner* grouped 3-fold CV on each
   training fold only (maximising ROC-AUC), so test folds are never used for tuning.
3. Standardised inputs; coefficients are converted back to **raw log10-feature units** so the
   formula can be pasted verbatim (`coef_raw = w_std / scale`, intercept corrected for the mean).
4. `log_v_over_r` is excluded from linear models (exactly `log_rel_velocity − log_nn_distance`,
   perfectly collinear); constant columns and `stable` (complement of the other flags) dropped.
5. A runtime assertion confirms the "lasso" really applies an L1 penalty (scikit-learn 1.8
   changed the API; the wrong spelling silently gives ridge).
6. Metrics: AUC, PR-AUC, and sampling-weighted versions (`w_AUC`, `w_AP`).
7. **Threshold table:** for target recall 0.80/0.90/0.95/0.99, find the score threshold
   achieving it on out-of-fold scores and report the false-positive rate (raw and
   weight-adjusted). The weighted rate approximates "share of ordinary steps sent to IAS15
   needlessly". The 0–1 threshold is `sigmoid(safety_logit)`.

### 3.4 The formula that was adopted [code: `raw_safety_index`; docstring]

Ridge logistic, causal features only, `C = 0.001`, **fit on 3-body calibration data only**
(printed by `python fit_safety_models.py --three-body`):

```
safety_logit = +0.7838
             + 0.6963 * log_rel_velocity
             + 0.2168 * log_jerk
             + 0.4218 * log_acceleration
             - 0.2614 * log_nn_distance
             - 2.0146 * log_mass
             - 2.5698 * log_mass_ratio
safety_score = 1 / (1 + exp(-safety_logit))      # per body; system score = max over bodies
```

- **Quality** (grouped 5-fold CV, per docstring): AUC 0.744 ± 0.174, weighted AUC 0.791.
- **Usable operating point:** only the 0.80-recall threshold, `safety_threshold ≈ 0.134`,
  sending ~19% of ordinary steps to IAS15. Higher-recall thresholds fall at logits ≤ −15 with
  ~97% false-positive rate: the index does not separate well at high recall.
- **Not a probability:** negatives are enriched, so the sigmoid output is uncalibrated. Pick the
  threshold from the recall/cost table, not 0.5 (the CLI default).
- **Predecessor:** ridge on mixed 2–5 body data, CV AUC 0.907:
  `−5.263 − 4.979·log_rel_velocity + 7.484·log_jerk − 6.372·log_acceleration + 8.370·log_nn_distance − 0.088·log_mass − 0.273·log_mass_ratio`.
  It was replaced by the 3-body-only fit. The much lower AUC of the 3-body fit with a large fold
  spread (±0.174) should be discussed honestly: fewer systems, harder negatives, and few
  experiments per fold.
- **Observations a reviewer will ask about [inferred, not tested here]:** (a) the sign of
  `log_rel_velocity` flipped between versions (−4.98 → +0.70), consistent with the
  collinearity the scripts warn about, so individual coefficients should not be interpreted
  physically; (b) the largest weights are on `log_mass` and `log_mass_ratio`, which are constant
  within an experiment, so the index partly encodes "which kind of system" rather than "which
  moment". I did not see the lasso output; if you have it, report which features it kept, since
  that is the intended sparsity check.

### 3.5 Alternative: additive situation-weight index (`fit_situation_weights.py`, `core/situations.py`)

**Fit.** One L2 logistic regression on the 18 binary situation flags from
`calibration/events.py`, so overlapping flags share credit. `C` by inner grouped CV; weights by
`sampling_weight` (or `--unweighted`). Outputs `situation_weights.json/csv`.
`safety_logit = intercept + Σ weight[s]` over active situations; score = sigmoid.

**Result** [data: `outputs/situation_weights.json`]: 584,130 rows, 48 experiments, `C = 10`,
intercept −12.83. Grouped 5-fold AUC **0.913**, weighted AUC **0.928**, versus **0.873** for the
naive "count of active anomaly flags". Thresholds (out-of-fold):

| Target recall | 0–1 threshold | False-positive rate (weighted) |
|---|---|---|
| 0.80 | 0.0112 | 7.4% |
| 0.90 | 0.00285 | 15.0% |
| 0.95 | 0.00205 | 22.0% |
| 0.99 | 7.0e-5 | 56.0% |

Largest weights: `ejection_escape` +6.84, `involved_in_hierarchical_tight_pair` +4.61,
`near_collision` +4.38, `strong_mass_ratio_interaction` +2.14, `acceleration_spike` +1.49;
negative: `hierarchical_configuration` −2.54, `rapid_approach` −2.16. Fold sign agreement is low
for `close_encounter` (0.4), `high_jerk` (0.4), `strong_acceleration` (0.4), `rapid_separation`
and `high_relative_velocity_encounter` (0.6): these weights are unstable. Negative weights on
risk-sounding flags (e.g. `rapid_approach`) reflect overlap with `near_collision`, not that
approaches are safe.

**The causal gap.** The flags were fitted on offline definitions (central differences, whole-run
percentiles, post-snap state). `core/situations.py` re-implements them causally: evaluated on the
sample grid and held between samples; backward differences; percentile thresholds from history
so far, refreshed every 50 samples, with a 10-sample warm-up; trailing window 10% of the run;
"temporary capture" relative to the bound pairs at sample 0. **The weights were not refitted on
the causal flags**, so its AUC does not transfer; judge it by validation error counts. It is
stateful, so it implements `snapshot_state/restore_state` for rewinding (the history arrays are
intentionally not copied: entries past the restored sample are always rewritten before being
read). Enabled with `--situation-index` (keep `--index-scale sigmoid`); the CLI suggests
thresholds 0.011/0.0029/0.0021/7e-5 for recall 0.80/0.90/0.95/0.99.

**Which index did each saved run use?** The summaries do not say. The formula index is the
default and the docstring of `raw_safety_index` documents it as the adopted one; treat the
situation index as an experimental alternative unless your logs show otherwise.

## 4. Strategies used to reduce error counts

All implemented in `core/adaptive.py` [code]. Ordered roughly as they were layered on.

### S1. Adaptive switching with hysteresis
Per step: in Leapfrog, switch to IAS15 if `score ≥ safety_threshold`; in IAS15, stay while
`score ≥ release_fraction · safety_threshold` (default 0.8), then return. `--ias15-hold-steps`
keeps IAS15 for extra steps after release. The first step is always Leapfrog (no jerk history).
State passes between integrators as plain positions/velocities; a fresh REBOUND simulation is
built at each switch, consecutive IAS15 steps reuse one. **Key limitation:** switching stops
*further* error growth but does not repair error already accumulated.

### S2. Rewinding
Detection is late: by the time the score fires, Leapfrog has already drifted. With `--rewind`, a
checkpoint (full state plus all bookkeeping) is stored every `checkpoint_interval` (default 10)
steps in a ring of `checkpoint_count` (default 5). On a trigger, restore the checkpoint
`rewind_back` places before the latest (default 1 = second-latest) and replay with IAS15 up to
and including the trigger step (guaranteeing progress, no rewind loops). Everything recorded
after the checkpoint (samples, switch events, counters, index memory) is discarded and
regenerated, so outputs describe only the final timeline. Cost: replayed steps
(`n_rewound_steps`).

### S3. False-positive detection (diagnostic only)
For each IAS15 stretch, a **shadow Leapfrog** runs in lockstep from the same start state (for a
rewind, from the restored checkpoint). If its RMS deviation from the actual state stays below
`--false-positive-threshold` (default = correction threshold) for the whole stretch, the trigger
was a false positive (IAS15 was not needed). No reference needed. Writes
`<sim>_stretches.csv`. Limitation: the verdict covers only the stretch's duration; drift that
would have exceeded the threshold later is not counted against the trigger.

### S4. Rollback-depth analysis (diagnostic only; uses the IAS15 oracle)
Question: **how many errors would a deeper rollback have prevented?** At every rewind, each
queued checkpoint is used as a counterfactual restore point: IAS15 is replayed to the trigger
step and compared with the exact IAS15 state. A candidate is **effective** if its deviation is
≤ `max(rollback_clean_threshold, (1 − rollback_min_gain) · deviation of the un-rewound state)`
(defaults 1e-9 and 0.5). An absolute "clean" test alone cannot work because Leapfrog is never
bit-exact. Each rewind is classed:
- **adequate**: the checkpoint used was effective;
- **improvable**: it was not, but a deeper queued one was;
- **drift_dominated**: none was; the error built up long before the queue's window.

Every later error (pre-check error above the threshold) is attributed to the rewinds since the
last clean point (start, snap, or resync): `improvable` > `accumulated_drift` >
`after_adequate_rewinds` (chaos or drift after release) > `undetected` (score never fired).
It is an attribution, not proof: the whole run is not re-run with the deeper rollback.
Outputs: `<sim>_rewinds.csv` (per-depth deviations), `<sim>_rollback_errors.csv`.

**Finding that drove everything after it** [data, run A]: of 300 rewinds, **285 were
drift-dominated** and only 1 of 15 errors was `improvable`; **13 of 15 were accumulated drift**.
So deeper rollback inside a 5×10-step queue cannot help: the dominant error is slow phase drift
over thousands of calm steps, which neither the safety score (it only sees close approaches) nor
a short rewind can see.

### S5. Drift budget (Richardson resync)
While in Leapfrog, a shadow Leapfrog with half the step runs alongside. Velocity Verlet is
second order, so accumulated error ≈ (4/3)·RMS|x(dt) − x(dt/2)|. When this estimate reaches
`drift_budget · correction_threshold`, the state is **resynced** to the Richardson extrapolation
`(4·x_fine − x_coarse)/3` (same for velocity), removing the leading error term, and the shadow
restarts. Needs no reference. Cost ≈ 3 force evaluations per Leapfrog step instead of 1.
Caveat: assumes smooth dynamics; close encounters remain the score's job. Suggested range
0.1–0.5.

### S6. Anchor rewinding (`--rewind-to-anchor`)
Instead of the checkpoint `rewind_back` places back, rewind to the **anchor**: the last verified
state (start, last resync, last snap to the reference, or end of an IAS15 stretch that began at
the anchor). Replays with IAS15 from there, repairing accumulated drift at the price of a long
IAS15 stretch, hence meant to be combined with a drift budget to bound the distance.

### S7. Resync on switch (`--resync-on-switch`)
Observation: the drift budget only sees error inside the current Leapfrog stretch, and an IAS15
handoff *freezes* that error into the state. So budget resyncs almost never fire while every
stretch still hands over a contaminated state. Fix: apply the Richardson repair to the state IAS15
takes over from (with rewind: the restored checkpoint, using the dt/2 shadow saved in it). Every
Leapfrog stretch then ends with a repair, so error cannot pile up from stretch to stretch and no
long rollback is needed. The repaired state becomes the anchor. Can be combined with the budget.

### S8. Compensated (Kahan) sums (`--compensated-sums` / `--kahan`)
At ~1e-16 per step, double-precision roundoff in `x += v·dt + ...` accumulates over ~200k steps
and becomes the seed that chaos (Lyapunov time ≈ 3) amplifies into a late threshold error, long
after truncation error was removed by Richardson. Position and velocity sums are Kahan-compensated
in the main step and the shadows; compensation terms are checkpointed and restored; the repair is
written as `base + small correction` with the final rounding kept in the compensation term.
Compensation resets to zero when the state comes from REBOUND or the reference. Off by default,
and then results are bit-identical to the earlier version.

### S9. Higher-order Richardson (`--richardson-levels 3`)
Adds a dt/4 shadow and repairs with `(64·x_q − 20·x_h + x_c)/45`. Velocity Verlet is symmetric
(error series in even powers of dt), so this removes the h⁴ term as well. Cost ≈ 7 force
evaluations per step. Only worthwhile with Kahan sums, otherwise roundoff is the floor.

### S10. Online correction as the accounting rule (`--correct`)
Both adaptive and baseline snap back to IAS15 when the pre-check error exceeds the threshold, so
"errors" = corrections needed, exactly as in calibration. Without it, one failure taints every
later sample.

### Instrumentation added alongside
- **Chaos-floor control (`--chaos-control` / `--ias15-stepped`).** Pure IAS15 forced to stop at
  every dt, like the adaptive run's IAS15 stretches. Its error against native IAS15 is the best
  any integrator matching IAS15 to rounding can score in a chaotic system. Adaptive errors where
  the control has also failed are **floor** (not fixable); control error ≥ 0.1×threshold is
  **marginal**; otherwise **fixable** (attributable to the algorithm). Also reports chaos floor
  time, exponential growth rate (1/rate = Lyapunov time), and `residual_factor_needed` =
  `exp(rate · gap)`: how many times smaller the injected error must be to reach the floor.
- **Like-for-like timing.** Native IAS15 (C, stops only at sample times) is not comparable with
  Python-loop Leapfrog/adaptive. The control run supplies a same-configuration IAS15 baseline.
  `cpu_*_s` (process CPU time) columns were added so core contention does not distort timing.
- **Per-resync quality.** `<sim>_resyncs.csv` stores true error before/after each repair
  (needs `--analyze-rollback`).

## 5. Results

### 5.1 Run-by-run comparison [data; configurations inferred]

Folders `outputs/validation_A` … `validation_G`: same 32 three-body systems (seeds 956–987),
correction on. Baseline in every run: **53 errors, 20/32 experiments with errors**.

Configurations are reconstructed from which summary columns are populated and the counts of
resync events:

| Run | Inferred configuration | Evidence |
|---|---|---|
| A | rewind + rollback analysis, no drift repair | `n_resyncs` empty |
| B | A + drift budget | 2 resyncs |
| C | rewind-to-anchor + drift budget | IAS15 share jumps to 86.8%; 287/300 rewinds "adequate" (anchor is verified) |
| D | resync-on-switch | 300 switch resyncs = one per rewind; 45 columns (new fields) |
| E | D + a drift budget | 300 switch + 2 budget resyncs |
| F | D + a tighter budget | 300 switch + 235 budget |
| G | D + a still tighter budget | 300 switch + 1,201 budget |

The exact budget fractions, and whether Kahan / Richardson-3 were on, cannot be recovered from
the files.

| Run | Adaptive errors | Experiments with errors | Errors fixable by deeper rollback | Accumulated-drift errors | Mean IAS15 step share | Switches | Median error reduction* |
|---|---|---|---|---|---|---|---|
| baseline (all) | 53 | 20 | n/a | n/a | 0% | n/a | n/a |
| A | 15 | 9 | 1 | 13 | 51.0% | 581 | 0.616 |
| B | 15 | 9 | 1 | 12 | 51.0% | 581 | 0.616 |
| C | 2 | 2 | 0 | 1 | **86.8%** | 313 | 1.000 |
| D | 4 | 4 | 0 | 3 | 51.0% | 581 | 0.9993 |
| E | 4 | 4 | 0 | 2 | 51.0% | 581 | 0.9993 |
| F | 3 | 3 | 0 | 2 | 51.0% | 581 | 0.9993 |
| G | 3 | 3 | 0 | 2 | 51.0% | 581 | 0.9996 |

\*Median over experiments of `mean_error_reduction = 1 − adaptive_mean_err / baseline_mean_err`.
All runs: 300 rewinds.

How to read it:
1. **A → B:** a small drift budget alone changes nothing (2 resyncs in 32 experiments), exactly
   the problem S7 describes.
2. **A → D:** repairing the handoff state cut errors **15 → 4** at *unchanged* IAS15 share
   (51%), and the median true error after/before a repair is **≈3×10⁻⁴** (D: 0.000324; E: 0.000302;
   F: 0.000406; G: 0.000302), i.e. a ~3,000× reduction per repair.
3. **C** reaches 2 errors but by sending 86.8% of steps to IAS15, so it largely gives up the
   speed benefit.
4. **D → G:** more budget resyncs (302 → 535 → 1,501) trims 4 → 3 errors. Diminishing returns.
5. In G the rewind classification shifts: 17 adequate / 23 improvable / 260 drift-dominated,
   versus 11 / 1 / 288 in D. Plausibly because repaired checkpoints are closer to exact, but this
   is untested.
6. **Residual errors** in G: 2 accumulated drift, 1 after adequate rewinds. The last category is
   consistent with chaos or roundoff amplification (S8/S9, chaos control), not a safety-score miss.
7. Rewind depth classification across A–G is dominated by `drift_dominated` (260–288 of 300) in
   all except C, supporting the S4 conclusion.

### 5.2 Latest run (`outputs/validation_G/validation/` and `outputs/validation/`) [data]

A later run on the same 32 systems, flags not recorded:

| Metric | Run G (above) | Latest run |
|---|---|---|
| Adaptive errors / baseline | 3 / 53 | **3 / 53** |
| Mean IAS15 step share | 51.0% | **22.2%** |
| IAS15 steps (total) | n/a | 1,421,000 |
| Rewinds | 300 | 666 |
| Resyncs | 1,501 | 2,397 |

Same error count at **less than half the IAS15 usage** is the strongest cost/accuracy result in
the folder. I cannot tell from files which change produced it (candidates: Kahan sums,
`--richardson-levels 3`, a different threshold, hold steps, a different index). Reconstruct this
before claiming a cause. `outputs/validation/validation_summary.csv` reproduces the same totals
(adaptive 3, baseline 53).

### 5.3 Cost: an honest reading [data]

Summed wall-clock over 32 experiments (run G): adaptive **1,921 s**, Leapfrog-only **281 s**,
native IAS15 **5.6 s**. Latest run: adaptive 2,406 s, Leapfrog 318 s, IAS15 5.9 s. The adaptive
method is **slower in wall time than native IAS15**. Reasons: the native reference is C and stops
only at sample times, while adaptive/Leapfrog are Python per-step loops with per-step feature
computation and REBOUND call overhead; `--analyze-rollback` adds oracle IAS15 replays to the
adaptive time; and `--workers` above the core count adds queueing. The `--ias15-stepped` baseline
and `cpu_*` columns were added precisely to make a fair comparison, but no saved summary contains
them (`wall_control_s` is empty or zero), so **no like-for-like speed claim is currently
supported by the data**. The machine-independent measure that *is* supported is IAS15 step share.

## 6. Other experiments found in `outputs/validation_G/csv`

Per-experiment files from several earlier or parallel runs share this folder. Seed families and
file types identify them [data: file listing; interpretation inferred]:

| Seed family | Systems | Files present | Likely purpose |
|---|---|---|---|
| 10000–10015 | mixed 2/3/4/5 bodies (default seed) | `*_validation.csv` only | first end-to-end validation, default settings |
| 598476–598507 | mixed 2–5 bodies | `*_stretches.csv` + validation | false-positive detection on mixed systems |
| 712–715 | 3 bodies | `*_rewinds.csv`, `*_rollback_errors.csv` | early rollback-depth analysis |
| 780–811 | 3 bodies | `*_stretches.csv` + validation | false-positive detection on 3-body systems |
| 956–987 | 3 bodies | rewinds, rollback errors, **resyncs**, validation | full drift-repair experiments (runs A–G) |

Per-experiment files are overwritten when a seed is reused, so they reflect the **last** run on
that seed and may not match the summary row for it. Example: `sim_0000_n3_seed956_rewinds.csv`
holds a single rewind (step 2 → 0, score 0.189, class adequate) while a summary shows 47 rewinds
for the same system, so these came from different runs.

One illustrative stretch (`sim_0000_n3_seed780`): a single IAS15 stretch starting at step 1 after
a rewind to the start, trigger score 0.196, peak 0.948, **never released** (100,000 steps,
`complete = False`), with a shadow-Leapfrog deviation of ~832. A clear true positive: pure
Leapfrog would have diverged. I did not aggregate false-positive rates across the 780–811 and
598476+ families; do that from the `*_stretches.csv` files before reporting a figure.

## 7. Threats to validity and open items

1. **Small sample.** 32 systems, 3 bodies, 1 dt, 1 threshold (0.01). 53 baseline errors is a
   small count; going 53 → 3 is a large relative change on few events and runs concentrate in few
   experiments.
2. **Run configurations are not logged.** Add the full argv and git hash to the summary CSV or a
   sidecar JSON. This is the single biggest reproducibility gap.
3. **Index quality at the operating point.** AUC 0.744 ± 0.174 (3-body ridge) is modest and the
   0.134 threshold costs ~19% of ordinary steps. Much of the final accuracy comes from the
   **repair** strategies (S5–S9), not from the index; the ablation A → D shows this directly.
4. **Oracle use.** Rollback analysis, per-resync quality and the chaos control all use IAS15; the
   algorithm itself does not. State this clearly so the method is not described as using the
   reference.
5. **Chaos floor not quantified.** Chaos-control fields are empty in all saved summaries, so the
   claim that the remaining 3 errors are unfixable is unproven. Run with `--chaos-control`.
6. **Weights not refit on causal situation flags**, and the situation index has no saved
   validation result I could identify.
7. **No wall-clock win demonstrated** (section 5.3).
8. **Generalisation.** Index fitted on 3-body only; the 2-, 4-, 5-body behaviour is untested here.
9. **Drift estimator assumptions.** Richardson needs smooth dynamics; repairs during close
   encounters are not guaranteed.

## 8. Suggested report structure (mapping to this document)

1. Introduction and objective (§1)
2. Methodology: ground truth, metrics, error definition, leakage control (§2)
3. Safety index: features, regression → classification, grouped nested CV, threshold selection (§3)
4. Alternative situation index and the causality gap (§3.5)
5. Error-reduction strategies and their rationale, S1–S10 (§4)
6. Ablation results A–G and the latest run (§5)
7. Diagnostics: false positives, rollback attribution, chaos floor (§4 S3, S4, instrumentation; §6)
8. Limitations and future work (§7)

## 9. Reproduction commands


```bash
# index derivation
python main.py --three-body                      # calibration data -> outputs/csv_3body
python fit_safety_models.py --three-body         # prints the ridge/lasso formulas and threshold tables
python fit_situation_weights.py                  # -> outputs/situation_weights.json

# validation (examples)
python validate_adaptive.py --three-body --seed 956 --n-experiments 32 --total-time 200 \
    --correct --safety-threshold 0.134 --rewind --analyze-rollback \
    --resync-on-switch --drift-budget 0.1 --compensated-sums --richardson-levels 3 \
    --chaos-control
python validate_adaptive.py --three-body --seed 780 --n-experiments 32 --correct --rewind \
    --detect-false-positives
```
