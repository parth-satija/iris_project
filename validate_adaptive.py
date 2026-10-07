"""
validate_adaptive.py

Entry point for the IRIS VALIDATION experiment. Mirrors main.py (same flags,
same config generator, same parallel structure), but instead of correcting
Leapfrog against IAS15 it runs the ADAPTIVE method: Leapfrog by default,
switching to IAS15 as soon as the safety score reaches the safety threshold.
Each experiment compares:

    adaptive   the method under test (core/adaptive.py)
    baseline   pure, uncorrected Leapfrog        (cheap, inaccurate)
    IAS15      pure IAS15                        (ground truth, expensive)

    python validate_adaptive.py --safety-threshold 0.5

The safety index formula lives in core/safety_index.py::raw_safety_index and
is NOT filled in yet. Until it is, run with --smoke-test-index to exercise
the pipeline (results are then meaningless).

Flags shared with main.py: --n-experiments --workers --seed --dt --total-time
--correction-threshold --plot. New here: --safety-threshold (0-1),
--index-scale/--index-lo/--index-hi (raw index -> 0-1), --release-fraction,
--smoke-test-index, and the rewind flags --rewind / --checkpoint-count /
--checkpoint-interval / --rewind-back / --ias15-hold-steps (see core/adaptive.py),
and the false-positive detection flags --detect-false-positives /
--false-positive-threshold, the rollback analysis flags --analyze-rollback /
--rollback-clean-threshold / --rollback-min-gain, and the drift-repair flags
--drift-budget / --rewind-to-anchor / --resync-on-switch.

Note on --seed: the default differs from main.py's on purpose. Validation
should use systems the safety index was NOT calibrated on, so pick a seed
range that does not overlap your calibration runs (experiment i uses
seed + i).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from calibration.generator import DEFAULT_CORRECTION_THRESHOLD, generate_calibration_batch
from core.safety_index import (
    INDEX_SCALE_MODES,
    IndexScale,
    check_index_available,
    resolve_raw_index_fn,
)
from validation.runner import DEFAULT_WORKERS, run_validation_batch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Separate from calibration's experiments/configs so the two never overwrite each other.
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "experiments", "validation_configs")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "validation")
CSV_DIR = os.path.join(OUT_DIR, "csv")
FIGURES_DIR = os.path.join(OUT_DIR, "figures")

N_EXPERIMENTS = 16
DEFAULT_VALIDATION_SEED = 10_000


def _unit_interval(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {value}")
    return value


def _release_fraction(text: str) -> float:
    value = float(text)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in (0, 1], got {value}")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def _int_at_least(minimum: int):
    def parse(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be an integer >= {minimum}, got {value}")
        return value

    return parse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (main.py's flags plus the safety-index flags)."""
    parser = argparse.ArgumentParser(description="IRIS adaptive-integrator validation experiment")

    # --- same flags as main.py ---
    parser.add_argument(
        "--n-experiments",
        type=int,
        default=N_EXPERIMENTS,
        help=f"Number of experiments to generate and run (default: {N_EXPERIMENTS}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel worker processes (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_VALIDATION_SEED,
        help=(
            "Base random seed; experiment i uses seed + i (default: "
            f"{DEFAULT_VALIDATION_SEED}). Use a range disjoint from your calibration runs."
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0e-3,
        help="Leapfrog integration timestep (default: 1e-3).",
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=5.0,
        help="Total simulated duration per experiment (default: 5.0).",
    )
    parser.add_argument(
        "--correction-threshold",
        type=float,
        default=DEFAULT_CORRECTION_THRESHOLD,
        help=(
            "RMS position-error threshold that defines a FAILED sample: one where the "
            "calibration procedure would have had to snap Leapfrog back to IAS15. Used "
            f"only for reporting here (default: {DEFAULT_CORRECTION_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--three-body",
        action="store_true",
        help=(
            "Generate only 3-body systems. Without it, experiments cycle through 2, 3, 4 and 5 "
            "bodies. Seeds are unchanged (experiment i still uses seed + i), so a 3-body run "
            "draws different systems than the same seed range does in a mixed run."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate an optional comparison plot for the first experiment.",
    )

    # --- adaptive-method flags ---
    parser.add_argument(
        "--safety-threshold",
        type=_unit_interval,
        default=0.5,
        help=(
            "0-1 safety score at or above which the adaptive method uses IAS15 "
            "(default: 0.5). Higher = tolerate more risk, use IAS15 less."
        ),
    )
    parser.add_argument(
        "--index-scale",
        choices=INDEX_SCALE_MODES,
        default="sigmoid",
        help=(
            "How the raw safety index is converted to a 0-1 score: 'sigmoid' for a "
            "logit/log-odds index (default), 'minmax' or 'log_minmax' for indices on an "
            "arbitrary linear / positive log scale (these need --index-lo and --index-hi)."
        ),
    )
    parser.add_argument(
        "--index-lo",
        type=float,
        default=None,
        help="Raw-index value mapped to score 0 (minmax / log_minmax only).",
    )
    parser.add_argument(
        "--index-hi",
        type=float,
        default=None,
        help="Raw-index value mapped to score 1 (minmax / log_minmax only).",
    )
    parser.add_argument(
        "--release-fraction",
        type=_release_fraction,
        default=0.8,
        help=(
            "Hysteresis: once in IAS15, return to Leapfrog only when the score drops "
            "below release-fraction * safety-threshold (default: 0.8; 1.0 = no hysteresis)."
        ),
    )
    parser.add_argument(
        "--smoke-test-index",
        action="store_true",
        help=(
            "Use a MEANINGLESS stand-in index (log10 relative velocity) instead of the "
            "real formula, only to test that the pipeline runs end to end."
        ),
    )
    parser.add_argument(
        "--situation-index",
        action="store_true",
        help=(
            "Use the additive situation-weight index (core/situations.py: causal versions of the "
            "situations weighted by fit_situation_weights.py) instead of the fitted-formula index. "
            "Its score is a sigmoid, so keep --index-scale sigmoid. Thresholds from the fit's "
            "out-of-fold table (recall 0.80 / 0.90 / 0.95 / 0.99): 0.011 / 0.0029 / 0.0021 / 7e-5."
        ),
    )
    parser.add_argument(
        "--correct",
        action="store_true",
        help=(
            "Enable online correction, exactly as in main.py: at every sample, if the RMS "
            "position error vs IAS15 exceeds --correction-threshold, snap the state back to "
            "the IAS15 reference. Applied to BOTH the adaptive run and the Leapfrog-only "
            "baseline (which then IS main.py's calibration run). Errors detected then equals "
            "the number of corrections each method needed. Off by default: without it the "
            "error accumulates uncorrected and one failure taints every later sample."
        ),
    )

    # --- rewinding ---
    parser.add_argument(
        "--rewind",
        action="store_true",
        help=(
            "Rewind instead of switching late: keep --checkpoint-count checkpoints (one every "
            "--checkpoint-interval steps) in RAM; when the safety score reaches the threshold "
            "while in Leapfrog, restore the checkpoint --rewind-back places before the latest "
            "(default: the SECOND-LATEST) and continue with IAS15 from there rather than "
            "switching at the current state."
        ),
    )
    parser.add_argument(
        "--rewind-back",
        type=_int_at_least(0),
        default=1,
        help=(
            "How many checkpoints back from the latest one to rewind to (default: 1 = the "
            "second-latest; 2 = third-latest; 0 = the latest). Must be < --checkpoint-count. "
            "If fewer checkpoints exist yet, the oldest is used. Only used with --rewind."
        ),
    )
    parser.add_argument(
        "--checkpoint-count",
        type=_int_at_least(1),
        default=5,
        help="Number of checkpoints kept in RAM (default: 5). Only used with --rewind.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=_int_at_least(1),
        default=10,
        help="Steps (of --dt) between checkpoints (default: 10). Only used with --rewind.",
    )
    parser.add_argument(
        "--ias15-hold-steps",
        type=_int_at_least(0),
        default=0,
        help=(
            "Steps IAS15 keeps being used after the score has dropped below the release "
            "threshold (release-fraction * safety-threshold) (default: 0 = release at once)."
        ),
    )

    # --- false-positive detection ---
    parser.add_argument(
        "--detect-false-positives",
        action="store_true",
        help=(
            "Classify every IAS15 stretch as a false positive (IAS15 was not needed) or not. "
            "From the moment IAS15 takes over (for a rewind: from the restored checkpoint) a "
            "shadow Leapfrog runs in lockstep from the same state; if its RMS position "
            "difference from the actual state stays below --false-positive-threshold until "
            "the stretch ends, the trigger was a false positive. Detection only - does not "
            "change the integration. Writes <sim>_stretches.csv per experiment and adds fp "
            "columns to the summary. Costs one extra Leapfrog step per IAS15 step."
        ),
    )
    parser.add_argument(
        "--false-positive-threshold",
        type=_positive_float,
        default=None,
        help=(
            "RMS position difference (shadow Leapfrog vs actual) below which a stretch counts "
            "as a false positive (default: --correction-threshold, i.e. Leapfrog would not "
            "have needed a correction). Only used with --detect-false-positives."
        ),
    )

    # --- rollback-depth analysis ---
    parser.add_argument(
        "--analyze-rollback",
        action="store_true",
        help=(
            "Count how many errors a DEEPER rollback could have prevented (needs --rewind). "
            "At every rewind each checkpoint still in the queue is tried as a counterfactual "
            "restore point (IAS15 replayed to the trigger step, compared with the exact IAS15 "
            "state); rewinds are classed adequate / improvable / inadequate and every error is "
            "attributed to the rewinds before it. Validation-only (uses the IAS15 reference as "
            "an oracle); does not change the integration. Writes <sim>_rewinds.csv and "
            "<sim>_rollback_errors.csv per experiment. Costs one short IAS15 replay per queued "
            "checkpoint per rewind."
        ),
    )
    parser.add_argument(
        "--rollback-clean-threshold",
        type=_positive_float,
        default=1e-9,
        help=(
            "Largest RMS position deviation from the exact state at the trigger step that still "
            "counts as an ADEQUATE rewind target (default: 1e-9; the raw deviations are in "
            "<sim>_rewinds.csv, so this can be re-applied offline). Only used with --analyze-rollback."
        ),
    )
    parser.add_argument(
        "--rollback-min-gain",
        type=_positive_float,
        default=0.5,
        help=(
            "A rewind target counts as EFFECTIVE in the rollback analysis if it removes at least "
            "this fraction (0-1] of the un-rewound state's error (default: 0.5). Rewinds where no "
            "queued checkpoint is effective are 'drift_dominated' (slow accumulated Leapfrog "
            "drift that no rollback depth inside the queue can fix). Only used with "
            "--analyze-rollback."
        ),
    )

    # --- drift repair ---
    parser.add_argument(
        "--drift-budget",
        type=float,
        default=0.0,
        help=(
            "Accumulated-drift budget as a fraction of --correction-threshold (default: 0 = off). "
            "While in Leapfrog a shadow Leapfrog with half the step runs alongside; Richardson "
            "gives an estimate of the accumulated error, ~ (4/3)*RMS|x(dt)-x(dt/2)|. When it "
            "reaches budget * correction-threshold the state is RESYNCED to the Richardson "
            "extrapolation (4*x_fine - x_coarse)/3, which removes the leading error term. Costs "
            "~3 force evaluations per Leapfrog step instead of 1. Needs no reference. "
            "Try 0.1-0.5."
        ),
    )
    parser.add_argument(
        "--rewind-to-anchor",
        action="store_true",
        help=(
            "With --rewind: on a trigger rewind to the ANCHOR (the last verified state: the "
            "start, the last --drift-budget resync, the last validation snap, or the end of an "
            "IAS15 stretch that itself began at the anchor) instead of --rewind-back checkpoints, "
            "and replay with IAS15 from there. Repairs accumulated drift at the cost of a long "
            "IAS15 stretch (up to the distance between anchors), so combine it with --drift-budget."
        ),
    )
    parser.add_argument(
        "--resync-on-switch",
        action="store_true",
        help=(
            "Repair the state IAS15 takes over from. The dt/2 shadow Leapfrog runs alongside "
            "Leapfrog and, at every trigger, the Richardson extrapolation (4*x_fine - x_coarse)/3 "
            "is applied to the state IAS15 continues from (with --rewind: the restored checkpoint, "
            "using the shadow saved in it), so Leapfrog error is not frozen into the IAS15 stretch "
            "and cannot pile up from stretch to stretch. Needs no reference and no long rollback; "
            "costs ~3 force evaluations per Leapfrog step. Can be combined with --drift-budget."
        ),
    )
    parser.add_argument(
        "--compensated-sums",
        "--kahan",
        dest="compensated_sums",
        action="store_true",
        help=(
            "Kahan-compensate the position / velocity sums in the Leapfrog step and in the dt/2 (dt/4) "
            "shadows, and keep the compensation through checkpoints and Richardson repairs. Removes the "
            "~1e-16-per-step roundoff that chaos otherwise amplifies into a late threshold error. "
            "Off by default (results are then bit-identical to before). Best with --richardson-levels 3; "
            "only matters when a dt/2 shadow runs (--drift-budget and/or --resync-on-switch)."
        ),
    )
    parser.add_argument(
        "--richardson-levels",
        type=int,
        choices=(2, 3),
        default=2,
        help=(
            "2 (default): dt/2 shadow, 4th-order Richardson repair. 3: also a dt/4 shadow and the 6th-order "
            "repair (64 x_q - 20 x_h + x_c)/45. ~7 instead of ~3 force evaluations per Leapfrog step. "
            "Only useful together with --compensated-sums (otherwise roundoff is the floor)."
        ),
    )
    parser.add_argument(
        "--ias15-stepped",
        "--chaos-control",
        dest="chaos_control",
        action="store_true",
        help=(
            "Also run pure IAS15 forced to stop at EVERY dt, exactly like the adaptive run steps its IAS15 "
            "stretches (same dt, same Python stepping/recording loop as Leapfrog and adaptive). Used for (a) "
            "a like-for-like IAS15 TIMING baseline in the performance block, and (b) the chaos-floor control: "
            "adaptive errors where this run has failed too are not fixable. Adds ~one extra IAS15 run per "
            "experiment and a control_position_error CSV column. Off by default (IAS15-only timing is then "
            "REBOUND's native run, which only stops at sample times)."
        ),
    )
    return parser.parse_args()


def make_comparison_plot(csv_dir: str, figures_dir: str, safety_threshold: float) -> str | None:
    """
    Plot position error (adaptive vs baseline), the safety score against the
    threshold, and the share of IAS15 steps, for the first experiment CSV.

    Returns:
        Path to the saved figure, or None if no CSVs were found.
    """
    import glob

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*_validation.csv")))
    if not csv_files:
        return None
    df = pd.read_csv(csv_files[0])
    sim_id = df["simulation_id"].iloc[0]

    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(df["time"], df["baseline_position_error"].clip(lower=1e-16), label="Leapfrog only")
    axes[0].plot(df["time"], df["adaptive_position_error"].clip(lower=1e-16), label="Adaptive")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("RMS position error vs IAS15")
    axes[0].set_title(f"Adaptive vs Leapfrog-only vs IAS15: {sim_id}")
    axes[0].legend()

    axes[1].plot(df["time"], df["safety_score"])
    axes[1].axhline(safety_threshold, color="r", linestyle="--", label="safety threshold")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("Safety score (0-1)")
    axes[1].legend()

    axes[2].fill_between(df["time"], df["ias15_step_fraction"], step="mid")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_ylabel("Fraction of steps in IAS15")
    axes[2].set_xlabel("Time")

    fig.tight_layout()
    out_path = os.path.join(figures_dir, f"{sim_id}_adaptive_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    """Run the full validation experiment pipeline."""
    args = parse_args()

    try:
        index_scale = IndexScale(mode=args.index_scale, lo=args.index_lo, hi=args.index_hi)
    except ValueError as exc:
        sys.exit(f"error: invalid --index-scale settings: {exc}")

    if args.smoke_test_index and args.situation_index:
        sys.exit("Choose one of --smoke-test-index and --situation-index, not both.")
    if args.analyze_rollback and not args.rewind:
        sys.exit("error: --analyze-rollback needs --rewind.")
    if args.rewind_to_anchor and not args.rewind:
        sys.exit("error: --rewind-to-anchor needs --rewind.")
    if args.drift_budget < 0:
        sys.exit("error: --drift-budget must be >= 0.")
    if not 0.0 < args.rollback_min_gain <= 1.0:
        sys.exit("error: --rollback-min-gain must be in (0, 1].")
    if args.rewind and args.checkpoint_count < args.rewind_back + 1:
        sys.exit(
            f"error: --checkpoint-count ({args.checkpoint_count}) must be at least "
            f"--rewind-back + 1 ({args.rewind_back + 1})."
        )
    index_source = "smoke" if args.smoke_test_index else ("situations" if args.situation_index else "formula")
    ok, message = check_index_available(resolve_raw_index_fn(index_source))
    if not ok:
        sys.exit(f"error: {message}")
    if args.smoke_test_index:
        print(
            "WARNING: --smoke-test-index is active. The safety index is a meaningless "
            "stand-in; the results below only show that the pipeline runs.\n"
        )

    start_time = time.perf_counter()

    print(f"Generating {args.n_experiments} experiment configurations (seed {args.seed}+i)...")
    config_paths = generate_calibration_batch(
        n_experiments=args.n_experiments,
        configs_dir=CONFIGS_DIR,
        base_seed=args.seed,
        dt=args.dt,
        total_time=args.total_time,
        correction_threshold=args.correction_threshold,
        body_counts=(3,) if args.three_body else None,
    )
    print(f"Generated {len(config_paths)} configurations in {CONFIGS_DIR}\n")
    print(
        f"Launching {len(config_paths)} validation experiments across {args.workers} workers "
        f"(index {index_source}, safety threshold {args.safety_threshold:g}, scale {index_scale.mode}, "
        f"release fraction {args.release_fraction:g}, IAS15 hold {args.ias15_hold_steps} steps, "
        + (
            f"rewind ON: {args.checkpoint_count} checkpoints every {args.checkpoint_interval} steps, "
            f"back {args.rewind_back})..."
            if args.rewind
            else "rewind off)..."
        )
        + (
            f" False-positive detection ON (threshold "
            f"{args.false_positive_threshold or args.correction_threshold:g})."
            if args.detect_false_positives
            else ""
        )
        + (
            f" Rollback analysis ON (clean threshold {args.rollback_clean_threshold:g}, "
            f"min gain {args.rollback_min_gain:g})."
            if args.analyze_rollback
            else ""
        )
        + (
            f" Drift budget {args.drift_budget:g} x {args.correction_threshold:g} (dt/2 shadow + Richardson resync)."
            if args.drift_budget > 0
            else ""
        )
        + (" Rewinding to the ANCHOR." if args.rewind_to_anchor else "")
        + (" Resync-on-switch ON (Richardson repair of the IAS15 handoff state)." if args.resync_on_switch else "")
        + (" Compensated (Kahan) sums ON." if args.compensated_sums else "")
        + (" IAS15 stepped per dt (timing baseline + chaos control) ON." if args.chaos_control else "")
        + (f" Richardson levels {args.richardson_levels}." if args.richardson_levels != 2 else "")
    )
    results = run_validation_batch(
        config_paths=config_paths,
        csv_dir=CSV_DIR,
        safety_threshold=args.safety_threshold,
        index_scale=index_scale,
        release_fraction=args.release_fraction,
        index_source=index_source,
        n_workers=args.workers,
        correct=args.correct,
        rewind=args.rewind,
        checkpoint_count=args.checkpoint_count,
        checkpoint_interval=args.checkpoint_interval,
        ias15_hold_steps=args.ias15_hold_steps,
        rewind_back=args.rewind_back,
        detect_false_positives=args.detect_false_positives,
        false_positive_threshold=args.false_positive_threshold,
        analyze_rollback=args.analyze_rollback,
        rollback_clean_threshold=args.rollback_clean_threshold,
        rollback_min_gain=args.rollback_min_gain,
        drift_budget=args.drift_budget,
        rewind_to_anchor=args.rewind_to_anchor,
        resync_on_switch=args.resync_on_switch,
        richardson_levels=args.richardson_levels,
        compensated_sums=args.compensated_sums,
        chaos_control=args.chaos_control,
    )
    elapsed = time.perf_counter() - start_time

    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    print("\n===== IRIS Adaptive Validation Summary =====")
    print(f"Experiments completed: {len(successful)}/{len(results)}")
    if failed:
        print(f"Experiments failed:    {len(failed)}")
        for r in failed:
            print(f"  - {r.simulation_id}: {r.error}")

    if successful:
        summary = pd.DataFrame([r.summary for r in successful]).sort_values("simulation_id")
        os.makedirs(OUT_DIR, exist_ok=True)
        summary_path = os.path.join(OUT_DIR, "validation_summary.csv")
        summary.to_csv(summary_path, index=False)

        def mean(col: str) -> float:
            return float(np.nanmean(summary[col]))

        print(f"Safety threshold:      {args.safety_threshold:g}")
        print("--- accuracy vs IAS15 (mean over experiments) ---")
        print(f"Mean position error:   adaptive {mean('adaptive_mean_pos_err'):.3e} | "
              f"Leapfrog-only {mean('baseline_mean_pos_err'):.3e}")
        print(f"Max position error:    adaptive {mean('adaptive_max_pos_err'):.3e} | "
              f"Leapfrog-only {mean('baseline_max_pos_err'):.3e}")
        print(f"Samples over {args.correction_threshold:g}:   adaptive "
              f"{100 * mean('adaptive_frac_over_threshold'):.2f}% | Leapfrog-only "
              f"{100 * mean('baseline_frac_over_threshold'):.2f}%")
        total_samples = int(summary["n_samples"].sum())
        err_label = "Errors corrected:" if args.correct else "Errors detected:"
        print(f"{err_label:<22} adaptive {int(summary['adaptive_n_errors'].sum())} | "
              f"Leapfrog-only {int(summary['baseline_n_errors'].sum())} "
              f"(of {total_samples} samples vs IAS15, error = RMS position error > "
              f"{args.correction_threshold:g})")
        if args.correct:
            print("Correction:            ON (each error snapped the state back to IAS15, as in main.py)")
        n_with_errors = int((summary["adaptive_n_errors"] > 0).sum())
        print(f"Experiments w/ errors: adaptive {n_with_errors}/{len(summary)} | Leapfrog-only "
              f"{int((summary['baseline_n_errors'] > 0).sum())}/{len(summary)}")
        print(f"Final energy drift:    adaptive {mean('adaptive_final_energy_drift'):.3e} | "
              f"Leapfrog-only {mean('baseline_final_energy_drift'):.3e} | "
              f"IAS15 {mean('ias15_final_energy_drift'):.3e}")
        print("--- cost ---")
        print(f"IAS15 share of steps:  {100 * mean('ias15_step_fraction'):.1f}%")
        print(f"Integrator switches:   {int(summary['n_switches'].sum())} total "
              f"({summary['n_switches'].mean():.1f} per experiment)")
        if args.rewind:
            print(f"Rewinds:               {int(summary['n_rewinds'].sum())} total "
                  f"({summary['n_rewinds'].mean():.1f} per experiment), "
                  f"{int(summary['n_rewound_steps'].sum())} steps replayed")
        if args.detect_false_positives:
            n_str = int(summary["n_stretches"].sum())
            n_fp = int(summary["n_false_positive_stretches"].sum())
            fp_steps = int(summary["fp_ias15_steps"].sum())
            ias15_steps = int(summary["n_ias15_steps"].sum())
            threshold_used = args.false_positive_threshold or args.correction_threshold
            print(f"False positives:       {n_fp}/{n_str} IAS15 stretches "
                  f"({100 * n_fp / n_str if n_str else float('nan'):.1f}%), "
                  f"{fp_steps} of {ias15_steps} IAS15 steps "
                  f"({100 * fp_steps / ias15_steps if ias15_steps else float('nan'):.1f}%) "
                  f"were unneeded (shadow-Leapfrog deviation < {threshold_used:g})")
        if args.analyze_rollback:
            n_err = int(summary["n_rollback_errors"].sum())
            n_imp = int(summary["errors_improvable"].sum())
            n_shallow = int(summary["errors_accumulated_drift"].sum())
            n_after = int(summary["errors_after_adequate"].sum())
            n_und = int(summary["errors_undetected"].sum())
            print(f"Rollback analysis:     {n_imp}/{n_err} errors preventable by a deeper rollback "
                  f"(a deeper queued checkpoint was effective)")
            print(f"                       {n_shallow} are accumulated drift (no queued checkpoint helps), "
                  f"{n_after} follow only effective rewinds (not rollback-related), "
                  f"{n_und} undetected (no rewind)")
            print(f"                       rewinds: {int(summary['rewinds_adequate'].sum())} adequate, "
                  f"{int(summary['rewinds_improvable'].sum())} improvable, "
                  f"{int(summary['rewinds_drift_dominated'].sum())} drift-dominated "
                  f"(min gain {args.rollback_min_gain:g})")
        if args.drift_budget > 0 or args.resync_on_switch:
            print(f"Drift resyncs:         {int(summary['n_resyncs'].sum())} total "
                  f"({summary['n_resyncs'].mean():.1f} per experiment)")
            if args.resync_on_switch:
                ratio = summary["resync_median_dev_ratio"].dropna()
                if len(ratio):
                    print(f"                       median true error after/before a resync: "
                          f"{ratio.median():.2e} (lower is better; needs --analyze-rollback)")
        print(f"Wall-clock (sum, s):   adaptive {summary['wall_adaptive_s'].sum():.2f} | "
              f"Leapfrog-only {summary['wall_leapfrog_s'].sum():.2f} | "
              f"IAS15 {summary['wall_ias15_s'].sum():.2f}")
        print("--- performance vs IAS15-only ---")
        has_ctl = "wall_control_s" in summary and summary["wall_control_s"].notna().any()

        def _perf(prefix: str, label: str, ias15_col: str) -> None:
            cols = {"IAS15-only": ias15_col, "Leapfrog": prefix + "leapfrog_s", "adaptive": prefix + "adaptive_s"}
            if any(c not in summary or summary[c].isna().all() for c in cols.values()):
                return
            base = summary[ias15_col]
            print(f"{label}:")
            for name, col in cols.items():
                v = summary[col]
                r = (v / base).replace([np.inf, -np.inf], np.nan).dropna()
                print(f"  {name:<10} total {v.sum():9.2f} s | mean {v.mean():7.2f} s/exp | x IAS15-only: "
                      f"{v.sum() / base.sum():7.2f} (median {r.median():.2f}, min {r.min():.2f}, max {r.max():.2f})")
            n_fast = int((summary[cols["adaptive"]] < base).sum())
            print(f"  adaptive faster than IAS15-only in {n_fast}/{len(summary)} experiments")

        if has_ctl:
            print("IAS15-only = IAS15 stepped at every dt: same dt and same Python stepping/recording loop as the others")
            _perf("wall_", "Wall-clock", "wall_control_s")
            _perf("cpu_", "CPU time (per worker process; unaffected by core contention)", "cpu_control_s")
            print(f"Native IAS15 (REBOUND C, stops only at sample times; the ground truth): "
                  f"{summary['wall_ias15_s'].sum():.2f} s wall total")
            if "errors_floor" in summary and summary["errors_floor"].notna().any():
                print(f"Error attribution:     adaptive errors at the chaos floor {int(summary['errors_floor'].sum())} "
                      f"(pure IAS15 fails too) | marginal {int(summary['errors_marginal'].sum())} | "
                      f"fixable {int(summary['errors_fixable'].sum())}")
        else:
            print("IAS15-only = REBOUND's native adaptive-step run (C, stops only at sample times), NOT like-for-like")
            print("with Leapfrog/adaptive (Python, every dt). Add --ias15-stepped for a same-configuration baseline.")
            _perf("wall_", "Wall-clock", "wall_ias15_s")
            _perf("cpu_", "CPU time (per worker process; unaffected by core contention)", "cpu_ias15_s")
        print("Note:                  with --workers above the core count, wall-clock includes waiting for a core (use the")
        print("                       CPU-time lines); --analyze-rollback adds oracle IAS15 work to the adaptive time.")
        print(f"Summary CSV:           {summary_path}")
    print(f"Total runtime:         {elapsed:.2f} s")
    print("============================================\n")

    if args.plot:
        plot_path = make_comparison_plot(CSV_DIR, FIGURES_DIR, args.safety_threshold)
        if plot_path:
            print(f"Comparison plot saved to: {plot_path}")
        else:
            print("No validation CSV files found; skipped plot.")


if __name__ == "__main__":
    main()
