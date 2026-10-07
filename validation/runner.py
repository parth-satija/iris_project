"""
validation/runner.py

Runs validation experiments in parallel (same ProcessPoolExecutor pattern as
calibration/runner.py). Each worker independently:

    1. Loads one experiment's JSON configuration.
    2. Runs pure IAS15                       -> ground truth (timed).
    3. Runs pure, uncorrected Leapfrog       -> cheap baseline (timed).
    4. Runs the adaptive Leapfrog <-> IAS15  -> the method under test (timed).
    5. Builds the per-sample comparison table and a one-row summary, and
       saves the table as <simulation_id>_validation.csv.

Workers are handed only picklable primitives (the raw-index function is
passed by NAME and resolved inside the worker).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass

import pandas as pd

from calibration.generator import load_experiment_config
from calibration.runner import DEFAULT_WORKERS, _config_to_system
from core.adaptive import run_adaptive
from core.leapfrog import run_leapfrog, run_leapfrog_with_correction
from core.rebound_reference import run_ias15, run_ias15_stepped
from core.safety_index import IndexScale, resolve_raw_index_fn
from validation.compare import build_validation_frame, summarize_validation

__all__ = ["DEFAULT_WORKERS", "ValidationResult", "run_single_validation", "run_validation_batch"]


@dataclass
class ValidationResult:
    """
    Outcome of one worker's validation experiment.

    Attributes:
        simulation_id: Identifier of the experiment.
        csv_path: Path of the per-sample comparison CSV, or None on failure.
        summary: One-row summary (see validation.compare.summarize_validation),
            or None on failure.
        elapsed_seconds: Wall-clock time the worker took in total.
        error: Error message if the worker failed, else None.
    """

    simulation_id: str
    csv_path: str | None
    summary: dict | None
    elapsed_seconds: float
    error: str | None = None


def _write_rollback_csvs(log, csv_dir: str, simulation_id: str) -> None:
    """<sim>_rewinds.csv (one row per rewind, every candidate depth) and <sim>_rollback_errors.csv."""
    rows = []
    for r in log.rewind_events:
        a = r.analysis
        chosen_dev = a.dev_anchor if a.chosen_depth < 0 else a.dev_by_depth[a.chosen_depth]
        row = {
            "step": r.step,
            "to_step": r.to_step,
            "time": r.time,
            "score": r.score,
            "chosen_depth": r.depth,
            "category": a.category,
            "dev_leapfrog": a.dev_leapfrog,
            "dev_chosen": chosen_dev,
            "best_depth": a.best_depth,
            "dev_best": a.dev_by_depth[a.best_depth] if a.dev_by_depth else float("nan"),
            "min_adequate_depth": a.min_adequate_depth,
            "anchor_step": a.anchor_step,
            "dev_anchor": a.dev_anchor,
        }
        for d, (dev, st) in enumerate(zip(a.dev_by_depth, a.steps_by_depth)):
            row[f"dev_depth{d}"] = dev
            row[f"step_depth{d}"] = st
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(csv_dir, f"{simulation_id}_rewinds.csv"), index=False)
    pd.DataFrame([asdict(e) for e in log.rollback_errors]).to_csv(
        os.path.join(csv_dir, f"{simulation_id}_rollback_errors.csv"), index=False
    )
    if log.resync_events:  # per-repair quality: dev_before / dev_after are true errors vs the oracle
        pd.DataFrame([asdict(e) for e in log.resync_events]).to_csv(
            os.path.join(csv_dir, f"{simulation_id}_resyncs.csv"), index=False
        )


def run_single_validation(
    config_path: str,
    csv_dir: str,
    safety_threshold: float,
    index_scale: IndexScale,
    release_fraction: float,
    index_source: str,
    correct: bool = False,
    rewind: bool = False,
    checkpoint_count: int = 5,
    checkpoint_interval: int = 10,
    ias15_hold_steps: int = 0,
    rewind_back: int = 1,
    detect_false_positives: bool = False,
    false_positive_threshold: float | None = None,
    analyze_rollback: bool = False,
    rollback_clean_threshold: float = 1e-9,
    rollback_min_gain: float = 0.5,
    drift_budget: float = 0.0,
    rewind_to_anchor: bool = False,
    resync_on_switch: bool = False,
    chaos_control: bool = False,
    richardson_levels: int = 2,
    compensated_sums: bool = False,
) -> ValidationResult:
    """
    Execute one validation experiment end to end. Top-level and picklable so
    it can be submitted to a ProcessPoolExecutor.

    Args:
        config_path: Path to the experiment's JSON configuration.
        csv_dir: Directory to write the comparison CSV into.
        safety_threshold: 0-1 score at or above which IAS15 is used.
        index_scale: How the raw safety index is converted to a 0-1 score.
        release_fraction: Hysteresis: return to Leapfrog below
            release_fraction * safety_threshold.
        index_source: "formula" (core.safety_index.raw_safety_index) or
            "smoke" (meaningless stand-in, for pipeline testing only).
        correct: If True, BOTH the adaptive run and the Leapfrog baseline snap
            back to the IAS15 reference whenever the RMS position error exceeds
            config.correction_threshold (the exact rule main.py uses). The
            baseline is then run_leapfrog_with_correction, i.e. main.py's
            calibration run.
        rewind: Adaptive run only: on a safety trigger, restore the second-latest
            checkpoint and continue with IAS15 from there (see core/adaptive.py).
        checkpoint_count: Checkpoints kept in RAM (rewind only).
        checkpoint_interval: Steps between checkpoints (rewind only).
        ias15_hold_steps: Steps IAS15 is kept after the score falls below the
            release threshold.
        rewind_back: Checkpoints before the latest to restore (1 = second-latest).
        detect_false_positives: Adaptive run only: classify every IAS15 stretch as a
            false positive or not with a shadow Leapfrog (see core/adaptive.py) and
            write <simulation_id>_stretches.csv next to the comparison CSV.
        false_positive_threshold: RMS position difference below which a stretch is a
            false positive. None -> config.correction_threshold.
        analyze_rollback: Adaptive run only (needs rewind): attribute every error to the
            rewinds before it and count how many a deeper rollback could have prevented,
            using the IAS15 run as an exact oracle (see core/adaptive.py). Writes
            <simulation_id>_rewinds.csv and <simulation_id>_rollback_errors.csv.
        rollback_clean_threshold: Largest RMS deviation still called 'adequate' for a
            rollback candidate.
        rollback_min_gain: A rollback target is 'effective' in the analysis if it removes
            at least this fraction of the un-rewound state's error.
        drift_budget: Adaptive run only: fraction of the correction threshold at which
            the dt/2-shadow drift estimate triggers a Richardson resync (0 = off).
        rewind_to_anchor: Adaptive run only (needs rewind): rewind to the last verified
            state (start / resync / snap) instead of --rewind-back checkpoints.
        resync_on_switch: Adaptive run only: repair the state IAS15 takes over from with
            the Richardson extrapolation (dt/2 shadow), so Leapfrog error is not frozen in.
        chaos_control: Also run pure IAS15 stepped every dt (the way the adaptive run steps
            its IAS15 stretches) and score it against the IAS15 reference. Its error is the
            chaos floor: adaptive errors where it has failed too are NOT fixable (see
            validation.compare._chaos_fields). Adds a control_position_error CSV column.
            It is also the like-for-like IAS15 TIMING baseline (same dt, same Python stepping
            loop as Leapfrog and the adaptive run); wall_control_s / cpu_control_s.
    """
    start = time.perf_counter()
    config = load_experiment_config(config_path)
    try:
        system = _config_to_system(config)

        t0 = time.perf_counter()
        c0 = time.process_time()
        ias15 = run_ias15(system, total_time=config.total_time, sample_interval=config.sample_interval)
        wall_ias15 = time.perf_counter() - t0
        cpu_ias15 = time.process_time() - c0

        control = None
        wall_control = None
        cpu_control = None
        if chaos_control:
            t0 = time.perf_counter()
            c0 = time.process_time()
            control = run_ias15_stepped(
                system, total_time=config.total_time, sample_interval=config.sample_interval, dt=config.dt
            )
            wall_control = time.perf_counter() - t0
            cpu_control = time.process_time() - c0

        t0 = time.perf_counter()
        c0 = time.process_time()
        if correct:
            baseline, _ = run_leapfrog_with_correction(
                system,
                dt=config.dt,
                total_time=config.total_time,
                reference_positions=ias15.positions,
                reference_velocities=ias15.velocities,
                reference_times=ias15.times,
                error_threshold=config.correction_threshold,
            )
        else:
            baseline = run_leapfrog(
                system, dt=config.dt, total_time=config.total_time, sample_interval=config.sample_interval
            )
        wall_leapfrog = time.perf_counter() - t0
        cpu_leapfrog = time.process_time() - c0

        t0 = time.perf_counter()
        c0 = time.process_time()
        adaptive, log = run_adaptive(
            system,
            dt=config.dt,
            total_time=config.total_time,
            sample_interval=config.sample_interval,
            safety_threshold=safety_threshold,
            index_scale=index_scale,
            raw_index_fn=resolve_raw_index_fn(index_source),
            release_fraction=release_fraction,
            reference=ias15 if correct else None,
            error_threshold=config.correction_threshold if correct else None,
            rewind=rewind,
            checkpoint_count=checkpoint_count,
            checkpoint_interval=checkpoint_interval,
            ias15_hold_steps=ias15_hold_steps,
            rewind_back=rewind_back,
            detect_false_positives=detect_false_positives,
            false_positive_threshold=(
                false_positive_threshold
                if false_positive_threshold is not None
                else config.correction_threshold
            ),
            analyze_rollback=analyze_rollback,
            oracle=ias15 if analyze_rollback else None,
            rollback_error_threshold=config.correction_threshold,
            rollback_clean_threshold=rollback_clean_threshold,
            rollback_min_gain=rollback_min_gain,
            drift_budget=drift_budget,
            drift_error_threshold=config.correction_threshold,
            rewind_to_anchor=rewind_to_anchor,
            resync_on_switch=resync_on_switch,
            richardson_levels=richardson_levels,
            compensated_sums=compensated_sums,
        )
        wall_adaptive = time.perf_counter() - t0
        cpu_adaptive = time.process_time() - c0

        df = build_validation_frame(config, ias15, baseline, adaptive, log, control=control)
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"{config.simulation_id}_validation.csv")
        df.to_csv(csv_path, index=False)
        if detect_false_positives:
            pd.DataFrame([asdict(s) for s in log.stretches]).to_csv(
                os.path.join(csv_dir, f"{config.simulation_id}_stretches.csv"), index=False
            )
        if analyze_rollback:
            _write_rollback_csvs(log, csv_dir, config.simulation_id)

        summary = summarize_validation(
            config, df, log, wall_ias15, wall_leapfrog, wall_adaptive, correct=correct,
            wall_control=wall_control,
        )
        # CPU seconds of this worker process per method: unlike wall-clock, unaffected by the
        # workers competing for cores (32 workers on fewer cores inflate every wall time).
        summary.update(
            cpu_ias15_s=cpu_ias15,
            cpu_control_s=cpu_control,
            cpu_leapfrog_s=cpu_leapfrog,
            cpu_adaptive_s=cpu_adaptive,
        )
        return ValidationResult(
            simulation_id=config.simulation_id,
            csv_path=csv_path,
            summary=summary,
            elapsed_seconds=time.perf_counter() - start,
        )
    except Exception as exc:  # noqa: BLE001 - report failures without killing the pool
        return ValidationResult(
            simulation_id=config.simulation_id,
            csv_path=None,
            summary=None,
            elapsed_seconds=time.perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_validation_batch(
    config_paths: list[str],
    csv_dir: str,
    safety_threshold: float,
    index_scale: IndexScale,
    release_fraction: float,
    index_source: str,
    n_workers: int = DEFAULT_WORKERS,
    show_progress: bool = True,
    correct: bool = False,
    rewind: bool = False,
    checkpoint_count: int = 5,
    checkpoint_interval: int = 10,
    ias15_hold_steps: int = 0,
    rewind_back: int = 1,
    detect_false_positives: bool = False,
    false_positive_threshold: float | None = None,
    analyze_rollback: bool = False,
    rollback_clean_threshold: float = 1e-9,
    rollback_min_gain: float = 0.5,
    drift_budget: float = 0.0,
    rewind_to_anchor: bool = False,
    resync_on_switch: bool = False,
    chaos_control: bool = False,
    richardson_levels: int = 2,
    compensated_sums: bool = False,
) -> list[ValidationResult]:
    """
    Run a batch of validation experiments in parallel, printing a line per
    completed experiment.

    Returns:
        ValidationResult objects in COMPLETION order.
    """
    results: list[ValidationResult] = []
    total = len(config_paths)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                run_single_validation,
                path,
                csv_dir,
                safety_threshold,
                index_scale,
                release_fraction,
                index_source,
                correct,
                rewind,
                checkpoint_count,
                checkpoint_interval,
                ias15_hold_steps,
                rewind_back,
                detect_false_positives,
                false_positive_threshold,
                analyze_rollback,
                rollback_clean_threshold,
                rollback_min_gain,
                drift_budget,
                rewind_to_anchor,
                resync_on_switch,
                chaos_control,
                richardson_levels,
                compensated_sums,
            ): path
            for path in config_paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if not show_progress:
                continue
            if result.error is None:
                s = result.summary
                fp_text = (
                    f"fp={int(s['n_false_positive_stretches'])}/{int(s['n_stretches'])}, "
                    if detect_false_positives
                    else ""
                )
                rb_text = (
                    f"deeper-rollback-preventable={int(s['errors_improvable'])}/{int(s['n_rollback_errors'])}, "
                    if analyze_rollback
                    else ""
                )
                rs_text = (
                    f"resyncs={int(s['n_resyncs'])}, " if (drift_budget > 0 or resync_on_switch) else ""
                )
                ch_text = (
                    f"floor/marginal/fixable={int(s['errors_floor'])}/{int(s['errors_marginal'])}/"
                    f"{int(s['errors_fixable'])}, "
                    if chaos_control
                    else ""
                )
                print(
                    f"[{completed}/{total}] OK   {result.simulation_id} "
                    f"(adaptive_err={s['adaptive_mean_pos_err']:.3e} "
                    f"baseline_err={s['baseline_mean_pos_err']:.3e}, "
                    f"ias15_steps={100 * s['ias15_step_fraction']:.1f}%, "
                    f"switches={s['n_switches']}, rewinds={s['n_rewinds']}, {fp_text}{rb_text}{rs_text}{ch_text}"
                    f"errors={s['adaptive_n_errors']}/{s['baseline_n_errors']}, "
                    f"{result.elapsed_seconds:.2f}s)"
                )
            else:
                print(f"[{completed}/{total}] FAIL {result.simulation_id}: {result.error}")

    return results
