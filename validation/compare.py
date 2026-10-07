"""
validation/compare.py

Turns the three trajectories of one validation experiment into
  * a per-sample, system-level DataFrame (one CSV per experiment), and
  * a one-row summary dict (collected into validation_summary.csv).

The three trajectories, all sampled on the same grid:

    ias15     pure IAS15 from the initial state: the ground truth.
    baseline  pure Leapfrog, NO correction and NO switching: the cheap
              method the adaptive integrator is trying to improve on.
    adaptive  core.adaptive.run_adaptive: Leapfrog by default, IAS15 when the
              safety score reaches the safety threshold.

Accuracy is always measured against `ias15` with the same RMS position error
used everywhere else (core.metrics.position_error). The calibration's
`correction_threshold` is reused as the definition of a FAILED sample: a
sample whose RMS position error exceeds it is one where the calibration
procedure would have had to snap Leapfrog back to IAS15.

Cost is reported two ways because wall-clock time in this pure-Python
implementation depends heavily on implementation details (per-step feature
computation, REBOUND call overhead): the machine-independent
`ias15_step_fraction` (share of steps IAS15 had to take), and measured
wall-clock seconds for all three methods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibration.generator import ExperimentConfig
from core.adaptive import AdaptiveLog
from core.leapfrog import Trajectory
from core.metrics import angular_momentum_drift, energy_drift, position_error, velocity_error
from core.physics import compute_angular_momentum, compute_total_energy


def _resync_dev_ratio(log: AdaptiveLog) -> float | None:
    """Median of (true error after resync) / (true error before), from the rollback analysis."""
    ratios = [
        e.dev_after / e.dev_before
        for e in log.resync_events
        if e.dev_before is not None and e.dev_after is not None and e.dev_before > 0
    ]
    if not ratios:
        return None
    ratios.sort()
    return ratios[len(ratios) // 2]


def _count_errors(log: AdaptiveLog, category: str) -> int | None:
    """Errors attributed to `category` by the rollback analysis (None if it was off)."""
    if not log.rollback_analysis:
        return None
    return sum(1 for e in log.rollback_errors if e.category == category)


def _count_rewinds(log: AdaptiveLog, category: str) -> int | None:
    """Rewinds classed as `category` by the rollback analysis (None if it was off)."""
    if not log.rollback_analysis:
        return None
    return sum(1 for r in log.rewind_events if r.analysis is not None and r.analysis.category == category)


def _energy_series(traj: Trajectory, config: ExperimentConfig) -> np.ndarray:
    return np.array(
        [
            compute_total_energy(
                traj.positions[i], traj.velocities[i], traj.masses, g=config.g, softening=config.softening
            )
            for i in range(traj.times.shape[0])
        ]
    )


def _angmom_drift_series(traj: Trajectory) -> np.ndarray:
    l0 = compute_angular_momentum(traj.positions[0], traj.velocities[0], traj.masses)
    return np.array(
        [
            angular_momentum_drift(
                compute_angular_momentum(traj.positions[i], traj.velocities[i], traj.masses), l0
            )
            for i in range(traj.times.shape[0])
        ]
    )


def _raw_positions(traj: Trajectory) -> np.ndarray:
    """Pre-correction positions if the run recorded them, else the trajectory itself."""
    return traj.pre_check_positions if traj.pre_check_positions is not None else traj.positions


def build_validation_frame(
    config: ExperimentConfig,
    ias15: Trajectory,
    baseline: Trajectory,
    adaptive: Trajectory,
    log: AdaptiveLog,
    control: Trajectory | None = None,
) -> pd.DataFrame:
    """
    Build the per-sample, system-level comparison table for one experiment.

    Raises:
        ValueError: If the three trajectories are not on the same sample grid.
    """
    n_s = ias15.times.shape[0]
    if baseline.times.shape[0] != n_s or adaptive.times.shape[0] != n_s:
        raise ValueError(
            "trajectories must share a sample grid; got lengths "
            f"ias15={n_s}, baseline={baseline.times.shape[0]}, adaptive={adaptive.times.shape[0]}"
        )

    idx = range(n_s)
    e_ias, e_base, e_ada = (_energy_series(t, config) for t in (ias15, baseline, adaptive))
    e0 = e_ias[0]

    df = pd.DataFrame(
        {
            "simulation_id": config.simulation_id,
            "seed": config.seed,
            "body_count": config.body_count,
            "time": ias15.times,
            "adaptive_position_error": [position_error(adaptive.positions[i], ias15.positions[i]) for i in idx],
            "baseline_position_error": [position_error(baseline.positions[i], ias15.positions[i]) for i in idx],
            # Error of the RAW state at each sample, before any snap-back. Identical to
            # the columns above when correction is off; when it is on, THIS is what the
            # error counts use (a detected error = a correction that had to be applied).
            "adaptive_pre_check_error": [
                position_error(_raw_positions(adaptive)[i], ias15.positions[i]) for i in idx
            ],
            "baseline_pre_check_error": [
                position_error(_raw_positions(baseline)[i], ias15.positions[i]) for i in idx
            ],
            "adaptive_velocity_error": [velocity_error(adaptive.velocities[i], ias15.velocities[i]) for i in idx],
            "baseline_velocity_error": [velocity_error(baseline.velocities[i], ias15.velocities[i]) for i in idx],
            "adaptive_energy_drift": [energy_drift(e, e0) for e in e_ada],
            "baseline_energy_drift": [energy_drift(e, e0) for e in e_base],
            "ias15_energy_drift": [energy_drift(e, e0) for e in e_ias],
            "adaptive_angular_momentum_drift": _angmom_drift_series(adaptive),
            "baseline_angular_momentum_drift": _angmom_drift_series(baseline),
            "active_integrator": log.active_integrator,
            "ias15_step_fraction": log.ias15_step_fraction,
            "raw_safety_index": log.raw_index,
            "safety_score": log.safety_score,
            "n_switches": log.n_switches,
        }
    )
    thr = config.correction_threshold
    df["adaptive_exceeds_threshold"] = df["adaptive_pre_check_error"] > thr
    df["baseline_exceeds_threshold"] = df["baseline_pre_check_error"] > thr
    if control is not None:  # chaos-floor control: IAS15 stepped every dt vs the IAS15 reference
        if control.times.shape[0] != n_s:
            raise ValueError(f"control has {control.times.shape[0]} samples, expected {n_s}")
        df["control_position_error"] = [position_error(control.positions[i], ias15.positions[i]) for i in idx]
    return df


def _first_exceed_time(df: pd.DataFrame, flag_col: str) -> float:
    hit = df.loc[df[flag_col], "time"]
    return float(hit.min()) if len(hit) else float("nan")


_CHAOS_MARGIN = 0.1  # control error >= this * threshold counts as 'marginal'


def _chaos_fields(df: pd.DataFrame, thr: float) -> dict:
    """
    Is each adaptive error fixable? Compare it with the chaos-floor control (pure IAS15
    stepped every dt, scored against the IAS15 reference), sample by sample.

        floor     control error >= threshold there: even pure IAS15 fails: NOT fixable
        marginal  control error >= 0.1 * threshold: nearly at the floor
        fixable   control still accurate there: a better algorithm would not have failed

    Also: the time the control first fails (chaos_floor_time), the exponential growth
    rate of the control's error (chaos_rate, per time unit; 1/rate is the Lyapunov
    time), the gap between the floor time and the first adaptive error, and
    residual_factor_needed = exp(rate * gap): how many times smaller the adaptive
    run's injected error would have to be to reach the floor (NaN if unknown).
    Everything is None when the control was not run.
    """
    keys = (
        "chaos_floor_time", "chaos_rate", "errors_floor", "errors_marginal", "errors_fixable",
        "floor_gap", "residual_factor_needed",
    )
    if "control_position_error" not in df.columns:
        return {k: None for k in keys}
    t = df["time"].to_numpy()
    ctrl = df["control_position_error"].to_numpy()
    hit = t[ctrl > thr]
    t_floor = float(hit.min()) if len(hit) else float("nan")
    grow = (ctrl > 1e-13) & (ctrl < 1e-3) & (t > 0)
    rate = float(np.polyfit(t[grow], np.log(ctrl[grow]), 1)[0]) if grow.sum() >= 5 else float("nan")
    at_err = ctrl[df["adaptive_exceeds_threshold"].to_numpy()]
    n_floor = int((at_err >= thr).sum())
    n_marg = int(((at_err >= _CHAOS_MARGIN * thr) & (at_err < thr)).sum())
    first = _first_exceed_time(df, "adaptive_exceeds_threshold")
    gap = t_floor - first if np.isfinite(t_floor) and np.isfinite(first) else float("nan")
    factor = float(np.exp(rate * gap)) if np.isfinite(rate) and rate > 0 and np.isfinite(gap) else float("nan")
    return {
        "chaos_floor_time": t_floor,
        "chaos_rate": rate,
        "errors_floor": n_floor,
        "errors_marginal": n_marg,
        "errors_fixable": int(len(at_err)) - n_floor - n_marg,
        "floor_gap": gap,
        "residual_factor_needed": factor,
    }


def summarize_validation(
    config: ExperimentConfig,
    df: pd.DataFrame,
    log: AdaptiveLog,
    wall_ias15: float,
    wall_leapfrog: float,
    wall_adaptive: float,
    correct: bool = False,
    wall_control: float | None = None,
) -> dict:
    """
    Reduce one experiment's comparison table to a flat summary dict.

    Args:
        config: The experiment configuration.
        df: Output of build_validation_frame.
        log: The adaptive run's log.
        wall_ias15: Wall-clock seconds the pure IAS15 reference run took.
        wall_leapfrog: Wall-clock seconds the pure Leapfrog baseline took.
        wall_adaptive: Wall-clock seconds the adaptive run took.
    """
    a, b = df["adaptive_position_error"], df["baseline_position_error"]
    a_mean, b_mean = float(a.mean()), float(b.mean())
    return {
        "simulation_id": config.simulation_id,
        "seed": config.seed,
        "body_count": config.body_count,
        "n_samples": int(len(df)),
        "correction_threshold": config.correction_threshold,
        # accuracy vs IAS15 ground truth
        "adaptive_mean_pos_err": a_mean,
        "adaptive_max_pos_err": float(a.max()),
        "adaptive_final_pos_err": float(a.iloc[-1]),
        "baseline_mean_pos_err": b_mean,
        "baseline_max_pos_err": float(b.max()),
        "baseline_final_pos_err": float(b.iloc[-1]),
        "mean_error_reduction": (1.0 - a_mean / b_mean) if b_mean > 0 else float("nan"),
        # failures = samples where calibration would have needed a correction
        "adaptive_frac_over_threshold": float(df["adaptive_exceeds_threshold"].mean()),
        "baseline_frac_over_threshold": float(df["baseline_exceeds_threshold"].mean()),
        "adaptive_n_errors": int(df["adaptive_exceeds_threshold"].sum()),
        "baseline_n_errors": int(df["baseline_exceeds_threshold"].sum()),
        "correction_enabled": bool(correct),
        "adaptive_first_exceed_time": _first_exceed_time(df, "adaptive_exceeds_threshold"),
        "baseline_first_exceed_time": _first_exceed_time(df, "baseline_exceeds_threshold"),
        # conservation
        "adaptive_final_energy_drift": float(df["adaptive_energy_drift"].iloc[-1]),
        "baseline_final_energy_drift": float(df["baseline_energy_drift"].iloc[-1]),
        "ias15_final_energy_drift": float(df["ias15_energy_drift"].iloc[-1]),
        # cost
        "n_steps": log.n_steps,
        "n_ias15_steps": log.n_ias15_steps,
        "ias15_step_fraction": log.n_ias15_steps / log.n_steps if log.n_steps else float("nan"),
        "n_switches": len(log.switch_events),
        "n_rewinds": log.n_rewinds,
        "n_rewound_steps": log.n_rewound_steps,
        # false-positive detection (None -> NaN when it was not enabled)
        "n_stretches": len(log.stretches) if log.fp_detection else None,
        "n_false_positive_stretches": (
            sum(s.false_positive for s in log.stretches) if log.fp_detection else None
        ),
        "fp_ias15_steps": (
            sum(s.n_steps for s in log.stretches if s.false_positive) if log.fp_detection else None
        ),
        # rollback-depth analysis (None -> NaN when it was not enabled)
        "n_rollback_errors": len(log.rollback_errors) if log.rollback_analysis else None,
        "errors_improvable": _count_errors(log, "improvable"),
        "errors_accumulated_drift": _count_errors(log, "accumulated_drift"),
        "errors_after_adequate": _count_errors(log, "after_adequate_rewinds"),
        "errors_undetected": _count_errors(log, "undetected"),
        "rewinds_adequate": _count_rewinds(log, "adequate"),
        "rewinds_improvable": _count_rewinds(log, "improvable"),
        "rewinds_drift_dominated": _count_rewinds(log, "drift_dominated"),
        "n_resyncs": log.n_resyncs if (log.drift_budget > 0 or log.resync_on_switch) else None,
        "n_switch_resyncs": (
            sum(1 for e in log.resync_events if e.kind == "switch") if log.resync_on_switch else None
        ),
        "resync_median_dev_ratio": _resync_dev_ratio(log),
        **_chaos_fields(df, config.correction_threshold),
        "wall_control_s": wall_control,
        "wall_ias15_s": wall_ias15,
        "wall_leapfrog_s": wall_leapfrog,
        "wall_adaptive_s": wall_adaptive,
    }
