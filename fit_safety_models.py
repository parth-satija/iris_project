"""
fit_safety_models.py

Classification-based safety-index analysis: uses BOTH the correction rows
(positives: Leapfrog needed fixing) and the sampled negative rows
(Leapfrog was fine) that calibration/logger.py now writes, and answers
"which combination of Leapfrog-only features best predicts a correction,
and where should safety_threshold sit?".

Run after regenerating calibration data with the negative-example logging
(python main.py):

    python fit_safety_models.py
    python fit_safety_models.py --csv-dir outputs/csv --folds 5 --seed 0

--------------------------------------------------------------------------
What it fits
--------------------------------------------------------------------------
Every candidate (log10 continuous features + the situational boolean
flags) goes into ONE model at a time, standardized, instead of being
hand-mixed:

  * logistic + rv_only     baseline: your current "relative velocity is the
                           safety index" hypothesis.
  * ridge (L2 logistic)    all features. Shrinks correlated features
                           toward sharing weight instead of flipping signs
                           arbitrarily (the acceleration / jerk /
                           nearest-neighbor-distance problem OLS had).
  * lasso (L1 logistic)    all features. Zeroes out redundant ones; if it
                           keeps only relative velocity, that is a
                           data-backed confirmation, not a guess.
  * gradient-boosted trees rv_only / rv+flags / all. A nonlinearity and
                           interaction check: if trees on all features do
                           not beat the linear models, the power-law-style
                           index is not leaving signal on the table; if
                           `rv+flags` beats `rv_only`, situational flags
                           carry information beyond relative velocity.

Regularization strength (C) for ridge/lasso is chosen by an INNER grouped
cross-validation on each training fold only, so the reported scores are
not tuned on their own test data.

--------------------------------------------------------------------------
Methodology notes worth knowing
--------------------------------------------------------------------------
* Cross-validation is GROUPED BY simulation_id. Rows from one experiment
  (and all of its bodies) are strongly correlated; splitting rows
  randomly would put near-duplicates in both train and test and inflate
  every score.
* Label: sample_role == "correction" -> 1, any "negative_*" role -> 0.
* The negative rows are a deliberately enriched sample (see
  calibration/comparer.py: pre-failure, near-miss/risky, and every 20th
  baseline sample), so AUC/PR-AUC on them are measured on a HARDER-than-
  natural mix. The threshold table therefore also reports rates weighted
  by `sampling_weight` (approximately "per real sampled step"); treat
  those as estimates, since hard and baseline strata overlap slightly.
* Features are limited to quantities a real-time check could compute from
  Leapfrog's own state: pre_correction_* (acceleration, jerk, nearest-
  neighbor distance, relative velocity), mass, mass_ratio, and the
  situational flags (computed from the same pre-check state). Anything
  that needs IAS15 (position_error, error_at_trigger,
  system_position_error_pre_check) is deliberately excluded -- it would
  be label leakage.
* `stable` is excluded (it is exactly the complement of the other flags,
  so it would be perfectly collinear), as are constant columns.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    from threadpoolctl import threadpool_limits
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    import sklearn
except ImportError:  # pragma: no cover
    raise SystemExit(
        "This script needs scikit-learn. Install it with:\n    pip install scikit-learn"
    )

CONTINUOUS_SOURCES: dict[str, str] = {
    # model feature name -> source column (all log10-transformed)
    "log_rel_velocity": "pre_correction_relative_velocity",
    "log_jerk": "pre_correction_jerk",
    "log_acceleration": "pre_correction_acceleration",
    "log_nn_distance": "pre_correction_nearest_neighbor_distance",
    "log_mass": "mass",
    "log_mass_ratio": "mass_ratio",
}

FLAG_COLUMNS: tuple[str, ...] = (
    "strong_acceleration",
    "high_jerk",
    "acceleration_spike",
    "close_encounter",
    "rapid_approach",
    "rapid_separation",
    "high_relative_velocity_encounter",
    "near_collision",
    "ejection_escape",
    "temporary_capture",
    "strong_mass_ratio_interaction",
    "closest_pair_switch",
    "involved_in_closest_pair_switch",
    "hierarchical_configuration",
    "involved_in_hierarchical_tight_pair",
    "near_symmetric_configuration",
    "chaotic_scattering",
    "quiescent_regime",
)

C_GRID = np.logspace(-3, 1, 5)
RECALL_TARGETS = (0.80, 0.90, 0.95, 0.99)


# --------------------------------------------------------------------------
# Data loading / feature building
# --------------------------------------------------------------------------
def find_body_csvs(csv_dir: str) -> list[str]:
    matches = []
    for root, _dirs, files in os.walk(csv_dir):
        for name in files:
            if fnmatch.fnmatch(name, "*_body*.csv") and not fnmatch.fnmatch(
                name, "*_situations.csv"
            ):
                matches.append(os.path.join(root, name))
    return sorted(matches)


def _read_one_csv(path: str, needed: set):
    """Return (status, DataFrame | None); status is 'ok', 'empty' or 'old'."""
    with open(path, "r", encoding="utf-8") as fh:
        fh.readline()
        if not fh.readline():
            return "empty", None
    df = pd.read_csv(path)
    if not needed.issubset(df.columns):
        return "old", None
    return "ok", df


def load_data(csv_dir: str, n_jobs: int = 1) -> pd.DataFrame:
    needed = {"sample_role", "sampling_weight", "simulation_id"} | set(
        CONTINUOUS_SOURCES.values()
    )
    paths = find_body_csvs(csv_dir)
    # CSV parsing is CPU-bound, so read files in parallel worker processes.
    results = Parallel(n_jobs=n_jobs)(delayed(_read_one_csv)(p, needed) for p in paths)
    frames = [df for status, df in results if status == "ok"]
    n_empty = sum(1 for status, _ in results if status == "empty")
    n_old_schema = sum(1 for status, _ in results if status == "old")

    if n_old_schema:
        print(
            f"NOTE: skipped {n_old_schema} file(s) written before negative-example "
            f"logging existed (no sample_role column). Re-run the calibration batch "
            f"to include them.",
            file=sys.stderr,
        )
    if not frames:
        raise SystemExit(
            f"No usable CSVs under {csv_dir!r} (empty: {n_empty}, old schema: {n_old_schema}). "
            f"Regenerate data with `python main.py` so negative examples are logged."
        )
    return pd.concat(frames, ignore_index=True)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (feature frame, continuous names, flag names) with constant / unusable columns removed."""
    out = pd.DataFrame(index=df.index)
    for name, src in CONTINUOUS_SOURCES.items():
        vals = df[src].to_numpy(dtype=float)
        out[name] = np.log10(np.maximum(vals, 1e-15))
    # Scale-free encounter ratio: relative speed per unit separation.
    # (dt is constant across the data, so v*dt/r differs from v/r only by a
    # constant; if dt ever varies, add it here.)
    out["log_v_over_r"] = out["log_rel_velocity"] - out["log_nn_distance"]

    flag_names = []
    for col in FLAG_COLUMNS:
        if col in df.columns:
            out[col] = df[col].astype(float)
            flag_names.append(col)

    cont_names = list(CONTINUOUS_SOURCES.keys()) + ["log_v_over_r"]

    finite = np.isfinite(out[cont_names].to_numpy()).all(axis=1)
    out = out[finite]

    drop = [c for c in out.columns if out[c].std() == 0]
    if drop:
        print(f"Dropping constant feature(s): {drop}")
    out = out.drop(columns=drop)
    cont_names = [c for c in cont_names if c in out.columns]
    flag_names = [c for c in flag_names if c in out.columns]
    return out, cont_names, flag_names


# --------------------------------------------------------------------------
# Model factories
# --------------------------------------------------------------------------
def _sklearn_version() -> tuple[int, int]:
    parts = sklearn.__version__.split(".")
    return int(parts[0]), int(parts[1])


def make_ridge(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=1000)


def make_lasso(C: float) -> LogisticRegression:
    """
    L1-penalized logistic regression. scikit-learn 1.8 deprecated `penalty=`
    in favor of `l1_ratio=`; on older versions `l1_ratio` alone is silently
    ignored (you would get ridge), so pick the right spelling by version.
    """
    if _sklearn_version() >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C, max_iter=1000)
    return LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=1000)


def _assert_lasso_is_l1() -> None:
    """Guard against a version/API mismatch quietly turning the lasso into ridge."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 5))
    y = (X[:, 0] + 0.3 * rng.normal(size=500) > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = make_lasso(0.02).fit(X, y)
    if not np.any(m.coef_ == 0):
        raise SystemExit(
            "Internal check failed: the 'lasso' model produced no exactly-zero "
            "coefficients, so this scikit-learn version is not applying an L1 "
            "penalty with the arguments used here. Update make_lasso()."
        )


def make_gbt(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=8,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=seed,
    )


def _score_c(factory, X, y, tr, va, c) -> float:
    """Validation ROC-AUC for one (inner fold, C) pair."""
    sc = StandardScaler().fit(X[tr])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = factory(c).fit(sc.transform(X[tr]), y[tr])
    return float(roc_auc_score(y[va], m.decision_function(sc.transform(X[va]))))


def select_C(
    factory, X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_jobs: int = 1
) -> float:
    """
    Pick C by inner grouped CV (ROC-AUC) on the training data only.

    n_jobs=1 (default) runs serially in-process -- what the outer CV workers
    use, so they don't spawn nested pools. The final full-data fit passes
    n_jobs=args.jobs to spread the (inner fold x C) grid across all cores.
    """
    n_groups = len(np.unique(groups))
    n_splits = min(3, n_groups)
    if n_splits < 2:
        return 1.0
    splits = [
        (tr, va)
        for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups)
        if len(np.unique(y[tr])) >= 2 and len(np.unique(y[va])) >= 2
    ]
    jobs = [(c, tr, va) for tr, va in splits for c in C_GRID]
    vals = Parallel(n_jobs=n_jobs)(
        delayed(_score_c)(factory, X, y, tr, va, c) for c, tr, va in jobs
    )
    scores = {c: [] for c in C_GRID}
    for (c, _tr, _va), v in zip(jobs, vals):
        scores[c].append(v)
    valid = {c: np.mean(v) for c, v in scores.items() if v}
    return max(valid, key=valid.get) if valid else 1.0


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def weighted_auc_ap(y, score, w) -> tuple[float, float]:
    return (
        float(roc_auc_score(y, score, sample_weight=w)),
        float(average_precision_score(y, score, sample_weight=w)),
    )


def _fit_fold(kind, X, y, groups, tr, te, seed):
    """
    Fit one model on one training fold and score the held-out fold. Runs
    inside a joblib worker process; BLAS/OpenMP threads are pinned to 1 so
    that many workers in parallel do not oversubscribe the CPU.
    """
    with threadpool_limits(limits=1):
        perm = None
        if kind in ("ridge", "lasso", "logistic"):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            if kind == "logistic":
                model = LogisticRegression(C=1e3, max_iter=1000)
            else:
                factory = make_ridge if kind == "ridge" else make_lasso
                model = factory(select_C(factory, X[tr], y[tr], groups[tr]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(Xtr, y[tr])
            score = model.decision_function(Xte)
        else:  # gbt
            model = make_gbt(seed).fit(X[tr], y[tr])
            score = model.predict_proba(X[te])[:, 1]
            perm = permutation_importance(
                model, X[te], y[te], scoring="roc_auc", n_repeats=3, random_state=seed
            ).importances_mean
    return te, score, perm


def cross_validate_all(
    configs: list[tuple[str, str, list[str]]],
    F: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
    n_jobs: int,
) -> dict:
    """
    Grouped K-fold evaluation of every (model kind, feature set) in
    `configs`, with ALL (config, fold) jobs run in parallel across
    `n_jobs` worker processes. Returns, per (kind, set_name): fold metrics
    and out-of-fold scores (every row scored by a model that never saw its
    simulation).
    """
    tasks = []  # (config_index, X, tr, te)
    for ci, (_kind, _set_name, feats) in enumerate(configs):
        X = F[feats].to_numpy(dtype=float)
        for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            tasks.append((ci, X, tr, te))

    # Dispatch the slowest jobs first (gbt + permutation importance, then the
    # lasso/ridge inner-CV searches) so the run doesn't end on one straggler.
    # Results are matched back to configs via the task tuple, so order is free.
    cost = {"gbt": 0, "lasso": 1, "ridge": 2, "logistic": 3}
    tasks.sort(key=lambda t: cost[configs[t[0]][0]])

    print(f"Running {len(tasks)} model fits on {n_jobs} worker(s)...", flush=True)
    outs = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fit_fold)(configs[ci][0], X, y, groups, tr, te, seed)
        for ci, X, tr, te in tasks
    )

    results = {}
    for ci, (kind, set_name, feats) in enumerate(configs):
        oof = np.full(len(y), np.nan)
        aucs, aps, waucs, waps = [], [], [], []
        perm_sum, n_perm = np.zeros(len(feats)), 0
        for (tci, _X, _tr, _te), (te, score, perm) in zip(tasks, outs):
            if tci != ci:
                continue
            oof[te] = score
            aucs.append(roc_auc_score(y[te], score))
            aps.append(average_precision_score(y[te], score))
            wa, wp = weighted_auc_ap(y[te], score, w[te])
            waucs.append(wa)
            waps.append(wp)
            if perm is not None:
                perm_sum += perm
                n_perm += 1
        if not aucs:
            raise SystemExit(
                "Not enough simulations with both classes present to cross-validate."
            )
        results[(kind, set_name)] = {
            "auc": (np.mean(aucs), np.std(aucs)),
            "ap": (np.mean(aps), np.std(aps)),
            "w_auc": (np.mean(waucs), np.std(waucs)),
            "w_ap": (np.mean(waps), np.std(waps)),
            "oof": oof,
            "perm_importance": (perm_sum / n_perm) if n_perm else None,
        }
    return results


def fit_full_linear(kind, features, F, y, groups, n_jobs=1):
    """Fit ridge/lasso on ALL data (C chosen by grouped CV) and express it in raw feature units."""
    X = F[features].to_numpy(dtype=float)
    factory = make_ridge if kind == "ridge" else make_lasso
    C = select_C(factory, X, y, groups, n_jobs=n_jobs)
    sc = StandardScaler().fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = factory(C).fit(sc.transform(X), y)
    w_std = model.coef_.ravel()
    coef_raw = w_std / sc.scale_
    intercept_raw = float(model.intercept_[0] - np.sum(w_std * sc.mean_ / sc.scale_))
    return C, w_std, coef_raw, intercept_raw


def threshold_table(
    name: str, score: np.ndarray, y: np.ndarray, w: np.ndarray, targets=RECALL_TARGETS
) -> pd.DataFrame:
    """
    For each target recall of real corrections, find the score threshold
    that achieves it and report what it costs: the (weighted, approximate)
    fraction of ordinary steps that would be sent to IAS15 unnecessarily.
    """
    ok = np.isfinite(score)
    s, yy, ww = score[ok], y[ok], w[ok]
    pos = np.sort(s[yy == 1])
    rows = []
    for r in targets:
        # threshold such that >= r of positives have score >= threshold
        k = int(np.floor((1 - r) * len(pos)))
        thr = pos[min(k, len(pos) - 1)]
        flagged = s >= thr
        neg = yy == 0
        recall = float((flagged & (yy == 1)).sum() / (yy == 1).sum())
        fpr_raw = float((flagged & neg).sum() / neg.sum())
        fpr_w = float((ww * (flagged & neg)).sum() / ww[neg].sum())
        rows.append(
            {
                "target_recall": r,
                "threshold": thr,
                "recall": recall,
                "false_pos_rate_enriched": fpr_raw,
                "false_pos_rate_weighted": fpr_w,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Fit and compare safety-index models.")
    ap.add_argument(
        "--csv-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "csv"),
    )
    ap.add_argument("--folds", type=int, default=5, help="Grouped CV folds (default 5).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Parallel worker processes (default -1 = all cores).",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=120000,
        help="Cap on rows used for model fitting (default 120000; 0 = no cap). All "
        "correction rows are always kept; negatives are randomly subsampled and "
        "their sampling_weight scaled up so weighted metrics stay honest.",
    )
    args = ap.parse_args()

    _assert_lasso_is_l1()

    print(f"Scanning {args.csv_dir!r} ...")
    df = load_data(args.csv_dir, args.jobs)
    role = df["sample_role"].astype(str)
    df = df[role.isin(["correction"]) | role.str.startswith("negative")].copy()

    F, cont_names, flag_names = build_features(df)
    df = df.loc[F.index]
    y = (df["sample_role"].astype(str) == "correction").astype(int).to_numpy()
    w = df["sampling_weight"].fillna(1.0).to_numpy(dtype=float)
    groups = df["simulation_id"].astype(str).to_numpy()

    if args.max_rows and len(df) > args.max_rows:
        # Keep every correction; randomly thin negatives to fit the budget and
        # up-weight the survivors so weighted rates still represent the full set.
        rng = np.random.default_rng(args.seed)
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        n_neg_keep = max(args.max_rows - len(pos_idx), 1)
        if n_neg_keep < len(neg_idx):
            keep_neg = rng.choice(neg_idx, size=n_neg_keep, replace=False)
            w = w.copy()
            w[keep_neg] *= len(neg_idx) / n_neg_keep
            keep = np.sort(np.concatenate([pos_idx, keep_neg]))
            print(
                f"Subsampling {len(df)} -> {len(keep)} rows (all {len(pos_idx)} corrections "
                f"kept; negatives thinned to {n_neg_keep}). Use --max-rows 0 for everything."
            )
            F, df, y, w, groups = F.iloc[keep], df.iloc[keep], y[keep], w[keep], groups[keep]

    n_groups = len(np.unique(groups))
    print(f"\nRows: {len(df)} from {n_groups} experiment(s)")
    print(df["sample_role"].value_counts().to_string())
    if y.sum() < 20 or (1 - y).sum() < 20:
        raise SystemExit("Need at least ~20 rows of each class (correction / negative).")
    if n_groups < args.folds:
        args.folds = n_groups
        print(f"NOTE: only {n_groups} experiments; reducing folds to {args.folds}.")
    if args.folds < 2:
        raise SystemExit("Need at least 2 experiments to cross-validate by simulation.")

    all_feats = cont_names + flag_names
    rv = ["log_rel_velocity"]
    rv_flags = rv + flag_names
    # CAUSAL sets ("rv_only", "continuous") use only quantities Leapfrog can
    # compute at run time from its own state: pre_correction_* (backward-
    # difference jerk at the integrator's own dt), mass, mass_ratio.
    # Sets containing the situational flags ("rv+flags*", "all*") are OFFLINE
    # REFERENCES ONLY: calibration/events.py computes those flags with
    # central differences (np.gradient, which uses the NEXT sample) and
    # percentile thresholds taken over the whole run (which include the
    # future), so they cannot be evaluated by a real-time check and may leak
    # look-ahead information. They are kept to show how much of a score
    # comes from that, not as candidate index inputs.
    causal_sets = {"rv_only", "continuous"}
    configs = [
        ("logistic", "rv_only", rv),
        ("ridge", "continuous", cont_names),
        ("lasso", "continuous", cont_names),
        ("gbt", "rv_only", rv),
        ("gbt", "continuous", cont_names),
        ("ridge", "all*", all_feats),
        ("gbt", "rv+flags*", rv_flags),
        ("gbt", "all*", all_feats),
    ]

    print(f"\n=== Grouped {args.folds}-fold CV (grouped by simulation_id) ===")
    print("AUC / PR-AUC on the enriched sample; w_AUC / w_AP weighted by sampling_weight.\n")
    results = cross_validate_all(
        configs, F, y, w, groups, args.folds, args.seed, args.jobs
    )
    rows = []
    for kind, set_name, _feats in configs:
        res = results[(kind, set_name)]
        rows.append(
            {
                "model": kind,
                "features": set_name,
                "AUC": f"{res['auc'][0]:.3f}±{res['auc'][1]:.3f}",
                "PR-AUC": f"{res['ap'][0]:.3f}±{res['ap'][1]:.3f}",
                "w_AUC": f"{res['w_auc'][0]:.3f}±{res['w_auc'][1]:.3f}",
                "w_AP": f"{res['w_ap'][0]:.3f}±{res['w_ap'][1]:.3f}",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nHow to read this (* = uses non-causal situational flags: offline reference only):")
    print("  * continuous >> rv_only                  -> the index needs more than relative velocity.")
    print("  * gbt(continuous) <= ridge(continuous)   -> no nonlinear signal a linear index misses.")
    print("  * ridge/gbt(all*) >> continuous          -> flags help, but they look ahead; do not deploy.")

    linear_fits = [
        fit_full_linear(kind, all_feats, F, y, groups, n_jobs=args.jobs)
        for kind in ("ridge", "lasso")
    ]
    for kind, (C, w_std, coef_raw, b0) in zip(("ridge", "lasso"), linear_fits):
        print(f"\n=== {kind} fit on all data (C={C:.3g}) ===")
        table = pd.DataFrame(
            {"feature": all_feats, "std_coef": w_std, "raw_coef": coef_raw}
        )
        table = table.reindex(table["std_coef"].abs().sort_values(ascending=False).index)
        print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        if kind == "lasso":
            kept = [f for f, c in zip(all_feats, w_std) if c != 0]
            print(f"Lasso kept {len(kept)}/{len(all_feats)}: {kept}")
        terms = " ".join(
            f"{c:+.3f}*{f}" for f, c in zip(all_feats, coef_raw) if c != 0
        )
        print(f"safety_logit = {b0:+.3f} {terms}")
        print("(log_* features are log10 of the raw quantity; flags are 0/1.)")

    gbt_res = results[("gbt", "all")]
    if gbt_res["perm_importance"] is not None:
        imp = pd.Series(gbt_res["perm_importance"], index=all_feats).sort_values(ascending=False)
        print("\n=== GBT permutation importance (drop in held-out AUC when shuffled) ===")
        print(imp.head(10).to_string(float_format=lambda v: f"{v:.4f}"))

    # Threshold guidance: compare the simple rv index with the best model's
    # out-of-fold scores.
    best_key = max(results, key=lambda k: results[k]["auc"][0])
    print(f"\n=== Threshold trade-offs (out-of-fold scores; best model = {best_key}) ===")
    rv_score = F["log_rel_velocity"].to_numpy()
    print("\n-- Simple index: relative velocity (threshold shown in raw velocity units) --")
    t = threshold_table("rv", rv_score, y, w)
    t["threshold"] = 10 ** t["threshold"]
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\n-- Best model {best_key} (threshold on its score) --")
    t2 = threshold_table("best", results[best_key]["oof"], y, w)
    print(t2.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(
        "\nfalse_pos_rate_weighted ~ share of ordinary steps sent to IAS15 needlessly "
        "(approximate; see notes at top). Pick the row whose recall/cost trade-off you "
        "can live with -- that IS your safety_threshold experiment's search range."
    )


if __name__ == "__main__":
    main()
