"""
fit_situation_weights.py

How does each SITUATION (the boolean flags calibration/events.py logs per body)
affect whether a body needed a correction, and what weight should each one
carry in an ADDITIVE safety index?

    python fit_situation_weights.py
    python fit_situation_weights.py --csv-dir outputs/csv --folds 5
    python fit_situation_weights.py --unweighted

Reads the per-body CSVs that main.py already produced (same loader and same
label as fit_safety_models.py: sample_role == "correction" -> 1, any
"negative_*" role -> 0) and writes:

    outputs/situation_weights.json   intercept + one weight per situation
    outputs/situation_weights.csv    the per-situation statistics table

--------------------------------------------------------------------------
The index this produces
--------------------------------------------------------------------------
    safety_logit = intercept + sum( weight[s] for every situation s that is active )
    safety_score = 1 / (1 + exp(-safety_logit))          # 0-1, higher = less safe

A weight is the change in log-odds of "this body needs a correction" when that
situation is active, holding the other situations fixed (positive = raises
risk, negative = lowers it, e.g. quiescent_regime). The weights come from one
L2-regularized logistic regression on all situations at once, so overlapping
situations (close_encounter / near_collision / rapid_approach ...) share credit
instead of each claiming all of it. The regularization strength is chosen by
inner cross-validation grouped by simulation_id. The univariate log-odds
weight of each situation on its own is reported alongside for comparison.

By default rows are weighted by `sampling_weight`, which undoes the deliberate
over-representation of hard negatives, so the intercept and the 0-1 score are
closer to per-step reality. --unweighted fits the raw enriched sample instead.
`stable` is excluded (exactly the complement of the anomaly flags, so it would
be perfectly collinear). A situation that is never (or always) active in the
data carries no information and gets weight 0.

--------------------------------------------------------------------------
Caveat: these flags are OFFLINE quantities
--------------------------------------------------------------------------
calibration/events.py computes them with central differences (np.gradient uses
the NEXT sample) and with percentile thresholds taken over the whole run, and at
a correction row they describe the already-snapped state. A real-time integrator
cannot evaluate them as logged, and scores obtained from them are optimistic
(see the `*` reference models in fit_safety_models.py). Use these weights to
rank/understand situations; to drive the adaptive integrator, each situation
first needs a causal, run-time implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
except ImportError:  # pragma: no cover
    raise SystemExit("This script needs scikit-learn. Install it with:\n    pip install scikit-learn")

from fit_safety_models import FLAG_COLUMNS, load_data, threshold_table

ROOT = os.path.dirname(os.path.abspath(__file__))
C_GRID = np.logspace(-3, 2, 6)
# Situations that describe calm rather than risk; excluded from the naive
# "count of active anomaly flags" baseline.
CALM_FLAGS = ("quiescent_regime",)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700.0, 700.0)))


# --------------------------------------------------------------------------
# Descriptive statistics per situation
# --------------------------------------------------------------------------
def situation_stats(X: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    """
    One row per situation: how often it is active, and how the correction
    rate differs when it is on vs off. Probabilities and the univariate
    log-odds weight use the sampling weights `w` (approx. per real sampled
    step); counts are raw row counts. A 0.5 pseudo-count keeps rare
    situations finite.
    """
    pos = y == 1
    rows = []
    for name in X.columns:
        on = X[name].to_numpy() > 0
        a, b = w[on & pos].sum(), w[on & ~pos].sum()
        c, d = w[~on & pos].sum(), w[~on & ~pos].sum()
        p_on = a / (a + b) if a + b > 0 else np.nan
        p_off = c / (c + d) if c + d > 0 else np.nan
        rows.append(
            {
                "situation": name,
                "n_active": int(on.sum()),
                "n_active_corrections": int((on & pos).sum()),
                "n_active_negatives": int((on & ~pos).sum()),
                "pct_of_corrections_with_it": 100.0 * (on & pos).sum() / max(pos.sum(), 1),
                "pct_of_negatives_with_it": 100.0 * (on & ~pos).sum() / max((~pos).sum(), 1),
                "p_correction_if_on": p_on,
                "p_correction_if_off": p_off,
                "lift": p_on / p_off if p_off and p_off > 0 else np.nan,
                "univariate_weight": float(np.log((a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5)))),
            }
        )
    return pd.DataFrame(rows).set_index("situation")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def fit_model(X, y, w, C) -> LogisticRegression:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LogisticRegression(C=C, max_iter=2000).fit(X, y, sample_weight=w)


def select_C(X, y, w, groups) -> float:
    """Pick C by inner grouped CV (ROC-AUC) on the given rows only."""
    n_splits = min(3, len(np.unique(groups)))
    if n_splits < 2:
        return 1.0
    splits = [
        (tr, va)
        for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups)
        if len(np.unique(y[tr])) >= 2 and len(np.unique(y[va])) >= 2
    ]
    if not splits:
        return 1.0
    best_c, best_auc = 1.0, -1.0
    for c in C_GRID:  # ties keep the smaller C (stronger regularization)
        auc = np.mean(
            [
                roc_auc_score(y[va], fit_model(X[tr], y[tr], w[tr], c).decision_function(X[va]))
                for tr, va in splits
            ]
        )
        if auc > best_auc:
            best_c, best_auc = float(c), float(auc)
    return best_c


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit additive per-situation safety-index weights.")
    ap.add_argument("--csv-dir", default=os.path.join(ROOT, "outputs", "csv"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--folds", type=int, default=5, help="Grouped CV folds (default 5).")
    ap.add_argument("--jobs", type=int, default=-1, help="Workers for CSV loading (default: all cores).")
    ap.add_argument(
        "--unweighted",
        action="store_true",
        help="Fit on the raw enriched sample instead of weighting rows by sampling_weight.",
    )
    ap.add_argument(
        "--min-active",
        type=int,
        default=30,
        help="Flag situations active in fewer than this many rows as rare (default 30).",
    )
    args = ap.parse_args()

    print(f"Scanning {args.csv_dir!r} ...")
    df = load_data(args.csv_dir, args.jobs)
    role = df["sample_role"].astype(str)
    df = df[(role == "correction") | role.str.startswith("negative")].reset_index(drop=True)

    y = (df["sample_role"].astype(str) == "correction").astype(int).to_numpy()
    w_raw = df["sampling_weight"].fillna(1.0).to_numpy(dtype=float)
    groups = df["simulation_id"].astype(str).to_numpy()
    w_fit = np.ones_like(w_raw) if args.unweighted else w_raw / w_raw.mean()

    flags = [c for c in FLAG_COLUMNS if c in df.columns]
    if not flags:
        raise SystemExit("None of the situation-flag columns were found in the CSVs.")
    X_all = df[flags].fillna(0).astype(float)

    n_groups = len(np.unique(groups))
    print(f"\nRows: {len(df)} from {n_groups} experiment(s); corrections: {int(y.sum())}, negatives: {int((1 - y).sum())}")
    print(f"Fit weighting: {'none (raw enriched sample)' if args.unweighted else 'sampling_weight'}")
    if y.sum() < 20 or (1 - y).sum() < 20:
        raise SystemExit("Need at least ~20 rows of each class (correction / negative).")
    folds = min(args.folds, n_groups)
    if folds < 2:
        raise SystemExit("Need at least 2 experiments to cross-validate by simulation.")

    stats = situation_stats(X_all, y, w_raw)

    varying = [f for f in flags if X_all[f].std() > 0]
    constant = [f for f in flags if f not in varying]
    if constant:
        print(f"No information (never or always active), weight fixed at 0: {constant}")
    X = X_all[varying].to_numpy()

    # ---- grouped cross-validation: out-of-fold scores + weight stability ----
    oof = np.full(len(y), np.nan)
    fold_coefs, aucs, aps, waucs, waps = [], [], [], [], []
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        model = fit_model(X[tr], y[tr], w_fit[tr], select_C(X[tr], y[tr], w_fit[tr], groups[tr]))
        s = model.decision_function(X[te])
        oof[te] = s
        fold_coefs.append(model.coef_.ravel())
        aucs.append(roc_auc_score(y[te], s))
        aps.append(average_precision_score(y[te], s))
        waucs.append(roc_auc_score(y[te], s, sample_weight=w_raw[te]))
        waps.append(average_precision_score(y[te], s, sample_weight=w_raw[te]))
    if not aucs:
        raise SystemExit("Not enough simulations with both classes present to cross-validate.")

    # naive baseline: every anomaly situation counts 1, no weights
    anomaly = [f for f in varying if f not in CALM_FLAGS]
    count_score = X_all[anomaly].sum(axis=1).to_numpy()

    print(f"\n=== Grouped {len(aucs)}-fold CV (grouped by simulation_id) ===")
    print(f"weighted situations:  AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
          f"PR-AUC {np.mean(aps):.3f}±{np.std(aps):.3f}  "
          f"w_AUC {np.mean(waucs):.3f}±{np.std(waucs):.3f}  w_AP {np.mean(waps):.3f}±{np.std(waps):.3f}")
    print(f"equal weights (count of active anomaly situations):  AUC {roc_auc_score(y, count_score):.3f}  "
          f"w_AUC {roc_auc_score(y, count_score, sample_weight=w_raw):.3f}")

    # ---- final fit on all data ----
    C = select_C(X, y, w_fit, groups)
    final = fit_model(X, y, w_fit, C)
    weights = {f: 0.0 for f in flags}
    weights.update(dict(zip(varying, final.coef_.ravel().tolist())))
    intercept = float(final.intercept_[0])
    fold_std = dict(zip(varying, np.std(np.array(fold_coefs), axis=0).tolist()))
    sign_agree = dict(
        zip(varying, np.mean(np.sign(np.array(fold_coefs)) == np.sign(final.coef_.ravel()), axis=0).tolist())
    )

    stats["weight"] = pd.Series(weights)
    stats["weight_fold_std"] = pd.Series(fold_std).reindex(stats.index).fillna(0.0)
    stats["fold_sign_agreement"] = pd.Series(sign_agree).reindex(stats.index)
    stats["rare"] = stats["n_active"] < args.min_active
    stats = stats.reindex(stats["weight"].abs().sort_values(ascending=False).index)

    print(f"\n=== Situation weights (ridge logistic, C={C:g}; log-odds change when the situation is active) ===")
    show = stats[
        [
            "weight", "weight_fold_std", "univariate_weight", "n_active",
            "pct_of_corrections_with_it", "pct_of_negatives_with_it", "lift", "rare",
        ]
    ].round(3)
    print(show.to_string())
    print("\nweight_fold_std: how much the weight moves between CV folds (large vs |weight| = unstable).")
    print("rare: fewer than --min-active rows; the weight is mostly regularization, do not trust it.")
    print("Overlapping situations (e.g. close_encounter / near_collision) trade weight, so read the group, not one row.")

    print(f"\n=== Paste-ready additive index ===")
    print(f"SITUATION_INTERCEPT = {intercept:+.4f}")
    print("SITUATION_WEIGHTS = {")
    for name in stats.index:
        print(f'    "{name}": {weights[name]:+.4f},')
    print("}")
    print("safety_logit = SITUATION_INTERCEPT + sum(w for s, w in SITUATION_WEIGHTS.items() if flags[s])")
    print("safety_score = 1 / (1 + exp(-safety_logit))   # 0-1, feeds --safety-threshold")

    # ---- thresholds on out-of-fold scores ----
    thr = threshold_table("situations", oof, y, w_raw)
    thr.insert(2, "safety_threshold_0_1", sigmoid(thr["threshold"]))
    print("\n=== Threshold trade-offs (out-of-fold safety_logit; safety_threshold_0_1 = sigmoid of it) ===")
    print(thr.rename(columns={"threshold": "safety_logit_threshold"}).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nfalse_pos_rate_weighted ~ share of ordinary steps sent to IAS15 needlessly (approximate).")
    print("Situation flags are discrete, so many rows tie on score and recall targets land in steps.")

    # ---- write outputs ----
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "situation_weights.csv")
    json_path = os.path.join(args.out_dir, "situation_weights.json")
    stats.reset_index().to_csv(csv_path, index=False)
    payload = {
        "model": "L2 logistic regression on binary situation flags",
        "score": "safety_logit = intercept + sum(weights[s] for active s); safety_score = sigmoid(safety_logit)",
        "intercept": intercept,
        "weights": {k: weights[k] for k in stats.index},
        "C": C,
        "fit_weighting": "none" if args.unweighted else "sampling_weight",
        "n_rows": int(len(df)),
        "n_experiments": int(n_groups),
        "cv": {
            "folds": len(aucs),
            "auc": float(np.mean(aucs)),
            "weighted_auc": float(np.mean(waucs)),
            "equal_weight_count_auc": float(roc_auc_score(y, count_score)),
        },
        "thresholds": thr.to_dict("records"),
        "caveat": "Flags are computed offline (central differences, whole-run percentiles, post-snap state at "
                  "corrections). A real-time integrator needs causal implementations of each situation.",
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nWrote {json_path}\nWrote {csv_path}")


if __name__ == "__main__":
    main()
