"""
export_model_results.py

fit_safety_models.py only PRINTS its results. This script runs the same pipeline
(same data loading, features, models, grouped CV, subsampling) and SAVES what the
report figures need to outputs/analysis/:

    model_cv.csv              fold mean/std of AUC, PR-AUC, weighted AUC / AP per model     -> fig 5
    oof_scores.npz            y, w, groups + out-of-fold score of every model               -> figs 8, 9, 10
    linear_coefficients.csv   ridge and lasso, standardized + raw coefficients, C           -> fig 6
    feature_corr.csv          correlation of the continuous features (log10 space)          -> fig 4
    thresholds_ridge.csv      recall-target threshold table of the ridge out-of-fold scores  -> fig 9

    python export_model_results.py --three-body

Use the same flags you used for fit_safety_models.py (--three-body, --csv-dir, --folds,
--seed, --max-rows) so the numbers match what you saw printed.

NOTE: the `configs` list below mirrors fit_safety_models.py::main. If you change the
model list there, change it here too.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import fit_safety_models as fsm


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the safety-model comparison and save the results for plotting.")
    ap.add_argument("--csv-dir", default=None, help="Default: outputs/csv, or outputs/csv_3body with --three-body.")
    ap.add_argument("--three-body", action="store_true", help="Train only on 3-body systems (outputs/csv_3body).")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=-1, help="Parallel worker processes (default -1 = all cores).")
    ap.add_argument("--max-rows", type=int, default=120000, help="Same cap as fit_safety_models.py (0 = no cap).")
    ap.add_argument("--out-dir", default=None, help="Default: outputs/analysis")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    csv_dir = args.csv_dir or os.path.join(root, "outputs", "csv_3body" if args.three_body else "csv")
    out_dir = args.out_dir or os.path.join(root, "outputs", "analysis")
    os.makedirs(out_dir, exist_ok=True)

    fsm._assert_lasso_is_l1()
    print(f"Scanning {csv_dir!r} ...")
    df = fsm.load_data(csv_dir, args.jobs)
    role = df["sample_role"].astype(str)
    df = df[role.isin(["correction"]) | role.str.startswith("negative")].copy()
    if args.three_body:
        df = fsm.keep_body_count(df, 3).copy()

    F, cont_names, flag_names = fsm.build_features(df)
    df = df.loc[F.index]
    y = (df["sample_role"].astype(str) == "correction").astype(int).to_numpy()
    w = df["sampling_weight"].fillna(1.0).to_numpy(dtype=float)
    groups = df["simulation_id"].astype(str).to_numpy()

    if args.max_rows and len(df) > args.max_rows:  # identical to fit_safety_models.main
        rng = np.random.default_rng(args.seed)
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        n_neg_keep = max(args.max_rows - len(pos_idx), 1)
        if n_neg_keep < len(neg_idx):
            keep_neg = rng.choice(neg_idx, size=n_neg_keep, replace=False)
            w = w.copy()
            w[keep_neg] *= len(neg_idx) / n_neg_keep
            keep = np.sort(np.concatenate([pos_idx, keep_neg]))
            print(f"Subsampling {len(df)} -> {len(keep)} rows")
            F, df, y, w, groups = F.iloc[keep], df.iloc[keep], y[keep], w[keep], groups[keep]

    n_groups = len(np.unique(groups))
    folds = min(args.folds, n_groups)
    print(f"Rows: {len(df)} from {n_groups} experiment(s); {folds} folds")

    lin_feats = [f for f in cont_names if f != "log_v_over_r"]
    rv = ["log_rel_velocity"]
    causal_sets = {"rv_only", "jerk_only", "jerk+rv", "continuous"}
    configs = [  # mirrors fit_safety_models.main
        ("logistic", "rv_only", rv),
        ("logistic", "jerk_only", ["log_jerk"]),
        ("logistic", "jerk+rv", ["log_jerk", "log_rel_velocity"]),
        ("ridge", "continuous", lin_feats),
        ("lasso", "continuous", lin_feats),
        ("gbt", "rv_only", rv),
        ("gbt", "continuous", cont_names),
        ("ridge", "all*", lin_feats + flag_names),
        ("gbt", "rv+flags*", rv + flag_names),
        ("gbt", "all*", cont_names + flag_names),
    ]

    results = fsm.cross_validate_all(configs, F, y, w, groups, folds, args.seed, args.jobs)

    rows, arrays = [], {"y": y, "w": w, "groups": groups.astype(str)}
    for kind, set_name, _feats in configs:
        r = results[(kind, set_name)]
        rows.append(
            {
                "model": kind,
                "features": set_name,
                "causal": set_name in causal_sets,
                "auc_mean": r["auc"][0], "auc_std": r["auc"][1],
                "ap_mean": r["ap"][0], "ap_std": r["ap"][1],
                "w_auc_mean": r["w_auc"][0], "w_auc_std": r["w_auc"][1],
                "w_ap_mean": r["w_ap"][0], "w_ap_std": r["w_ap"][1],
            }
        )
        key = f"oof__{kind}__{set_name.replace('*', '_offline').replace('+', '_plus_')}"
        arrays[key] = r["oof"]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "model_cv.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "oof_scores.npz"), **arrays)

    coef_rows = []
    for kind in ("ridge", "lasso"):
        C, w_std, coef_raw, b0 = fsm.fit_full_linear(kind, lin_feats, F, y, groups, n_jobs=args.jobs)
        for f, s, r in zip(lin_feats, w_std, coef_raw):
            coef_rows.append({"model": kind, "feature": f, "std_coef": s, "raw_coef": r, "C": C, "intercept_raw": b0})
    pd.DataFrame(coef_rows).to_csv(os.path.join(out_dir, "linear_coefficients.csv"), index=False)

    F[cont_names].corr().to_csv(os.path.join(out_dir, "feature_corr.csv"))

    tk = fsm.threshold_table("ridge", results[("ridge", "continuous")]["oof"], y, w)
    tk.insert(2, "safety_threshold_0_1", 1.0 / (1.0 + np.exp(-np.clip(tk["threshold"], -700, 700))))
    tk.to_csv(os.path.join(out_dir, "thresholds_ridge.csv"), index=False)

    print(f"\nSaved to {out_dir}:")
    for name in ("model_cv.csv", "oof_scores.npz", "linear_coefficients.csv", "feature_corr.csv", "thresholds_ridge.csv"):
        print(f"  {name}")
    print("\nThen: python plot_report_figures.py --only 5 9")


if __name__ == "__main__":
    main()
