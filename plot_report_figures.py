"""
plot_report_figures.py

Report figures for the IRIS adaptive-integrator work. This file implements the
MUST-HAVE set, numbered as in the figure plan:

All 39 figures are implemented. Needs extra data first:
  4, 5, 6, 8, 9, 10   python export_model_results.py --three-body
  3                   python export_feature_correlations.py (see fig03)
  34, 35, 36, 32      a summary with --chaos-control columns (the --summary-run folder)
  38, 39              python run_sweeps.py (writes outputs/sweeps/)

     5  Model comparison: grouped-CV AUC and weighted AUC with fold-std error bars   [needs export_model_results.py]
     9  Recall vs false-positive-rate trade-off, 0.80/0.90/0.95/0.99 recall marked    [needs export_model_results.py]
    14  Three-panel time series of one experiment (error / safety score / IAS15 share)
    20  Rollback depth vs deviation from the exact state (per rewind)
    24  Ablation: adaptive errors per run + inferred configuration table
    25  Accuracy-cost Pareto: errors vs mean IAS15 step share, one point per run
    26  Resync quality: true error before vs after (log-log, y = x and y = x/1000)

    python plot_report_figures.py                 # all must-haves (5 and 9 are skipped if their data is missing)
    python plot_report_figures.py --only 24 25    # a subset
    python plot_report_figures.py --pdf           # also write vector PDFs

Data layout it expects (what your outputs/ folder already looks like):

    outputs/validation_A ... validation_G/   one folder per ablation run (validation_summary.csv + csv/)
    outputs/validation/                      the latest run (same layout)
    outputs/analysis/                        written by export_model_results.py (figures 5 and 9)

Every per-run summary and per-simulation file is filtered to --seed-min..--seed-max
(default 956..987: the 32 experiments the ablation runs share), so figures 24/25 compare
like with like even though the csv folders also hold other seed families.

Figures are written to outputs/report_figures/.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(ROOT, "outputs")
CORRECTION_THRESHOLD = 0.01
RUN_LETTERS = list("ABCDEFG")
RECALL_TARGETS = (0.80, 0.90, 0.95, 0.99)

CAT_COLORS = {"adequate": "#2a9d8f", "improvable": "#e9a03b", "drift_dominated": "#c8553d"}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "legend.frameon": False,
    }
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def seed_of(path: str) -> int:
    m = re.search(r"_seed(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else -1


def sim_files(csv_dir: str, suffix: str, smin: int, smax: int) -> list[str]:
    files = glob.glob(os.path.join(csv_dir, f"*_{suffix}.csv"))
    return sorted(f for f in files if smin <= seed_of(f) <= smax and _has_header(f))


def _has_header(path: str) -> bool:
    """False for empty / whitespace-only files that pandas cannot parse."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return "," in fh.readline()
    except OSError:
        return False


def col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df else pd.Series(np.nan, index=df.index)


def save(fig, name: str, args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, name)
    fig.savefig(path + ".png", bbox_inches="tight")
    if args.pdf:
        fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}.png")


def discover_runs(outputs_dir: str) -> dict[str, str]:
    runs = {}
    for letter in RUN_LETTERS:
        d = os.path.join(outputs_dir, f"validation_{letter}")
        if os.path.isfile(os.path.join(d, "validation_summary.csv")):
            runs[letter] = d
    d = os.path.join(outputs_dir, "validation")
    if os.path.isfile(os.path.join(d, "validation_summary.csv")):
        runs["latest"] = d
    return runs


def load_summary(run_dir: str, smin: int, smax: int) -> pd.DataFrame:
    s = pd.read_csv(os.path.join(run_dir, "validation_summary.csv"))
    if "seed" not in s:
        s["seed"] = s["simulation_id"].str.extract(r"_seed(\d+)")[0].astype(int)
    return s[(s["seed"] >= smin) & (s["seed"] <= smax)].reset_index(drop=True)


def csv_dir_of(run_dir: str) -> str:
    return os.path.join(run_dir, "csv")


def pick_run(runs: dict[str, str], wanted: str) -> tuple[str, str]:
    if wanted in runs:
        return wanted, runs[wanted]
    fallback = list(runs)[-1]
    print(f"  run {wanted!r} not found, using {fallback!r}")
    return fallback, runs[fallback]


def need_analysis(args, *names: str) -> bool:
    missing = [n for n in names if not os.path.isfile(os.path.join(args.analysis_dir, n))]
    if missing:
        print(f"  SKIPPED: {missing} not found in {args.analysis_dir}. Run:\n"
              f"      python export_model_results.py --three-body")
        return False
    return True


# --------------------------------------------------------------------------
# Figure 5: model comparison
# --------------------------------------------------------------------------
def fig05(args, runs) -> None:
    if not need_analysis(args, "model_cv.csv"):
        return
    cv = pd.read_csv(os.path.join(args.analysis_dir, "model_cv.csv"))
    labels = [f"{m}\n{f}" for m, f in zip(cv["model"], cv["features"])]
    offline = cv["features"].str.endswith("*").to_numpy()
    x = np.arange(len(cv))
    bw = 0.38

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    for off, key, color, name in (
        (-bw / 2, "auc", "#3b6ea5", "grouped-CV AUC"),
        (+bw / 2, "w_auc", "#e07a3f", "weighted AUC (sampling_weight)"),
    ):
        mean, std = cv[f"{key}_mean"].to_numpy(), cv[f"{key}_std"].to_numpy()
        bars = ax.bar(x + off, mean, bw, yerr=std, capsize=3, color=color, label=name,
                      error_kw={"elinewidth": 1.1, "ecolor": "#222"})
        for b, is_off in zip(bars, offline):
            if is_off:
                b.set_hatch("//")
                b.set_alpha(0.55)
        for xi, m in zip(x + off, mean):
            ax.text(xi, 0.505, f"{m:.3f}", ha="center", va="bottom", rotation=90, fontsize=8, color="white")

    chosen = np.flatnonzero((cv["model"] == "ridge") & (cv["features"] == "continuous"))
    for i in chosen:
        ax.annotate("chosen", (x[i], max(cv.loc[i, "auc_mean"], cv.loc[i, "w_auc_mean"]) + 0.03),
                    ha="center", fontsize=9, fontweight="bold",
                    arrowprops={"arrowstyle": "-", "color": "#444"}, xytext=(x[i], 0.985))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("ROC-AUC (mean over folds, bar = ±1 std)")
    ax.set_title("Safety-index model comparison (CV grouped by simulation)")
    handles, lab = ax.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="#999", hatch="//", alpha=0.55))
    lab.append("* offline only (non-causal situational flags)")
    ax.legend(handles, lab, loc="lower right")
    save(fig, "fig05_model_comparison", args)


# --------------------------------------------------------------------------
# Figure 9: recall vs false-positive rate
# --------------------------------------------------------------------------
def recall_fpr_curve(score, y, w):
    ok = np.isfinite(score)
    s, yy, ww = score[ok], y[ok], w[ok]
    order = np.argsort(-s, kind="stable")
    ys, ws = yy[order], ww[order]
    recall = np.cumsum(ys) / ys.sum()
    fpr = np.cumsum(ws * (1 - ys)) / (ws * (1 - yy)).sum()
    return fpr, recall


def point_at_threshold(score, y, w, thr):
    """(recall, weighted FPR) when everything with score >= thr is flagged."""
    ok = np.isfinite(score)
    s, yy, ww = score[ok], y[ok], w[ok]
    flagged = s >= thr
    recall = (flagged & (yy == 1)).sum() / (yy == 1).sum()
    fpr = (ww * (flagged & (yy == 0))).sum() / ww[yy == 0].sum()
    return float(recall), float(fpr)


def threshold_for_recall(score, y, r):
    """Same rule as fit_safety_models.threshold_table."""
    pos = np.sort(score[(y == 1) & np.isfinite(score)])
    k = int(np.floor((1 - r) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def oof_key(kind: str, feats: str) -> str:
    return f"oof__{kind}__{feats.replace('*', '_offline').replace('+', '_plus_')}"


def fig09(args, runs) -> None:
    if not need_analysis(args, "oof_scores.npz"):
        return
    z = np.load(os.path.join(args.analysis_dir, "oof_scores.npz"))
    y, w = z["y"].astype(int), z["w"].astype(float)

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    others = [("logistic", "rv_only", "relative velocity only", "#888", ":"),
              ("logistic", "jerk+rv", "jerk + relative velocity", "#8e6bbf", "--"),
              ("lasso", "continuous", "lasso", "#2a9d8f", "-.")]
    for kind, feats, label, color, ls in others:
        k = oof_key(kind, feats)
        if k in z:
            fpr, rec = recall_fpr_curve(z[k], y, w)
            ax.plot(fpr, rec, color=color, ls=ls, lw=1.4, label=label)

    main_key = oof_key("ridge", "continuous")
    score = z[main_key]
    fpr, rec = recall_fpr_curve(score, y, w)
    ax.plot(fpr, rec, color="#1d4e89", lw=2.6, label="ridge (continuous, causal)")

    offsets = {0.80: (14, -22), 0.90: (14, -24), 0.95: (14, -26), 0.99: (-96, -34)}
    for r in RECALL_TARGETS:
        thr = threshold_for_recall(score, y, r)
        rr, ff = point_at_threshold(score, y, w, thr)
        ax.scatter([ff], [rr], s=70, color="#d1495b", zorder=5, edgecolor="white")
        sig = 1.0 / (1.0 + np.exp(-np.clip(thr, -700, 700)))
        ax.annotate(f"recall {r:.2f}\ncost {100 * ff:.0f}%  (score ≥ {sig:.2g})", (ff, rr),
                    xytext=offsets[r], textcoords="offset points", fontsize=8.8,
                    arrowprops={"arrowstyle": "-", "color": "#999", "lw": 0.8})

    if args.safety_threshold is not None:
        thr = float(np.log(args.safety_threshold / (1 - args.safety_threshold)))
        rr, ff = point_at_threshold(score, y, w, thr)
        ax.scatter([ff], [rr], marker="*", s=230, color="#f2b705", edgecolor="#333", zorder=6,
                   label=f"deployed threshold {args.safety_threshold:g}: recall {rr:.2f}, cost {100 * ff:.0f}%")

    ax.set_xlabel("False-positive rate (share of ordinary steps sent to IAS15, sampling-weighted)")
    ax.set_ylabel("Recall of real corrections")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1.01)
    ax.set_title("Recall vs. cost trade-off (out-of-fold ridge scores)")
    ax.legend(loc="lower right", fontsize=8.8)
    save(fig, "fig09_recall_vs_fpr", args)


# --------------------------------------------------------------------------
# Figure 14: three-panel time series
# --------------------------------------------------------------------------
def fig14(args, runs) -> None:
    name, run_dir = pick_run(runs, args.timeseries_run)
    s = load_summary(run_dir, args.seed_min, args.seed_max)
    if args.sim:
        sim = args.sim
    else:
        cand = s[(col(s, "baseline_n_errors") >= 2) & (col(s, "adaptive_n_errors") == 0)]
        cand = cand[(cand["ias15_step_fraction"] > 0.05) & (cand["ias15_step_fraction"] < 0.7)]
        if cand.empty:
            cand = s[col(s, "baseline_n_errors") >= 1]
        sim = cand.loc[cand["n_switches"].idxmax(), "simulation_id"]
        print(f"  auto-selected {sim} from run {name} (baseline failed >=2x, adaptive 0 errors, most switches); "
              f"override with --sim")
    path = os.path.join(csv_dir_of(run_dir), f"{sim}_validation.csv")
    if not os.path.isfile(path):
        print(f"  SKIPPED: {path} not found")
        return
    df = pd.read_csv(path)
    t = df["time"].to_numpy()
    base = df.get("baseline_pre_check_error", df["baseline_position_error"]).clip(lower=1e-15)
    adap = df.get("adaptive_pre_check_error", df["adaptive_position_error"]).clip(lower=1e-15)
    on_ias15 = (df["active_integrator"].astype(str).str.lower() == "ias15").to_numpy()

    fig, ax = plt.subplots(3, 1, figsize=(10, 8.6), sharex=True,
                           gridspec_kw={"height_ratios": [1.5, 1.0, 0.8], "hspace": 0.12})
    ax[0].plot(t, base, color="#c8553d", lw=1.2, label="Leapfrog only")
    ax[0].plot(t, adap, color="#1d4e89", lw=1.4, label="Adaptive")
    ax[0].axhline(CORRECTION_THRESHOLD, color="k", ls="--", lw=1, label=f"error threshold {CORRECTION_THRESHOLD:g}")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("RMS position error vs IAS15")
    ax[0].set_title(f"{sim}  (run: {name})")
    ax[0].legend(ncol=3, loc="lower right", fontsize=9)

    score = df["safety_score"].clip(lower=1e-6)
    ax[1].plot(t, score, color="#444", lw=1.1)
    ax[1].axhline(args.safety_threshold, color="#d1495b", ls="--", lw=1.2,
                  label=f"safety threshold {args.safety_threshold:g}")
    ax[1].axhline(args.safety_threshold * args.release_fraction, color="#d1495b", ls=":", lw=1.0,
                  label=f"release {args.safety_threshold * args.release_fraction:.3g}")
    ax[1].fill_between(t, 0, 1, where=on_ias15, step="mid", color="#1d4e89", alpha=0.13,
                       transform=ax[1].get_xaxis_transform(), label="IAS15 active")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Safety score")
    ax[1].legend(ncol=4, loc="lower right", fontsize=8.5)

    ax[2].fill_between(t, df["ias15_step_fraction"], step="mid", color="#1d4e89", alpha=0.55)
    ax[2].set_ylim(0, 1.02)
    ax[2].set_ylabel("IAS15 step share")
    ax[2].set_xlabel("Time")
    save(fig, "fig14_three_panel_timeseries", args)


# --------------------------------------------------------------------------
# Figure 20: rollback depth vs deviation
# --------------------------------------------------------------------------
def fig20(args, runs) -> None:
    name, run_dir = pick_run(runs, args.rewinds_run)
    files = sim_files(csv_dir_of(run_dir), "rewinds", args.seed_min, args.seed_max)
    frames = [pd.read_csv(f) for f in files if os.path.getsize(f) > 0]
    frames = [f for f in frames if len(f)]
    if not frames:
        print(f"  SKIPPED: no *_rewinds.csv in {csv_dir_of(run_dir)} for seeds {args.seed_min}..{args.seed_max}")
        return
    R = pd.concat(frames, ignore_index=True)
    dcols = sorted([c for c in R.columns if re.fullmatch(r"dev_depth\d+", c)], key=lambda c: int(c[9:]))
    V = R[dcols].to_numpy(float)
    V[V <= 0] = np.nan
    depths = np.arange(len(dcols))
    keep = np.isfinite(V).sum(axis=1) >= 2
    R, V = R[keep].reset_index(drop=True), V[keep]
    cat = R["category"].astype(str).to_numpy()

    vmax, vmin = np.nanmax(V, axis=1), np.nanmin(V, axis=1)
    flat_high = (vmax / vmin < args.flat_ratio) & (vmin > args.clean_threshold)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})
    for i in range(len(V)):
        c = CAT_COLORS.get(cat[i], "#777")
        if flat_high[i]:
            ax.plot(depths, V[i], color=c, lw=1.1, alpha=0.7, zorder=3)
        else:
            ax.plot(depths, V[i], color=c, lw=0.6, alpha=0.12, zorder=1)
    for k, c in CAT_COLORS.items():
        sel = cat == k
        if sel.any():
            ax.plot(depths, np.nanmedian(V[sel], axis=0), color=c, lw=3.2, zorder=5,
                    label=f"{k.replace('_', '-')} (n={sel.sum()}, median)")
    ax.axhline(args.clean_threshold, color="k", ls="--", lw=1)
    ax.text(depths[0], args.clean_threshold * 1.25, "adequate = deviation ≤ clean threshold",
            ha="left", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xticks(depths)
    ax.set_xlabel("Checkpoint depth (0 = latest checkpoint, higher = deeper rollback)")
    ax.set_ylabel("RMS deviation from exact state at the trigger step")
    ax.set_title(f"Per-rewind deviation vs rollback depth (run {name}, n={len(V)} rewinds)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93), fontsize=8.5)
    ax.text(0.99, 0.97, f"highlighted lines: flat and above the clean threshold ({flat_high.sum()} = {100 * flat_high.mean():.0f}% of rewinds)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5)

    rel = V / V[:, [0]]
    for k, c in CAT_COLORS.items():
        sel = cat == k
        if sel.sum() < 3:
            continue
        q25, med, q75 = (np.nanpercentile(rel[sel], q, axis=0) for q in (25, 50, 75))
        ax2.plot(depths, med, color=c, lw=2.4, label=k.replace("_", "-"))
        ax2.fill_between(depths, q25, q75, color=c, alpha=0.18)
    ax2.axhline(1.0, color="k", lw=1)
    ax2.set_yscale("log")
    ax2.set_xticks(depths)
    ax2.set_xlabel("Checkpoint depth")
    ax2.set_ylabel("Deviation relative to depth 0  (median, IQR)")
    ax2.set_title("Does a deeper rollback help?  (below 1 = yes)")
    ax2.legend(fontsize=8.5)
    save(fig, "fig20_rollback_depth", args)
    print(f"  rewinds by category: {R['category'].value_counts().to_dict()}")


# --------------------------------------------------------------------------
# Figures 24 / 25: ablation + Pareto (share a per-run table)
# --------------------------------------------------------------------------
def build_run_table(runs, args) -> pd.DataFrame:
    rows = []
    for name, d in runs.items():
        s = load_summary(d, args.seed_min, args.seed_max)
        if s.empty:
            print(f"  run {name}: no summary rows in seed range, skipped")
            continue
        files = sim_files(csv_dir_of(d), "rewinds", args.seed_min, args.seed_max)
        ckpts = np.nan
        if files:
            hdr = pd.read_csv(files[0], nrows=0).columns
            n = [int(c[9:]) for c in hdr if re.fullmatch(r"dev_depth\d+", c)]
            ckpts = (max(n) + 1) if n else np.nan
        n_sw = col(s, "n_switch_resyncs").fillna(0).sum()
        n_rs = col(s, "n_resyncs").fillna(0).sum()
        rows.append(
            {
                "run": name,
                "n_experiments": len(s),
                "errors": int(s["adaptive_n_errors"].sum()),
                "baseline_errors": int(s["baseline_n_errors"].sum()),
                "ias15_share_pct": 100 * float(s["ias15_step_fraction"].mean()),
                "rewinds": int(col(s, "n_rewinds").fillna(0).sum()),
                "resyncs": int(n_rs),
                "correct": bool(col(s, "correction_enabled").fillna(False).astype(bool).any()),
                "rewind": bool(col(s, "n_rewinds").fillna(0).sum() > 0),
                "resync_on_switch": bool(n_sw > 0),
                "drift_budget": bool(n_rs - n_sw > 0),
                "checkpoints": ckpts,
                "fp_detection": bool(col(s, "n_stretches").notna().any()),
                "rollback_analysis": bool(col(s, "rewinds_adequate").notna().any()),
                "chaos_control": bool(col(s, "cpu_control_s").notna().any()),
            }
        )
    return pd.DataFrame(rows)


def fig24(args, runs) -> None:
    T = build_run_table(runs, args)
    if T.empty:
        print("  SKIPPED: no run summaries found")
        return
    os.makedirs(args.out_dir, exist_ok=True)
    T.to_csv(os.path.join(args.out_dir, "run_table.csv"), index=False)
    if T["n_experiments"].nunique() > 1:
        print(f"  WARNING: runs have different experiment counts: {T.set_index('run')['n_experiments'].to_dict()}")
    base = int(T["baseline_errors"].iloc[-1])
    names = ["baseline"] + T["run"].tolist()
    vals = [base] + T["errors"].tolist()
    colors = ["#c8553d"] + ["#1d4e89" if r != "latest" else "#2a9d8f" for r in T["run"]]

    mark = lambda b: "✓" if b else "–"
    row_labels = ["IAS15 share", "rewinds", "resyncs", "rewind", "resync on switch", "drift budget", "checkpoints",
                  "rollback analysis"]
    cells = [["0%"] + [f"{v:.1f}%" for v in T["ias15_share_pct"]],
             ["–"] + [f"{v:,}" for v in T["rewinds"]],
             ["–"] + [f"{v:,}" for v in T["resyncs"]],
             ["–"] + [mark(v) for v in T["rewind"]],
             ["–"] + [mark(v) for v in T["resync_on_switch"]],
             ["–"] + [mark(v) for v in T["drift_budget"]],
             ["–"] + ["?" if np.isnan(v) else f"{int(v)}" for v in T["checkpoints"]],
             ["–"] + [mark(v) for v in T["rollback_analysis"]]]

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    x = np.arange(len(names))
    bars = ax.bar(x, vals, color=colors, width=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01, str(v), ha="center", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel(f"Samples with RMS error > {CORRECTION_THRESHOLD:g}  (summed over {int(T['n_experiments'].iloc[0])} experiments)")
    ax.set_title("Ablation: errors per run, with the configuration inferred from each run's outputs")
    ax.tick_params(axis="x", length=0)
    tab = ax.table(cellText=cells, rowLabels=row_labels, cellLoc="center", loc="bottom",
                   bbox=[0, -0.62, 1, 0.52])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#ccc")
    ax.text(0, -0.67, "Configuration is inferred from which columns/files each run produced (checkpoint count from "
            "rewinds.csv); Richardson levels, Kahan sums and dt/threshold values cannot be inferred.",
            transform=ax.transAxes, fontsize=7.5, color="#555", va="top")
    save(fig, "fig24_ablation", args)
    print(T[["run", "errors", "ias15_share_pct", "rewinds", "resyncs", "n_experiments"]].round(1).to_string(index=False))


def fig25(args, runs) -> None:
    T = build_run_table(runs, args)
    if T.empty:
        print("  SKIPPED: no run summaries found")
        return
    base_err = int(T["baseline_errors"].iloc[-1])
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    pts = T[["ias15_share_pct", "errors"]].to_numpy()
    order = np.argsort(pts[:, 0])
    best, front = np.inf, []
    for i in order:
        if pts[i, 1] < best:
            front.append(i)
            best = pts[i, 1]
    fx = np.array([0.0] + [pts[i, 0] for i in front] + [100.0])
    fy = np.array([base_err] + [pts[i, 1] for i in front] + [pts[front[-1], 1]])
    ax.step(fx, fy, where="post", color="#999", lw=1.2, ls="--", zorder=1, label="Pareto frontier (runs + references)")

    for i, row in T.iterrows():
        c = "#2a9d8f" if row["run"] == "latest" else "#1d4e89"
        ax.scatter(row["ias15_share_pct"], row["errors"], s=110, color=c, zorder=3, edgecolor="white")
        ax.annotate(f"{row['run']}\n{row['ias15_share_pct']:.1f}%, {row['errors']}",
                    (row["ias15_share_pct"], row["errors"]), xytext=(8, 7), textcoords="offset points", fontsize=9)
    ax.scatter([0], [base_err], marker="s", s=90, facecolor="white", edgecolor="#c8553d", lw=2, zorder=3)
    ax.annotate(f"Leapfrog only\n{base_err} errors", (0, base_err), xytext=(10, -8), textcoords="offset points", fontsize=9)
    ax.scatter([100], [0], marker="D", s=80, facecolor="white", edgecolor="#555", lw=2, zorder=3)
    ax.annotate("IAS15 stepped every dt\n(0 errors by definition)", (100, 0), xytext=(-8, 14), textcoords="offset points",
                ha="right", fontsize=9)
    ax.set_xlabel("Mean share of steps taken with IAS15 (%)  — cost")
    ax.set_ylabel(f"Errors (samples with RMS error > {CORRECTION_THRESHOLD:g}, summed)  — inaccuracy")
    ax.set_xlim(-4, 104)
    ax.set_ylim(-3, base_err * 1.08)
    ax.set_title("Accuracy-cost trade-off: one point per run (lower-left is better)")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "fig25_pareto", args)


# --------------------------------------------------------------------------
# Figure 26: resync quality
# --------------------------------------------------------------------------
def fig26(args, runs) -> None:
    name, run_dir = pick_run(runs, args.resync_run)
    files = sim_files(csv_dir_of(run_dir), "resyncs", args.seed_min, args.seed_max)
    frames = [pd.read_csv(f) for f in files if os.path.getsize(f) > 0]
    frames = [f for f in frames if len(f)]
    if not frames:
        print(f"  SKIPPED: no *_resyncs.csv in {csv_dir_of(run_dir)} for seeds {args.seed_min}..{args.seed_max}")
        return
    D = pd.concat(frames, ignore_index=True)
    D = D[(D["dev_before"] > 0) & (D["dev_after"].notna())].copy()
    D["dev_after"] = D["dev_after"].clip(lower=1e-18)
    D["ratio"] = D["dev_after"] / D["dev_before"]

    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    styles = {"budget": ("#e9a03b", "o", "drift-budget resync"), "switch": ("#1d4e89", "s", "resync on switch")}
    for kind, (c, m, lab) in styles.items():
        sub = D[D["kind"] == kind]
        if len(sub):
            ax.scatter(sub["dev_before"], sub["dev_after"], s=14, marker=m, color=c, alpha=0.45, edgecolor="none",
                       label=f"{lab} (n={len(sub)}, median after/before = {sub['ratio'].median():.1e})")
    lo = min(D["dev_before"].min(), D["dev_after"].min()) / 3
    hi = D["dev_before"].max() * 3
    g = np.array([lo, hi])
    ax.plot(g, g, "k-", lw=1, label="y = x  (no improvement)")
    ax.plot(g, g / 1e3, "k--", lw=1, label="y = x / 1000")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(max(lo / 1e3, 1e-18), hi)
    ax.set_xlabel("True error before resync (RMS position, vs IAS15)")
    ax.set_ylabel("True error after resync")
    ax.set_title(f"Richardson resync quality (run {name}, {len(D)} resyncs)")
    ax.legend(loc="upper left", fontsize=8.5)
    frac = (D["ratio"] < 1e-3).mean()
    ax.text(0.98, 0.03, f"overall median ratio {D['ratio'].median():.1e}\n{100 * frac:.0f}% of resyncs reach ≤ x/1000",
            transform=ax.transAxes, ha="right", fontsize=9.5)
    save(fig, "fig26_resync_quality", args)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Batch 2: figures built from existing files and from export_model_results.py
# --------------------------------------------------------------------------
STRETCH_ALL = (0, 10**12)  # stretches are not restricted to the 956-987 family
ERR_COLORS = {"accumulated_drift": "#c8553d", "improvable": "#e9a03b", "after_adequate_rewinds": "#3b6ea5",
              "undetected": "#777777"}


def read_sim(run_dir: str, sim: str, suffix: str = "validation"):
    p = os.path.join(csv_dir_of(run_dir), f"{sim}_{suffix}.csv")
    return pd.read_csv(p) if os.path.isfile(p) and _has_header(p) else None


def truthy(s: pd.Series) -> np.ndarray:
    return s.astype(str).str.lower().eq("true").to_numpy()


def summary_for(args, runs):
    name, d = pick_run(runs, args.summary_run)
    return name, d, load_summary(d, args.seed_min, args.seed_max)


def concat_sim_csvs(run_dir: str, suffix: str, smin: int, smax: int) -> pd.DataFrame:
    frames = []
    for f in sim_files(csv_dir_of(run_dir), suffix, smin, smax):
        d = pd.read_csv(f)
        if len(d):
            d["simulation_id"] = os.path.basename(f)[: -len(f"_{suffix}.csv")]
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---- A. problem and setup ------------------------------------------------
def fig01(args, runs) -> None:
    name, d, s = summary_for(args, runs)
    f = s.dropna(subset=["baseline_first_exceed_time"])
    if args.sim:
        sim = args.sim
    elif f.empty:
        print("  SKIPPED: no experiment where the baseline crosses the threshold")
        return
    else:
        med = f["baseline_first_exceed_time"].median()
        sim = f.loc[(f["baseline_first_exceed_time"] - med).abs().idxmin(), "simulation_id"]
        print(f"  auto-selected {sim} (median first-crossing time); override with --sim")
    df = read_sim(d, sim)
    if df is None:
        print(f"  SKIPPED: validation csv for {sim} not found in {csv_dir_of(d)}")
        return
    t = df["time"].to_numpy()
    err = df.get("baseline_pre_check_error", df["baseline_position_error"]).clip(lower=1e-15)
    row = s[s["simulation_id"] == sim]
    first = float(row["baseline_first_exceed_time"].iloc[0]) if len(row) else np.nan

    fig, ax = plt.subplots(2, 1, figsize=(9.5, 6.8), sharex=True,
                           gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.1})
    ax[0].plot(t, err, color="#c8553d", lw=1.3, label="Leapfrog error vs IAS15 (IAS15 = reference, error 0)")
    ax[0].axhline(CORRECTION_THRESHOLD, color="k", ls="--", lw=1, label=f"correction threshold {CORRECTION_THRESHOLD:g}")
    if np.isfinite(first):
        ax[0].axvline(first, color="#555", ls=":", lw=1.2, label=f"first crossing, t = {first:.1f}")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("RMS position error")
    ax[0].set_title(f"Why the problem exists: Leapfrog error growth ({sim})")
    ax[0].legend(loc="lower right", fontsize=8.8)
    if bool(col(s, "correction_enabled").fillna(False).astype(bool).any()):
        ax[0].text(0.01, 0.97, "run used --correct: the error resets after every snap", transform=ax[0].transAxes,
                   fontsize=8, va="top", color="#555")
    for c, lab, colr in (("baseline_energy_drift", "Leapfrog", "#c8553d"), ("ias15_energy_drift", "IAS15", "#1d4e89")):
        if c in df:
            ax[1].plot(t, df[c].abs().clip(lower=1e-17), color=colr, lw=1.1, label=lab)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("|relative energy drift|")
    ax[1].set_xlabel("Time")
    ax[1].legend(loc="lower right", fontsize=8.8)
    save(fig, "fig01_error_growth", args)


def fig02(args, runs) -> None:
    name, d, s = summary_for(args, runs)
    x = s["baseline_first_exceed_time"]
    failed = x.notna()
    if not failed.any():
        print("  SKIPPED: baseline never crosses the threshold")
        return
    xs = np.sort(x[failed].to_numpy())
    top = xs.max() * 1.18
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True, gridspec_kw={"width_ratios": [2.2, 1]})
    ax.scatter(np.arange(len(xs)), xs, s=42, color="#c8553d", zorder=3, label="baseline crosses threshold")
    n_ok = int((~failed).sum())
    ax.scatter(np.arange(len(xs), len(xs) + n_ok), np.full(n_ok, top), s=42, facecolor="white", edgecolor="#555",
               zorder=3, label="never crosses")
    ax.axhline(np.median(xs), color="#c8553d", ls=":", lw=1.2)
    ax.text(0, np.median(xs), f" median {np.median(xs):.1f}", va="bottom", fontsize=9, color="#c8553d")
    ax.set_xlabel("Experiment (sorted by crossing time)")
    ax.set_ylabel(f"Time of first RMS error > {CORRECTION_THRESHOLD:g}")
    ax.set_title(f"Leapfrog fails in {int(failed.sum())} of {len(s)} experiments")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93), fontsize=9)
    axh.hist(xs, bins=12, orientation="horizontal", color="#c8553d", alpha=0.75)
    axh.set_xlabel("count")
    save(fig, "fig02_first_crossing", args)


# ---- B. safety-index derivation -----------------------------------------
def fig04(args, runs) -> None:
    if not need_analysis(args, "feature_corr.csv"):
        return
    C = pd.read_csv(os.path.join(args.analysis_dir, "feature_corr.csv"), index_col=0)
    n = len(C)
    fig, ax = plt.subplots(figsize=(0.95 * n + 2.5, 0.85 * n + 1.8))
    im = ax.imshow(C.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            v = C.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(C.columns, rotation=40, ha="right")
    ax.set_yticklabels(C.index)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r (log10 space)")
    ax.set_title("Feature collinearity: why ridge, not OLS")
    if "log_v_over_r" in C:
        ax.text(0, -0.12, "log_v_over_r = log_rel_velocity - log_nn_distance exactly, so it is excluded from every linear model.",
                transform=ax.transAxes, fontsize=8, color="#555")
    save(fig, "fig04_collinearity", args)


def fig06(args, runs) -> None:
    if not need_analysis(args, "linear_coefficients.csv"):
        return
    c = pd.read_csv(os.path.join(args.analysis_dir, "linear_coefficients.csv"))
    ridge = c[c["model"] == "ridge"].set_index("feature")["std_coef"]
    lasso = c[c["model"] == "lasso"].set_index("feature")["std_coef"].reindex(ridge.index)
    order = ridge.abs().sort_values().index
    ridge, lasso = ridge[order], lasso[order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.6, 0.8 * len(order) + 2))
    ax.barh(y + 0.19, ridge.values, 0.36, color="#3b6ea5", label=f"ridge (C={c[c.model == 'ridge']['C'].iloc[0]:.3g})")
    ax.barh(y - 0.19, lasso.values, 0.36, color="#e07a3f", label=f"lasso (C={c[c.model == 'lasso']['C'].iloc[0]:.3g})")
    for yi, v in zip(y, lasso.values):
        if v == 0:
            ax.text(0, yi - 0.19, "  zeroed by lasso", va="center", fontsize=8.5, color="#e07a3f")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Standardized coefficient (per 1 std of the log10 feature)")
    ax.set_title("Ridge vs lasso: which features lasso drops")
    ax.legend(loc="lower right")
    save(fig, "fig06_ridge_vs_lasso", args)


# Source: core/safety_index.py::raw_safety_index (current, 3-body ridge) and its docstring ("previous version", 2-5 body).
COEF_3BODY = {"log_rel_velocity": 0.6963, "log_jerk": 0.2168, "log_acceleration": 0.4218,
              "log_nn_distance": -0.2614, "log_mass": -2.0146, "log_mass_ratio": -2.5698}
COEF_MIXED = {"log_rel_velocity": -4.979, "log_jerk": 7.484, "log_acceleration": -6.372,
              "log_nn_distance": 8.370, "log_mass": -0.088, "log_mass_ratio": -0.273}


def fig07(args, runs) -> None:
    feats = list(COEF_3BODY)
    y = np.arange(len(feats))[::-1]
    a = np.array([COEF_MIXED[f] for f in feats])
    b = np.array([COEF_3BODY[f] for f in feats])
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(y + 0.19, a, 0.36, color="#8e6bbf", label="fit on mixed 2-5 body data (CV AUC 0.907)")
    ax.barh(y - 0.19, b, 0.36, color="#2a9d8f", label="fit on 3-body data only (CV AUC 0.744)")
    for yi, va, vb in zip(y, a, b):
        ax.text(va + (0.15 if va >= 0 else -0.15), yi + 0.19, f"{va:+.2f}", va="center",
                ha="left" if va >= 0 else "right", fontsize=8.5)
        ax.text(vb + (0.15 if vb >= 0 else -0.15), yi - 0.19, f"{vb:+.2f}", va="center",
                ha="left" if vb >= 0 else "right", fontsize=8.5)
        if np.sign(va) != np.sign(vb):
            ax.text(0.985, yi, "sign flip", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.5,
                    fontweight="bold", color="#d1495b",
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#d1495b"})
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    ax.set_xlabel("Raw coefficient on the log10 feature")
    ax.set_title("Coefficient instability between the 2-5 body and 3-body ridge fits")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8.8)
    ax.set_xlim(min(a.min(), b.min()) - 2, max(a.max(), b.max()) + 4.5)
    save(fig, "fig07_coefficient_instability", args)


def _curves(score, y, w=None):
    ok = np.isfinite(score)
    s, yy = score[ok], y[ok]
    ww = np.ones_like(s) if w is None else w[ok]
    o = np.argsort(-s, kind="stable")
    yy, ww = yy[o], ww[o]
    tp, fp = np.cumsum(ww * yy), np.cumsum(ww * (1 - yy))
    tpr, fpr, prec = tp / tp[-1], fp / fp[-1], tp / (tp + fp)
    fpr0, tpr0 = np.r_[0, fpr], np.r_[0, tpr]
    auc = float(np.sum(np.diff(fpr0) * (tpr0[1:] + tpr0[:-1]) / 2))
    ap = float(np.sum(np.diff(np.r_[0, tpr]) * prec))
    return fpr0, tpr0, tpr, prec, auc, ap


def fig08(args, runs) -> None:
    if not need_analysis(args, "oof_scores.npz"):
        return
    z = np.load(os.path.join(args.analysis_dir, "oof_scores.npz"))
    y, w = z["y"].astype(int), z["w"].astype(float)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.8))
    others = [("lasso", "continuous", "lasso", "#2a9d8f", "-."), ("gbt", "continuous", "GBT (continuous)", "#e07a3f", "--"),
              ("logistic", "rv_only", "relative velocity only", "#888", ":")]
    for kind, feats, lab, c, ls in others:
        k = oof_key(kind, feats)
        if k in z:
            fpr, tpr, rec, prec, auc, ap = _curves(z[k], y)
            a1.plot(fpr, tpr, color=c, ls=ls, lw=1.3, label=f"{lab} (AUC {auc:.3f})")
            a2.plot(rec, prec, color=c, ls=ls, lw=1.3, label=f"{lab} (AP {ap:.3f})")
    sc = z[oof_key("ridge", "continuous")]
    for wt, ls, tag in ((None, "-", "unweighted"), (w, "--", "weighted")):
        fpr, tpr, rec, prec, auc, ap = _curves(sc, y, wt)
        a1.plot(fpr, tpr, color="#1d4e89", ls=ls, lw=2.4 if wt is None else 1.8, label=f"ridge, {tag} (AUC {auc:.3f})")
        a2.plot(rec, prec, color="#1d4e89", ls=ls, lw=2.4 if wt is None else 1.8, label=f"ridge, {tag} (AP {ap:.3f})")
    a1.plot([0, 1], [0, 1], color="#bbb", lw=1)
    a1.set(xlabel="False-positive rate", ylabel="True-positive rate (recall)", title="ROC (out-of-fold)")
    a1.legend(loc="lower right", fontsize=8.5)
    a2.axhline(y.mean(), color="#bbb", lw=1, label=f"prevalence {y.mean():.3f} (enriched sample)")
    a2.axhline((w * y).sum() / w.sum(), color="#bbb", ls="--", lw=1, label=f"weighted prevalence {(w * y).sum() / w.sum():.4f}")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (out-of-fold)", ylim=(0, 1.02))
    a2.legend(loc="upper right", fontsize=8.5)
    save(fig, "fig08_roc_pr", args)


def fig10(args, runs) -> None:
    if not need_analysis(args, "oof_scores.npz"):
        return
    z = np.load(os.path.join(args.analysis_dir, "oof_scores.npz"))
    y = z["y"].astype(int)
    s = z[oof_key("ridge", "continuous")]
    ok = np.isfinite(s)
    lo, hi = np.percentile(s[ok], [0.5, 99.5])
    bins = np.linspace(lo, hi, 70)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.hist(s[ok & (y == 0)], bins=bins, density=True, color="#3b6ea5", alpha=0.55, label="negatives (Leapfrog was fine)")
    ax.hist(s[ok & (y == 1)], bins=bins, density=True, color="#c8553d", alpha=0.55, label="corrections")
    for r in RECALL_TARGETS:
        thr = threshold_for_recall(s, y, r)
        if lo <= thr <= hi:
            ax.axvline(thr, color="k", ls="--", lw=0.9)
            ax.text(thr, ax.get_ylim()[1] * 0.97, f" recall {r:.2f}", rotation=90, va="top", fontsize=8.5)
    ax.set_xlabel("Out-of-fold safety logit (ridge, causal features)")
    ax.set_ylabel("Density")
    ax.set_title("Safety logit: corrections vs negatives (overlap = why high-recall thresholds collapse)")
    ax.legend(loc="upper left")
    save(fig, "fig10_logit_distribution", args)


def _situations(args):
    p = os.path.join(args.outputs_dir, "situation_weights.csv")
    if not os.path.isfile(p):
        print(f"  SKIPPED: {p} not found")
        return None
    d = pd.read_csv(p)
    d["unstable"] = d["fold_sign_agreement"].between(0.39, 0.61)
    return d


def fig11(args, runs) -> None:
    d = _situations(args)
    if d is None:
        return
    d = d.sort_values("weight")
    y = np.arange(len(d))
    colors = np.where(d["unstable"], "#e9a03b", np.where(d["weight"] >= 0, "#3b6ea5", "#c8553d"))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(d) + 2))
    ax.barh(y, d["weight"], xerr=d["weight_fold_std"], color=colors, error_kw={"elinewidth": 1, "capsize": 2})
    for i, (_, r) in enumerate(d.iterrows()):
        if r["unstable"]:
            ax.text(0.01, i, f" sign agreement {r['fold_sign_agreement']:.1f}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=8, color="#8a5a00")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d["situation"])
    ax.set_xlabel("Fitted weight (bar = ±1 std across CV folds)")
    ax.set_title("Situation weights")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="#3b6ea5"), plt.Rectangle((0, 0), 1, 1, fc="#c8553d"),
                       plt.Rectangle((0, 0), 1, 1, fc="#e9a03b")],
              labels=["positive", "negative", "unstable (fold sign agreement 0.4-0.6)"], loc="lower right", fontsize=8.5)
    save(fig, "fig11_situation_weights", args)


def fig12(args, runs) -> None:
    d = _situations(args)
    if d is None:
        return
    pos = d[d["lift"] > 0]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.scatter(pos["lift"], pos["weight"], s=60, c=np.where(pos["unstable"], "#e9a03b", "#3b6ea5"), zorder=3)
    focus = {"near_collision", "close_encounter", "rapid_approach"}
    for _, r in pos.iterrows():
        big = r["situation"] in focus or abs(r["weight"]) > 1.5
        ax.annotate(r["situation"], (r["lift"], r["weight"]), xytext=(5, 4), textcoords="offset points",
                    fontsize=9 if big else 7.5, fontweight="bold" if r["situation"] in focus else "normal")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Lift  P(correction | on) / P(correction | off)  (log)")
    ax.set_ylabel("Fitted weight in the joint model")
    ax.set_title("Overlapping flags trade credit: lift vs fitted weight")
    zero = d[d["lift"] <= 0]["situation"].tolist()
    if zero:
        ax.text(0.01, 0.01, f"not shown (lift 0): {', '.join(zero)}", transform=ax.transAxes, fontsize=7.5, color="#555")
    save(fig, "fig12_lift_vs_weight", args)


def fig13(args, runs) -> None:
    import json
    p = os.path.join(args.outputs_dir, "situation_weights.json")
    if not os.path.isfile(p):
        print(f"  SKIPPED: {p} not found")
        return
    j = json.load(open(p))
    cv = j["cv"]
    labels = ["weighted situations\n(CV AUC)", "weighted situations\n(sample-weighted AUC)", "count of flags\n(CV AUC)"]
    vals = [cv["auc"], cv["weighted_auc"], cv["equal_weight_count_auc"]]
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
    bars = ax.bar(range(3), vals, color=["#3b6ea5", "#7fa6cf", "#999"], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0.8, 0.96)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Weighted situations vs count of flags")
    axt.axis("off")
    t = pd.DataFrame(j["thresholds"])
    cells = [[f"{r.target_recall:.2f}", f"{r.recall:.3f}", f"{r.threshold:.2f}", f"{r.safety_threshold_0_1:.3g}",
              f"{100 * r.false_pos_rate_enriched:.1f}%", f"{100 * r.false_pos_rate_weighted:.1f}%"] for r in t.itertuples()]
    tab = axt.table(cellText=cells, colLabels=["target recall", "recall", "logit thr.", "score thr.", "FPR (enriched)",
                                               "FPR (weighted)"], loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    tab.scale(1, 1.8)
    axt.set_title("Threshold table (out-of-fold)")
    save(fig, "fig13_situations_vs_count", args)


# ---- C. switching ---------------------------------------------------------
def fig15(args, runs) -> None:
    name, d, s = summary_for(args, runs)
    cand = s[col(s, "n_switches").fillna(0) > 0].sort_values("simulation_id")
    if cand.empty:
        print("  SKIPPED: no experiment with switches")
        return
    pick = cand.iloc[np.unique(np.linspace(0, len(cand) - 1, min(args.strip_n, len(cand))).round().astype(int))]
    rows, ids, t = [], [], None
    for sim in pick["simulation_id"]:
        df = read_sim(d, sim)
        if df is None:
            continue
        rows.append((df["active_integrator"].astype(str).str.lower() == "ias15").to_numpy())
        ids.append(sim)
        t = df["time"].to_numpy() if t is None else t
    if not rows:
        print("  SKIPPED: no validation csvs found")
        return
    m = min(len(r) for r in rows)
    M = np.vstack([r[:m] for r in rows]).astype(float)
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(ids) + 1.8))
    ax.imshow(M, aspect="auto", interpolation="nearest", extent=[t[0], t[m - 1], len(ids), 0],
              cmap=ListedColormap(["#e8eef5", "#1d4e89"]), vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(ids)) + 0.5)
    ax.set_yticklabels([i.replace("sim_", "") for i in ids], fontsize=8)
    ax.set_xlabel("Time")
    ax.set_title(f"Integrator state per experiment (run {name}): dark = IAS15, light = Leapfrog")
    ax.grid(False)
    save(fig, "fig15_integrator_timeline", args)


# ---- D. diagnostics -------------------------------------------------------
def _stretches(args, runs):
    name, d = pick_run(runs, args.stretch_run)
    S = concat_sim_csvs(d, "stretches", *STRETCH_ALL)
    if S.empty:
        print(f"  SKIPPED: no *_stretches.csv in {csv_dir_of(d)}")
        return name, None
    S["false_positive"] = truthy(S["false_positive"])
    return name, S


def fig18(args, runs) -> None:
    name, S = _stretches(args, runs)
    if S is None:
        return
    dev = S["max_deviation"].clip(lower=1e-12)
    bins = np.logspace(np.log10(dev.min()), np.log10(max(dev.max(), 1)), 50)
    thr = args.fp_threshold
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.hist(dev[S["false_positive"]], bins=bins, color="#2a9d8f", alpha=0.8, label="false positive (IAS15 not needed)")
    ax.hist(dev[~S["false_positive"]], bins=bins, color="#c8553d", alpha=0.8, label="needed")
    ax.axvline(thr, color="k", ls="--", lw=1.2, label=f"FP threshold {thr:g}")
    ax.set_xscale("log")
    fp_share = S["false_positive"].mean()
    steps_share = S.loc[S["false_positive"], "n_steps"].sum() / S["n_steps"].sum()
    ax.set_xlabel("Max shadow-Leapfrog deviation within the stretch")
    ax.set_ylabel("IAS15 stretches")
    ax.set_title(f"False-positive stretches (run {name}, {len(S)} stretches, {S['simulation_id'].nunique()} experiments)")
    ax.text(0.98, 0.62, f"{100 * fp_share:.0f}% of stretches are false positives\n{100 * steps_share:.0f}% of IAS15 steps were unneeded",
            transform=ax.transAxes, ha="right", fontsize=10)
    ax.legend(loc="upper right")
    save(fig, "fig18_false_positive_hist", args)


def fig19(args, runs) -> None:
    name, S = _stretches(args, runs)
    if S is None:
        return
    n = S["n_steps"].clip(lower=1)
    bins = np.logspace(0, np.log10(n.max() * 1.05), 45)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.hist(n, bins=bins, color="#3b6ea5", alpha=0.85)
    ax.axvline(n.median(), color="k", ls=":", lw=1.2)
    ax.text(n.median(), ax.get_ylim()[1] * 0.95, f" median {n.median():,.0f} steps", fontsize=9, va="top")
    ax.annotate(f"longest: {int(n.max()):,} steps", (n.max(), 1), xytext=(-10, 30), textcoords="offset points",
                ha="right", arrowprops={"arrowstyle": "->"})
    ax.set_xscale("log")
    ax.set_xlabel("Stretch length (steps, log)")
    ax.set_ylabel("IAS15 stretches")
    ax.set_title(f"Stretch length distribution (run {name}, n={len(S)})")
    save(fig, "fig19_stretch_lengths", args)


def _stack(args, runs, cols, labels, colors, title, fname, ylabel) -> None:
    rows = []
    for name, d in runs.items():
        s = load_summary(d, args.seed_min, args.seed_max)
        if all(col(s, c).notna().any() for c in cols):
            rows.append((name, [int(col(s, c).fillna(0).sum()) for c in cols]))
    if not rows:
        print("  SKIPPED: no run has these columns")
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    bottom = np.zeros(len(rows))
    for k, (lab, c) in enumerate(zip(labels, colors)):
        v = np.array([r[1][k] for r in rows], float)
        ax.bar([r[0] for r in rows], v, bottom=bottom, color=c, label=lab)
        for i, (vi, bi) in enumerate(zip(v, bottom)):
            if vi > 0 and vi >= 0.04 * max(1, (np.array([sum(r[1]) for r in rows])).max()):
                ax.text(i, bi + vi / 2, f"{int(vi)}", ha="center", va="center", fontsize=8.5, color="white")
        bottom += v
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8.8, loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, fname, args)
    print("  " + ", ".join(f"{n}: {v}" for n, v in rows))


def fig21(args, runs) -> None:
    _stack(args, runs, ["rewinds_adequate", "rewinds_improvable", "rewinds_drift_dominated"],
           ["adequate", "improvable", "drift-dominated"], ["#2a9d8f", "#e9a03b", "#c8553d"],
           "Rewind classification per run", "fig21_rewind_classification", "Rewinds")


def fig22(args, runs) -> None:
    _stack(args, runs, ["errors_improvable", "errors_accumulated_drift", "errors_after_adequate", "errors_undetected"],
           ["improvable by deeper rollback", "accumulated drift", "after adequate rewind", "undetected"],
           ["#e9a03b", "#c8553d", "#3b6ea5", "#777777"], "Error attribution per run", "fig22_error_attribution",
           "Adaptive errors")


def fig23(args, runs) -> None:
    name, d = pick_run(runs, args.rewinds_run)
    E = concat_sim_csvs(d, "rollback_errors", args.seed_min, args.seed_max)
    if E.empty:
        print(f"  SKIPPED: no rollback_errors rows in {csv_dir_of(d)}")
        return
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for k, c in ERR_COLORS.items():
        sub = E[E["category"] == k]
        if len(sub):
            ax.scatter(sub["time"], sub["error"], s=48, color=c, edgecolor="white", label=f"{k.replace('_', ' ')} ({len(sub)})", zorder=3)
    ax.axhline(CORRECTION_THRESHOLD, color="k", ls="--", lw=1, label=f"threshold {CORRECTION_THRESHOLD:g}")
    ax.set_yscale("log")
    ax.set_xlabel("Time of error")
    ax.set_ylabel("RMS position error at the error")
    ax.set_title(f"When the remaining errors occur (run {name}, {len(E)} errors, {E['simulation_id'].nunique()} experiments)")
    ax.legend(fontsize=8.8)
    save(fig, "fig23_error_timeline", args)


# ---- E. drift repair and ablation ---------------------------------------
def fig27(args, runs) -> None:
    name, d, s = summary_for(args, runs)
    base, adap, t = None, None, None
    for sim in s["simulation_id"]:
        df = read_sim(d, sim)
        if df is None:
            continue
        b, a = truthy(df["baseline_exceeds_threshold"]).astype(int), truthy(df["adaptive_exceeds_threshold"]).astype(int)
        if base is None:
            base, adap, t = b.copy(), a.copy(), df["time"].to_numpy()
        else:
            m = min(len(base), len(b))
            base, adap, t = base[:m] + b[:m], adap[:m] + a[:m], t[:m]
    if base is None:
        print("  SKIPPED: no validation csvs")
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.step(t, np.cumsum(base), where="post", color="#c8553d", lw=2, label=f"Leapfrog only ({int(base.sum())})")
    ax.step(t, np.cumsum(adap), where="post", color="#1d4e89", lw=2, label=f"adaptive, run {name} ({int(adap.sum())})")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Cumulative samples with error > {CORRECTION_THRESHOLD:g} (all experiments)")
    ax.set_title("Cumulative error count: baseline vs adaptive")
    ax.legend(loc="upper left")
    save(fig, "fig27_cumulative_errors", args)


def fig28(args, runs) -> None:
    name, d, s = summary_for(args, runs)
    v = s.set_index("simulation_id")["mean_error_reduction"].dropna().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(np.arange(len(v)), v.values, color=np.where(v.values < 0, "#c8553d", "#1d4e89"))
    ax.axhline(0, color="k", lw=0.8)
    neg = v[v < 0]
    for i, (sim, val) in enumerate(v.items()):
        if val < 0:
            ax.annotate(f"{val:.2f}", (i, val), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=8.5,
                        color="#c8553d")
    ax.set_xticks(np.arange(len(v)))
    ax.set_xticklabels([i.split("_seed")[1] for i in v.index], rotation=90, fontsize=7.5)
    ax.set_xlabel("Experiment (seed)")
    ax.set_ylabel("mean_error_reduction")
    ax.set_title(f"Per-experiment error reduction vs baseline (run {name}); {len(neg)} negative outliers shown in red")
    save(fig, "fig28_error_reduction_ranking", args)


def fig29(args, runs) -> None:
    name, d, s = summary_for(args, runs)
    cols = [("baseline_final_energy_drift", "Leapfrog only", "#c8553d"), ("adaptive_final_energy_drift", "adaptive", "#1d4e89"),
            ("ias15_final_energy_drift", "IAS15", "#2a9d8f")]
    data = [s[c].abs().clip(lower=1e-17).dropna().to_numpy() for c, _, _ in cols]
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": "k"})
    rng = np.random.default_rng(0)
    for i, ((_, lab, c), v) in enumerate(zip(cols, data), start=1):
        bp["boxes"][i - 1].set(facecolor=c, alpha=0.35)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=14, color=c, alpha=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([l for _, l, _ in cols])
    ax.set_ylabel("|final relative energy drift|")
    ax.set_title(f"Energy drift distributions (run {name}, {len(s)} experiments)")
    save(fig, "fig29_energy_drift", args)


def fig30(args, runs) -> None:
    T = build_run_table(runs, args)
    T = T[T["resyncs"] > 0]
    if T.empty:
        print("  SKIPPED: no run with resyncs")
        return
    T = T.sort_values("resyncs")
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.plot(T["resyncs"], T["errors"], color="#999", lw=1.2, zorder=1)
    ax.scatter(T["resyncs"], T["errors"], s=90, color="#1d4e89", zorder=3)
    for _, r in T.iterrows():
        ax.annotate(f"{r['run']}: {r['resyncs']:,} → {r['errors']}", (r["resyncs"], r["errors"]), xytext=(8, 7),
                    textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Total resyncs (log)")
    ax.set_ylabel("Adaptive errors")
    ax.set_title("Budget-resync count vs errors: diminishing returns")
    save(fig, "fig30_resyncs_vs_errors", args)


# ---- F. cost --------------------------------------------------------------
def fig33(args, runs) -> None:
    name, d, s = summary_for(args, runs)
    if "n_rewound_steps" not in s:
        print("  SKIPPED: n_rewound_steps not in the summary")
        return
    share = (100 * s["n_rewound_steps"] / s["n_steps"]).fillna(0).sort_values(ascending=False)
    tot = 100 * s["n_rewound_steps"].sum() / s["n_steps"].sum()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(np.arange(len(share)), share.values, color="#3b6ea5")
    ax.axhline(tot, color="#c8553d", ls="--", lw=1.3, label=f"pooled overhead {tot:.2f}% of all steps")
    ax.set_xticks(np.arange(len(share)))
    ax.set_xticklabels([s.loc[i, "seed"] for i in share.index], rotation=90, fontsize=7.5)
    ax.set_xlabel("Experiment (seed)")
    ax.set_ylabel("Replayed steps (% of total steps)")
    ax.set_title(f"Rewind overhead (run {name}): {int(s['n_rewound_steps'].sum()):,} replayed of {int(s['n_steps'].sum()):,} steps")
    ax.legend()
    save(fig, "fig33_rewind_overhead", args)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Batch 3: schematics, chaos floor, cost, body-count generalization, sweeps
# --------------------------------------------------------------------------
def _need_chaos(args, s, name: str) -> bool:
    if "chaos_rate" not in s or s["chaos_rate"].notna().sum() == 0:
        print(f"  SKIPPED: the summary of run {name!r} has no chaos-control data. Re-run validate_adaptive.py with "
              f"--chaos-control (plus your usual flags).")
        return False
    return True


def fig03(args, runs) -> None:
    series = []
    for tag, label, color in (("all", "mixed 2-5 body data", "#8e6bbf"), ("3body", "3-body data only", "#2a9d8f")):
        p = os.path.join(args.analysis_dir, f"single_feature_corr_{tag}.csv")
        if os.path.isfile(p):
            series.append((label, color, pd.read_csv(p).set_index("feature")["log_log_corr"]))
    if not series:
        print("  SKIPPED: no single_feature_corr_*.csv. Run:\n"
              "      python export_feature_correlations.py --csv-dir outputs/csv --tag all\n"
              "      python export_feature_correlations.py --csv-dir outputs/csv_3body --tag 3body")
        return
    order = series[0][2].abs().sort_values().index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.6, 0.75 * len(order) + 2))
    h = 0.8 / len(series)
    for k, (label, color, s) in enumerate(series):
        v = s.reindex(order).to_numpy()
        ax.barh(y + (k - (len(series) - 1) / 2) * h, v, h * 0.92, color=color, label=label)
        for yi, vi in zip(y + (k - (len(series) - 1) / 2) * h, v):
            if np.isfinite(vi):
                ax.text(vi + (0.01 if vi >= 0 else -0.01), yi, f"{vi:+.2f}", va="center", ha="left" if vi >= 0 else "right", fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([o.replace("pre_correction_", "") for o in order])
    ax.set_xlabel("Pearson r of log10(feature) vs log10(error at trigger)")
    ax.set_title("Single-feature log-log correlations with correction severity")
    ax.legend(loc="lower right", fontsize=8.8)
    save(fig, "fig03_single_feature_corr", args)


def fig16(args, runs) -> None:
    from matplotlib.patches import FancyArrowPatch, Rectangle
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(-0.3, 13.2)
    ax.set_ylim(-2.1, 3.1)
    ax.axis("off")
    ax.grid(False)
    ax.add_patch(Rectangle((0, 0.9), 8.7, 0.5, fc="#e8eef5", ec="#999"))
    ax.text(0.1, 1.55, "Leapfrog (every dt)", fontsize=10, color="#555")
    cps = [1.0, 2.5, 4.0, 5.5, 7.0]
    for i, x in enumerate(cps):
        ax.add_patch(Rectangle((x - 0.17, 0.78), 0.34, 0.74, fc="#e9a03b", ec="#8a5a00", lw=1.2, zorder=3))
        ax.text(x, 0.55, f"CP{i}", ha="center", fontsize=8.5)
    ax.text(0.0, 2.9, "Checkpoint ring: the last N states (every --checkpoint-interval steps) are kept in RAM,\n"
            "the oldest is dropped when a new one is stored.", fontsize=9.5, va="top")
    ax.annotate("", xy=(cps[0], 1.62), xytext=(cps[-1], 1.62), arrowprops={"arrowstyle": "<->", "color": "#8a5a00"})
    ax.text((cps[0] + cps[-1]) / 2, 1.7, "N checkpoints in the ring (CP4 = latest)", ha="center", fontsize=8.5, color="#8a5a00")
    xt = 8.7
    ax.scatter([xt], [1.15], s=170, marker="v", color="#d1495b", zorder=5)
    ax.text(xt, 1.65, "TRIGGER\nscore ≥ threshold", ha="center", fontsize=9.5, color="#d1495b", fontweight="bold")
    ax.add_patch(FancyArrowPatch((xt, 0.8), (cps[3], 0.8), connectionstyle="arc3,rad=0.35", arrowstyle="-|>",
                                 mutation_scale=16, color="#d1495b", lw=1.8, zorder=4))
    ax.text(7.1, 0.12, "restore CP (--rewind-back: 1 = second-latest)", ha="center", fontsize=9, color="#d1495b")
    ax.add_patch(Rectangle((cps[3], -0.75), xt - cps[3], 0.42, fc="#1d4e89", ec="none"))
    ax.text((cps[3] + xt) / 2, -0.54, "IAS15 replay", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
    ax.add_patch(Rectangle((xt, -0.75), 2.6, 0.42, fc="#7fa6cf", ec="none"))
    ax.text(xt + 1.3, -0.54, "IAS15 continues", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
    ax.add_patch(Rectangle((xt + 2.6, -0.75), 1.3, 0.42, fc="#e8eef5", ec="#999"))
    ax.text(xt + 3.25, -0.54, "Leapfrog", ha="center", va="center", fontsize=9, color="#555")
    ax.annotate("", xy=(xt + 2.6, -0.95), xytext=(xt, -0.95), arrowprops={"arrowstyle": "<->", "color": "#555"})
    ax.text(xt + 1.3, -1.5, "score < release threshold\n(+ optional --ias15-hold-steps)", ha="center", fontsize=8.8, color="#555")
    ax.text(cps[3] - 0.15, -1.5, "Richardson resync of the restored state\n(--resync-on-switch, dt/2 shadow saved in the CP)",
            ha="center", fontsize=8.8, color="#1d4e89")
    ax.set_title("Rewind mechanism: trigger, restore, replay with IAS15, release", fontsize=12)
    save(fig, "fig16_rewind_schematic", args)


def fig17(args, runs) -> None:
    thr, rel = args.safety_threshold, args.safety_threshold * args.release_fraction
    t = np.linspace(0, 40, 1600)
    score = thr * (0.72 + 0.28 * np.sin(0.55 * t) + 0.22 * np.sin(2.3 * t + 1) + 0.10 * np.sin(7.1 * t) + 0.35 * np.exp(-((t - 31) / 1.6) ** 2))

    def run(release):
        on, out = False, np.zeros(len(t), bool)
        for i, sc in enumerate(score):
            on = (sc >= thr) or (on and sc >= release)
            out[i] = on
        return out

    hyst, plain = run(rel), run(thr)
    n_h, n_p = int(np.sum(np.diff(hyst.astype(int)) == 1)), int(np.sum(np.diff(plain.astype(int)) == 1))
    fig, ax = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.08})
    ax[0].plot(t, score, color="#444", lw=1.3, label="safety score (illustrative)")
    ax[0].axhline(thr, color="#d1495b", ls="--", lw=1.3, label=f"trigger threshold {thr:g}")
    ax[0].axhline(rel, color="#e07a3f", ls=":", lw=1.5, label=f"release threshold = {args.release_fraction:g} × {thr:g} = {rel:.3g}")
    ax[0].fill_between(t, rel, thr, color="#e07a3f", alpha=0.12, label="hysteresis band")
    ax[0].set_ylabel("Safety score")
    ax[0].set_title("Hysteresis: IAS15 is entered at the threshold and left only below the release threshold")
    ax[0].legend(loc="upper left", fontsize=8.8, ncol=2)
    ax[1].fill_between(t, 0.55, 1, where=hyst, step="mid", color="#1d4e89", label=f"with hysteresis ({n_h} entries)")
    ax[1].fill_between(t, 0, 0.45, where=plain, step="mid", color="#999", label=f"no hysteresis ({n_p} entries)")
    ax[1].set_yticks([])
    ax[1].set_xlabel("Time (illustrative)")
    ax[1].legend(loc="upper left", fontsize=8.8, ncol=2, bbox_to_anchor=(0, 1.35))
    save(fig, "fig17_hysteresis", args)


def fig31(args, runs) -> None:
    labels = ["Leapfrog", "+ dt/2 shadow\n(Richardson-2:\n--resync-on-switch,\n--drift-budget)", "+ dt/4 shadow\n(Richardson-3:\n--richardson-levels 3)"]
    vals = [1, 1 + 2, 1 + 2 + 4]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(range(3), vals, color=["#c8553d", "#e9a03b", "#1d4e89"], width=0.6)
    for b, v, txt in zip(bars, vals, ["1", "1 + 2 = 3", "1 + 2 + 4 = 7"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, txt, ha="center", fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Force evaluations per Leapfrog step")
    ax.set_title("Cost of the drift-repair shadows (per dt of Leapfrog)")
    ax.text(0.5, -0.32, "A shadow at dt/2 takes 2 half-steps and one at dt/4 takes 4 quarter-steps per dt, one force evaluation each.",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555")
    save(fig, "fig31_force_evaluations", args)


def fig32(args, runs) -> None:
    name, d, s = summary_for(args, runs)
    if "cpu_control_s" not in s or s["cpu_control_s"].notna().sum() == 0:
        print(f"  SKIPPED: run {name!r} has no cpu_control_s. Re-run with --chaos-control: without it the IAS15 timing is "
              f"REBOUND's native C run and the comparison is not like-for-like.")
        return
    s = s.sort_values("cpu_control_s").reset_index(drop=True)
    cols = [("cpu_leapfrog_s", "Leapfrog", "#c8553d"), ("cpu_adaptive_s", "adaptive", "#1d4e89"),
            ("cpu_control_s", "IAS15 stepped every dt", "#2a9d8f")]
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for k, (c, lab, colr) in enumerate(cols):
        ax.bar(x + (k - 1) * 0.27, s[c].clip(lower=1e-3), 0.27, color=colr, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in s["seed"]], rotation=90, fontsize=7.5)
    ax.set_xlabel("Experiment (seed), sorted by IAS15-stepped CPU time")
    ax.set_ylabel("CPU seconds (worker process)")
    tot = {lab: s[c].sum() for c, lab, _ in cols}
    ax.set_title(f"CPU time per experiment (run {name}): adaptive = {tot['adaptive'] / tot['IAS15 stepped every dt']:.1f}× and "
                 f"Leapfrog = {tot['Leapfrog'] / tot['IAS15 stepped every dt']:.1f}× the IAS15-stepped total")
    ax.legend(loc="upper left")
    ax.text(0, -0.3, "Adaptive CPU time includes the oracle / shadow work of --analyze-rollback and --detect-false-positives; "
            "re-time without them for a fair cost figure.", transform=ax.transAxes, fontsize=8, color="#555")
    save(fig, "fig32_cpu_time", args)


def fig34(args, runs) -> None:
    name, d, s = summary_for(args, runs)
    if not _need_chaos(args, s, name):
        return
    ok = s[s["chaos_rate"].notna() & (s["chaos_rate"] > 0)].sort_values("chaos_floor_time")
    pick = ok.iloc[np.unique(np.linspace(0, len(ok) - 1, min(6, len(ok))).round().astype(int))]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(pick)))
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw={"width_ratios": [1.8, 1]})
    for c, (_, r) in zip(colors, pick.iterrows()):
        df = read_sim(d, r["simulation_id"])
        if df is None or "control_position_error" not in df:
            continue
        t, ctrl = df["time"].to_numpy(), df["control_position_error"].clip(lower=1e-16).to_numpy()
        ax.plot(t, ctrl, color=c, lw=1.1, label=f"seed {int(r['seed'])}: Lyapunov time {1 / r['chaos_rate']:.1f}")
        grow = (ctrl > 1e-13) & (ctrl < 1e-3) & (t > 0)
        if grow.sum() >= 5:
            slope, icpt = np.polyfit(t[grow], np.log(ctrl[grow]), 1)
            tt = np.linspace(t[grow].min(), t[grow].max(), 50)
            ax.plot(tt, np.exp(icpt + slope * tt), ls="--", color=c, lw=1.5)
    ax.axhline(CORRECTION_THRESHOLD, color="k", ls="--", lw=1, label=f"threshold {CORRECTION_THRESHOLD:g}")
    ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Control error (IAS15 stepped every dt vs IAS15 reference)")
    ax.set_title("Chaos floor: exponential growth of the control error (dashed = fit)")
    ax.legend(fontsize=8, loc="lower right")
    lyap = 1.0 / ok["chaos_rate"]
    axh.hist(lyap, bins=15, color="#3b6ea5", alpha=0.85)
    axh.axvline(lyap.median(), color="k", ls=":", lw=1.2)
    axh.text(lyap.median(), axh.get_ylim()[1] * 0.95, f" median {lyap.median():.1f}", va="top", fontsize=9)
    axh.set_xlabel("Lyapunov time = 1 / fitted rate")
    axh.set_ylabel("Experiments")
    axh.set_title(f"All {len(ok)} experiments with a fit")
    save(fig, "fig34_chaos_floor_growth", args)


def fig35(args, runs) -> None:
    name, d, s = summary_for(args, runs)
    if not _need_chaos(args, s, name):
        return
    cols = [("errors_floor", "floor (pure IAS15 fails too: not fixable)", "#c8553d"),
            ("errors_marginal", "marginal (control error ≥ 0.1 × threshold)", "#e9a03b"),
            ("errors_fixable", "fixable (control still accurate)", "#2a9d8f")]
    tot = [int(s[c].fillna(0).sum()) for c, _, _ in cols]
    w = s[(s[[c for c, _, _ in cols]].fillna(0).sum(axis=1) > 0)].copy()
    w["tot"] = w[[c for c, _, _ in cols]].fillna(0).sum(axis=1)
    w = w.sort_values("tot", ascending=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 2.2]})
    bottom = 0
    for (c, lab, colr), v in zip(cols, tot):
        a1.bar([0], [v], bottom=bottom, color=colr, label=f"{lab}: {v}")
        if v:
            a1.text(0, bottom + v / 2, str(v), ha="center", va="center", color="white", fontweight="bold")
        bottom += v
    a1.set_xticks([0])
    a1.set_xticklabels([f"run {name}\n{sum(tot)} adaptive errors"])
    a1.set_ylabel("Adaptive errors")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    bottom = np.zeros(len(w))
    for c, lab, colr in cols:
        v = w[c].fillna(0).to_numpy(float)
        a2.bar(np.arange(len(w)), v, bottom=bottom, color=colr)
        bottom += v
    a2.set_xticks(np.arange(len(w)))
    a2.set_xticklabels([str(int(x)) for x in w["seed"]], rotation=90, fontsize=8)
    a2.set_xlabel("Experiment (seed), only experiments with errors")
    a2.set_ylabel("Adaptive errors")
    a2.set_title("Which remaining errors are unfixable?")
    save(fig, "fig35_error_floor_split", args)


def fig36(args, runs) -> None:
    name, d, s = summary_for(args, runs)
    if not _need_chaos(args, s, name):
        return
    v = s.set_index("seed")["residual_factor_needed"].dropna().sort_values(ascending=False)
    if v.empty:
        print("  SKIPPED: no experiment has a residual_factor_needed (needs an adaptive error and a positive chaos rate)")
        return
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(np.arange(len(v)), v.values, color="#3b6ea5")
    ax.axhline(1, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(v)))
    ax.set_xticklabels([str(int(i)) for i in v.index], rotation=90, fontsize=8)
    ax.set_xlabel("Experiment (seed), only experiments with an adaptive error")
    ax.set_ylabel("exp(rate × gap): how many times smaller the injected error must be")
    ax.set_title(f"Residual factor needed to reach the chaos floor (run {name}, median {v.median():.2g}×, n={len(v)})")
    save(fig, "fig36_residual_factor", args)


def fig37(args, runs) -> None:
    name, d = pick_run(runs, args.bodycount_run)
    rows = []
    for f in sim_files(csv_dir_of(d), "validation", *STRETCH_ALL):
        try:
            df = pd.read_csv(f, usecols=["body_count", "baseline_exceeds_threshold", "adaptive_exceeds_threshold",
                                         "ias15_step_fraction"])
        except ValueError:
            continue
        if df.empty:
            continue
        rows.append({"seed": seed_of(f), "bodies": int(df["body_count"].iloc[0]),
                     "baseline": int(truthy(df["baseline_exceeds_threshold"]).sum()),
                     "adaptive": int(truthy(df["adaptive_exceeds_threshold"]).sum()),
                     "share": 100 * float(df["ias15_step_fraction"].iloc[-1])})
    if not rows:
        print(f"  SKIPPED: no validation csvs in {csv_dir_of(d)}")
        return
    R = pd.DataFrame(rows)
    G = R.groupby("bodies").agg(n=("seed", "size"), baseline=("baseline", "mean"), adaptive=("adaptive", "mean"),
                                share=("share", "mean"), with_err=("adaptive", lambda x: (x > 0).mean() * 100))
    print(G.round(2).to_string())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5))
    x = np.arange(len(G))
    a1.bar(x - 0.18, G["baseline"], 0.36, color="#c8553d", label="Leapfrog only")
    a1.bar(x + 0.18, G["adaptive"], 0.36, color="#1d4e89", label="adaptive")
    for xi, (_, r) in zip(x, G.iterrows()):
        a1.text(xi, max(r["baseline"], r["adaptive"]) * 1.03 + 0.02, f"n={int(r['n'])}", ha="center", fontsize=8.5)
    a1.set_xticks(x)
    a1.set_xticklabels([f"{b} bodies" for b in G.index])
    a1.set_ylabel(f"Errors per experiment (error > {CORRECTION_THRESHOLD:g})")
    a1.set_title("Errors by body count (index fitted on 3-body only)")
    a1.legend()
    a2.bar(x, G["share"], 0.5, color="#2a9d8f")
    for xi, v in zip(x, G["share"]):
        a2.text(xi, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    a2.set_xticks(x)
    a2.set_xticklabels([f"{b} bodies" for b in G.index])
    a2.set_ylabel("Mean IAS15 step share (%)")
    a2.set_title(f"Cost by body count (run {name} files, all seed families)")
    save(fig, "fig37_by_body_count", args)


def _sweep_points(args, prefix: str):
    pts = []
    root = args.sweeps_dir
    for dname in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        m = re.fullmatch(rf"{prefix}_(.+)", dname)
        p = os.path.join(root, dname, "validation_summary.csv")
        if not m or not os.path.isfile(p):
            continue
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        s = pd.read_csv(p)
        pts.append((v, int(s["adaptive_n_errors"].sum()), 100 * float(s["ias15_step_fraction"].mean()), len(s)))
    return sorted(pts)


def _twin(ax, pts, xlabel: str, log: bool = False) -> None:
    x = [p[0] for p in pts]
    ax.plot(x, [p[1] for p in pts], "o-", color="#1d4e89", lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Adaptive errors", color="#1d4e89")
    ax.set_ylim(bottom=0)
    ax2 = ax.twinx()
    ax2.plot(x, [p[2] for p in pts], "s--", color="#e07a3f", lw=2)
    ax2.set_ylabel("Mean IAS15 step share (%)", color="#e07a3f")
    ax2.set_ylim(bottom=0)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    if log:
        ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x])


SWEEP_HINT = ('      python run_sweeps.py --base-args "<the flags of your latest run>" --groups {g}')


def fig38(args, runs) -> None:
    pts = _sweep_points(args, "threshold")
    if len(pts) < 2:
        print(f"  SKIPPED: no threshold sweep in {args.sweeps_dir}. Run:\n" + SWEEP_HINT.format(g="threshold"))
        return
    fig, ax = plt.subplots(figsize=(8.5, 5))
    _twin(ax, pts, "Safety threshold", log=True)
    ax.set_title(f"Threshold sweep ({pts[0][3]} experiments per point): accuracy vs cost")
    save(fig, "fig38_threshold_sweep", args)


def fig39(args, runs) -> None:
    params = [("checkpoint_count", "checkpoint_count"), ("checkpoint_interval", "checkpoint_interval (steps)"),
              ("rewind_back", "rewind_back")]
    data = [(p, lab, _sweep_points(args, p)) for p, lab in params]
    data = [d for d in data if len(d[2]) >= 2]
    if not data:
        print(f"  SKIPPED: no rewind-parameter sweeps in {args.sweeps_dir}. Run:\n"
              + SWEEP_HINT.format(g="checkpoint_count checkpoint_interval rewind_back"))
        return
    fig, axes = plt.subplots(1, len(data), figsize=(5.6 * len(data), 4.8), squeeze=False)
    for ax, (p, lab, pts) in zip(axes[0], data):
        _twin(ax, pts, lab)
        ax.set_title(f"Sensitivity to {p}")
    save(fig, "fig39_rewind_sensitivity", args)


# --------------------------------------------------------------------------
FIGS = {1: fig01, 2: fig02, 3: fig03, 4: fig04, 5: fig05, 6: fig06, 7: fig07, 8: fig08, 9: fig09, 10: fig10,
        11: fig11, 12: fig12, 13: fig13, 14: fig14, 15: fig15, 16: fig16, 17: fig17, 18: fig18, 19: fig19, 20: fig20,
        21: fig21, 22: fig22, 23: fig23, 24: fig24, 25: fig25, 26: fig26, 27: fig27, 28: fig28, 29: fig29, 30: fig30,
        31: fig31, 32: fig32, 33: fig33, 34: fig34, 35: fig35, 36: fig36, 37: fig37, 38: fig38, 39: fig39}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IRIS report figures (must-have set)")
    p.add_argument("--only", type=int, nargs="+", default=sorted(FIGS), help=f"figure numbers (default: {sorted(FIGS)})")
    p.add_argument("--outputs-dir", default=OUTPUTS)
    p.add_argument("--out-dir", default=None, help="default: <outputs-dir>/report_figures")
    p.add_argument("--analysis-dir", default=None, help="default: <outputs-dir>/analysis (export_model_results.py output)")
    p.add_argument("--pdf", action="store_true", help="also write vector PDFs")
    p.add_argument("--seed-min", type=int, default=956)
    p.add_argument("--seed-max", type=int, default=987)
    p.add_argument("--safety-threshold", type=float, default=0.134, help="deployed threshold (figs 9, 14)")
    p.add_argument("--release-fraction", type=float, default=0.8, help="hysteresis release fraction (fig 14)")
    p.add_argument("--timeseries-run", default="latest", help="run folder for fig 14 (A..G or latest)")
    p.add_argument("--sim", default=None, help="fig 14: simulation_id, e.g. sim_0031_n3_seed987 (default: auto)")
    p.add_argument("--rewinds-run", default="G", help="run folder whose *_rewinds.csv feed fig 20 (G has 8 checkpoint depths)")
    p.add_argument("--resync-run", default="latest", help="run folder whose *_resyncs.csv feed fig 26")
    p.add_argument("--flat-ratio", type=float, default=1.5, help="fig 20: max/min deviation across depths below this = 'flat'")
    p.add_argument("--summary-run", default="latest", help="run folder used by the summary-based figures (1, 2, 15, 27-29, 33)")
    p.add_argument("--stretch-run", default="G", help="run folder whose *_stretches.csv feed figs 18-19 (all seeds are used)")
    p.add_argument("--fp-threshold", type=float, default=CORRECTION_THRESHOLD, help="false-positive deviation threshold (fig 18)")
    p.add_argument("--bodycount-run", default="G", help="run folder whose *_validation.csv (all seed families) feed fig 37")
    p.add_argument("--sweeps-dir", default=None, help="default: <outputs-dir>/sweeps (written by run_sweeps.py; figs 38-39)")
    p.add_argument("--strip-n", type=int, default=12, help="number of experiments in the integrator timeline (fig 15)")
    p.add_argument("--clean-threshold", type=float, default=1e-9, help="fig 20: rollback-clean-threshold used in the runs")
    a = p.parse_args()
    a.out_dir = a.out_dir or os.path.join(a.outputs_dir, "report_figures")
    a.analysis_dir = a.analysis_dir or os.path.join(a.outputs_dir, "analysis")
    a.sweeps_dir = a.sweeps_dir or os.path.join(a.outputs_dir, "sweeps")
    return a


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.outputs_dir)
    print(f"runs found: {list(runs)}  | seeds {args.seed_min}..{args.seed_max}")
    unknown = [n for n in args.only if n not in FIGS]
    if unknown:
        sys.exit(f"not implemented yet: {unknown} (available: {sorted(FIGS)})")
    for n in args.only:
        print(f"[fig {n}] {FIGS[n].__name__}")
        FIGS[n](args, runs)


if __name__ == "__main__":
    main()
