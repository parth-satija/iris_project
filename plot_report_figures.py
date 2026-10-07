"""
plot_report_figures.py

Report figures for the IRIS adaptive-integrator work. This file implements the
MUST-HAVE set, numbered as in the figure plan:

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
FIGS = {5: fig05, 9: fig09, 14: fig14, 20: fig20, 24: fig24, 25: fig25, 26: fig26}


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
    p.add_argument("--clean-threshold", type=float, default=1e-9, help="fig 20: rollback-clean-threshold used in the runs")
    a = p.parse_args()
    a.out_dir = a.out_dir or os.path.join(a.outputs_dir, "report_figures")
    a.analysis_dir = a.analysis_dir or os.path.join(a.outputs_dir, "analysis")
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
