"""
analyze_safety_index.py

Standalone analysis script (no notebook / pytest dependency) that scans
EVERY experiment's per-body correction CSVs under outputs/csv/, fits a
"safety index" (a.k.a. accuracy index) formula that predicts how bad a
Leapfrog/IAS15 divergence is going to be, and reports how well it does --
so a safety_threshold can be chosen with actual data behind it instead of
a guess.

Run directly:

    python analyze_safety_index.py
    python analyze_safety_index.py --csv-dir outputs/csv --min-rows 20
    python analyze_safety_index.py --test-fraction 0.3 --seed 0

--------------------------------------------------------------------------
Why this looks at pre_correction_* columns, not the dense columns
--------------------------------------------------------------------------
calibration/logger.py only ever writes a row for a (body, timestep) when a
correction actually fired AND this body was one of the bodies that
triggered it (see logger.py's module docstring). Every CSV under
outputs/csv/ is therefore already "just the correction events" -- there
is no extra filtering to do here, just concatenation.

Each such row carries TWO different views of the system at that instant:

  1. `acceleration`, `jerk`, `nearest_neighbor_distance`, `relative_velocity`
     -- computed by calibration/events.py from the SAMPLED trajectory,
     which has ALREADY been snapped to the IAS15 reference by the time
     this row exists (see core/leapfrog.py::run_leapfrog_with_correction).
     These describe the corrected state, not the problem.

  2. `pre_correction_acceleration`, `pre_correction_jerk`,
     `pre_correction_nearest_neighbor_distance`,
     `pre_correction_relative_velocity` -- read straight off
     `CorrectionEvent.pre_correction_positions/velocities/acceleration/
     jerk`, i.e. the actual erroneous state Leapfrog was in one internal
     `dt` step before the snap. This is what a real-time predictor would
     actually have available, and is what this script fits against.

(If your outputs/csv/ predates this change, those four columns won't
exist yet -- re-run your calibration batch first. See README.md's
"Running things" section.)

--------------------------------------------------------------------------
What this script does
--------------------------------------------------------------------------
1. Recursively finds every `<simulation_id>_body<NN>.csv` under the given
   csv_dir (skips `*_situations.csv` files -- different schema).
2. Concatenates every non-empty file into one long DataFrame.
3. Reports simple log-log correlations of each candidate predictor
   against `body_position_error_at_trigger` (the per-body severity of
   the divergence at the moment it was caught), across ALL experiments.
4. Fits a multivariate log-log linear regression:

       log10(error) = c0 + sum_i( coef_i * log10(feature_i) )

   which is equivalent to a power-law safety index:

       safety_index = 10**c0 * prod_i( feature_i ** coef_i )

   on a held-out test split, so the reported R^2 isn't just fit-to-itself.
5. Prints the resulting formula plus a suggested starting point for
   `safety_threshold`: the fitted safety_index value at a chosen
   percentile of the observed correction severities (e.g. "the safety
   index value above which 90% of real corrections would have been
   caught").

This script does NOT itself decide correction_threshold or
safety_threshold for you -- picking those is exactly the experiment you
said you already have designed. It just gives you a data-backed formula
and error bars to run that experiment against, instead of a guess.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys

import numpy as np
import pandas as pd

# Candidate predictors: everything here must be computable by Leapfrog
# ALONE, at the moment a step is taken, with no access to the IAS15
# reference. (mass and mass_ratio are structural/static, not dynamical,
# but are included since a heavier or more lopsided system may need a
# tighter safety margin regardless of instantaneous kinematics.)
CANDIDATE_FEATURES: tuple[str, ...] = (
    "pre_correction_acceleration",
    "pre_correction_jerk",
    "pre_correction_nearest_neighbor_distance",
    "pre_correction_relative_velocity",
    "mass",
    "mass_ratio",
    "timestep",
)

TARGET_COLUMN = "body_position_error_at_trigger"


def find_body_csvs(csv_dir: str) -> list[str]:
    """
    Recursively find every per-body calibration CSV under csv_dir,
    excluding the separate *_situations.csv files (different schema,
    produced by calibration/logger.py::save_situation_log).
    """
    matches: list[str] = []
    for root, _dirs, files in os.walk(csv_dir):
        for name in files:
            if not name.endswith(".csv"):
                continue
            if fnmatch.fnmatch(name, "*_situations.csv"):
                continue
            if fnmatch.fnmatch(name, "*_body*.csv"):
                matches.append(os.path.join(root, name))
    return sorted(matches)


def load_all_corrections(csv_dir: str, min_rows: int) -> pd.DataFrame:
    """
    Load and concatenate every per-body correction CSV under csv_dir into
    one long DataFrame, keeping only rows with a real (non-NaN, > 0)
    target value. Files with too few total rows to be worth reading are
    skipped for speed (still counted in the summary as "skipped").
    """
    paths = find_body_csvs(csv_dir)
    frames: list[pd.DataFrame] = []
    n_skipped_empty = 0
    n_missing_columns = 0

    for path in paths:
        try:
            # Cheap pre-check: an empty (header-only) file has exactly one
            # line. Avoids a full pandas parse for the very common case of
            # a body that never triggered a correction.
            with open(path, "r", encoding="utf-8") as fh:
                first_two = [fh.readline() for _ in range(2)]
            if not first_two[1]:
                n_skipped_empty += 1
                continue
        except OSError:
            continue

        df = pd.read_csv(path)
        if len(df) < 1:
            n_skipped_empty += 1
            continue

        missing = [c for c in (TARGET_COLUMN, *CANDIDATE_FEATURES) if c not in df.columns]
        if missing:
            n_missing_columns += 1
            continue

        df["__source_file"] = os.path.relpath(path, csv_dir)
        frames.append(df)

    if n_missing_columns:
        print(
            f"NOTE: skipped {n_missing_columns} file(s) missing the new "
            f"pre_correction_* columns -- these predate the logging change "
            f"and need a re-run of the calibration batch to be usable here.",
            file=sys.stderr,
        )

    if not frames:
        raise SystemExit(
            f"No usable correction CSVs found under {csv_dir!r} "
            f"(scanned {len(paths)} file(s), {n_skipped_empty} were empty). "
            f"Run `python main.py` first to generate calibration data."
        )

    full = pd.concat(frames, ignore_index=True)
    full = full[full[TARGET_COLUMN].notna() & (full[TARGET_COLUMN] > 0)].copy()

    if len(full) < min_rows:
        raise SystemExit(
            f"Only {len(full)} usable correction rows found across "
            f"{len(frames)} file(s) (need >= {min_rows}). Run more "
            f"experiments (see README.md's `--n-experiments` flag) or "
            f"lower --min-rows."
        )

    return full


def log_log_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation of log10(feature) vs log10(target) for every
    candidate feature, purely descriptive (no fitting).
    """
    log_target = np.log10(df[TARGET_COLUMN].to_numpy())
    rows = []
    for feat in CANDIDATE_FEATURES:
        values = df[feat].to_numpy(dtype=float)
        valid = np.isfinite(values) & (values > 0)
        if valid.sum() < 2 or np.std(values[valid]) == 0:
            rows.append({"feature": feat, "n": int(valid.sum()), "log_log_corr": np.nan})
            continue
        log_feat = np.log10(values[valid])
        r = float(np.corrcoef(log_feat, log_target[valid])[0, 1])
        rows.append({"feature": feat, "n": int(valid.sum()), "log_log_corr": r})
    return pd.DataFrame(rows).sort_values("log_log_corr", key=np.abs, ascending=False)


def feature_collinearity(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Pairwise Pearson correlation, in log10-space, between every pair of
    candidate features actually going into the multivariate fit.

    A multivariate log-log regression trades coefficients arbitrarily
    (including flipping their sign) between features that are themselves
    strongly correlated -- e.g. acceleration ~ 1/r^2 and jerk ~ v/r^3 both
    move together with nearest_neighbor_distance in a close encounter, so
    a regression using all three at once can report a backwards-looking
    coefficient for one of them even though every single-feature
    correlation (see log_log_correlations) has the physically sensible
    sign. This table is what to check BEFORE trusting an individual
    coefficient in the fitted formula: any |r| above ~0.6-0.7 between two
    features means their individual coefficients are not reliably
    separable, and only their combined effect (or the fit's R^2) should be
    trusted -- consider dropping one of the pair with --drop.
    """
    values = df[features].to_numpy(dtype=float)
    valid = np.all(np.isfinite(values) & (values > 0), axis=1)
    log_x = np.log10(values[valid])
    corr = np.corrcoef(log_x, rowvar=False)
    return pd.DataFrame(corr, index=features, columns=features)


def fit_power_law(
    df: pd.DataFrame, features: list[str], rng: np.random.Generator, test_fraction: float
) -> dict:
    """
    Fit log10(target) = c0 + sum(coef_i * log10(feature_i)) by ordinary
    least squares on a train split, evaluate R^2 on a held-out test split.

    Returns a dict with the fitted intercept/coefficients (so callers can
    reconstruct the power-law safety_index formula) and both train/test R^2.
    """
    values = df[features].to_numpy(dtype=float)
    valid = np.all(np.isfinite(values) & (values > 0), axis=1)
    sub = df.loc[valid]
    n = len(sub)
    if n < 10:
        raise SystemExit(
            f"Only {n} rows have every candidate feature valid (> 0, finite); "
            f"need at least 10 to fit/evaluate a held-out split."
        )

    x = np.log10(sub[features].to_numpy(dtype=float))
    y = np.log10(sub[TARGET_COLUMN].to_numpy(dtype=float))

    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    a_train = np.column_stack([x[train_idx], np.ones(len(train_idx))])
    coef, *_ = np.linalg.lstsq(a_train, y[train_idx], rcond=None)

    def r_squared(idxs: np.ndarray) -> float:
        a = np.column_stack([x[idxs], np.ones(len(idxs))])
        pred = a @ coef
        actual = y[idxs]
        ss_res = float(np.sum((actual - pred) ** 2))
        ss_tot = float(np.sum((actual - actual.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "features": features,
        "intercept": float(coef[-1]),
        "coefficients": dict(zip(features, coef[:-1].tolist())),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "r2_train": r_squared(train_idx),
        "r2_test": r_squared(test_idx),
        "fitted_log_target_train": (
            np.column_stack([x[train_idx], np.ones(len(train_idx))]) @ coef
        ),
    }


def format_formula(fit: dict) -> str:
    terms = [f"{value:+.3f}*log10({name})" for name, value in fit["coefficients"].items()]
    return f"log10(safety_index) = {fit['intercept']:.3f} " + " ".join(terms)


def suggest_threshold_starting_points(
    df: pd.DataFrame, fit: dict, percentiles: tuple[float, ...]
) -> pd.DataFrame:
    """
    Compute the fitted safety_index for every usable row, then report the
    safety_index value at each requested percentile of OBSERVED
    correction events. A safety_threshold set at the p90 value, for
    example, is a starting point that would have flagged 90% of the
    corrections actually seen in this data as "risky enough to use IAS15"
    -- your own threshold-selection experiment should still be the final
    word, this just gives it a sane starting range instead of a blind
    guess.
    """
    features = fit["features"]
    values = df[features].to_numpy(dtype=float)
    valid = np.all(np.isfinite(values) & (values > 0), axis=1)
    sub = df.loc[valid]

    log_x = np.log10(sub[features].to_numpy(dtype=float))
    coef = np.array([fit["coefficients"][f] for f in features])
    log_safety_index = fit["intercept"] + log_x @ coef
    safety_index = 10**log_safety_index

    rows = []
    for p in percentiles:
        rows.append(
            {
                "percentile_of_observed_corrections": p,
                "safety_index_value": float(np.percentile(safety_index, p)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("---")[0])
    parser.add_argument(
        "--csv-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "csv"),
        help="Root directory to scan recursively for per-body correction CSVs "
        "(default: outputs/csv next to this script).",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=20,
        help="Minimum total usable correction rows required before fitting (default: 20).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
        help="Fraction of rows held out to evaluate the fit (default: 0.25).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the train/test split, for reproducible reports (default: 0).",
    )
    parser.add_argument(
        "--drop",
        nargs="*",
        default=["mass", "mass_ratio", "timestep"],
        help="Candidate features to leave out of the multivariate fit -- pass an "
        "empty list (--drop) to use every candidate. Default drops the three "
        "features that carried near-zero or unstable coefficients across "
        "exploratory fits, to keep the formula honest; re-add them explicitly "
        "if your own data disagrees.",
    )
    args = parser.parse_args()

    print(f"Scanning {args.csv_dir!r} for per-body correction CSVs...")
    df = load_all_corrections(args.csv_dir, args.min_rows)
    n_files = df["__source_file"].nunique()
    n_sims = df["simulation_id"].nunique() if "simulation_id" in df.columns else "?"
    print(
        f"Loaded {len(df)} correction-event rows from {n_files} file(s) "
        f"across {n_sims} experiment(s).\n"
    )

    print("=== Single-feature log-log correlations vs body_position_error_at_trigger ===")
    corr_table = log_log_correlations(df)
    print(corr_table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()

    features = [f for f in CANDIDATE_FEATURES if f not in set(args.drop)]

    print("=== Pairwise feature collinearity (log10-space Pearson r) ===")
    collin = feature_collinearity(df, features)
    print(collin.to_string(float_format=lambda v: f"{v:.2f}"))
    high_pairs = [
        (a, b, collin.loc[a, b])
        for i, a in enumerate(features)
        for b in features[i + 1 :]
        if abs(collin.loc[a, b]) >= 0.6
    ]
    if high_pairs:
        print(
            "\nWARNING: the following feature pairs are strongly correlated "
            "(|r| >= 0.6). The multivariate fit below can flip the SIGN of "
            "either coefficient in a pair like this -- treat individual "
            "coefficients skeptically and prefer the combined formula/R^2, "
            "or re-run with --drop to remove one of each pair:"
        )
        for a, b, r in high_pairs:
            print(f"  {a} <-> {b}: r={r:.2f}")
    print()

    print(f"=== Fitting multivariate power-law safety index on: {features} ===")
    rng = np.random.default_rng(args.seed)
    fit = fit_power_law(df, features, rng, args.test_fraction)
    print(format_formula(fit))
    print(
        f"R^2: train={fit['r2_train']:.3f} (n={fit['n_train']}), "
        f"test={fit['r2_test']:.3f} (n={fit['n_test']})"
    )
    if fit["r2_test"] < 0.3:
        print(
            "NOTE: test R^2 is low -- treat the formula above as a starting "
            "point, not a finished predictor. More corrections (more/longer "
            "experiments) and/or dropping noisy features with --drop will "
            "usually help more than adding features back in.",
        )
    print()

    print("=== Suggested safety_threshold starting points ===")
    thresholds = suggest_threshold_starting_points(df, fit, percentiles=(50.0, 75.0, 90.0, 95.0, 99.0))
    print(thresholds.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    print(
        "\nA safety_threshold near the p90-p95 row is a reasonable place to "
        "start your own experiment: it would flag most of the corrections "
        "actually observed in this data as 'use IAS15' while still leaving "
        "the bulk of quiet, uneventful timesteps on the cheap Leapfrog path."
    )


if __name__ == "__main__":
    main()
