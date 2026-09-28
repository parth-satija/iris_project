"""
calibration/logger.py

Responsible for persisting calibration comparison results (the per-body,
long-format Pandas DataFrame produced by calibration/comparer.py) to CSV
files under outputs/csv/, with a fixed, validated column schema.

Logs are generated PER BODY, not per system: each experiment now writes
one CSV file per body (`<simulation_id>_body<NN>.csv`) instead of a
single system-wide CSV. Each file contains (a) rows for timesteps where
an online correction fired AND this body was one of the bodies that
actually triggered it -- i.e. its own error also crossed
`correction_threshold`, not merely the system-wide RMS error (see
save_calibration_csv and calibration/comparer.py's
`body_triggered_correction` column) -- and (b) a thinned sample of
NEGATIVE examples (steps where Leapfrog did not need a correction), each
tagged by `sample_role`, so a safety index can be fit against both sides
of the decision. Each row carries that body's own
metrics/situational-event flags and the relevant system-wide context
columns broadcast onto it, so each file is self-contained. A
correction still resets every body's state at once, but a body whose own
deviation never got large enough to matter no longer gets a row logged
for an event it wasn't actually part of.

Keeping this as its own module (rather than inlining `.to_csv()` calls
elsewhere) gives a single place to enforce the CSV schema and file
naming convention, which matters for the parallel runner writing many
files concurrently.
"""

from __future__ import annotations

import os

import pandas as pd

# The exact, required column order for every per-body calibration CSV file.
REQUIRED_COLUMNS: tuple[str, ...] = (
    # identity
    "simulation_id",
    "seed",
    "timestep",
    "body_count",
    "time",
    "body_id",
    "body_name",
    "mass",
    "mass_ratio",
    # this body's own accuracy metrics
    "position_error",
    "velocity_error",
    # this body's own dynamics
    "acceleration",
    "jerk",
    "nearest_neighbor_id",
    "nearest_neighbor_distance",
    "relative_velocity",
    # system-wide state (broadcast onto every body's rows)
    "total_energy",
    "energy_drift",
    "angular_momentum",
    "angular_momentum_drift",
    "system_min_distance",
    "system_min_distance_pair",
    # online-correction bookkeeping (system-wide)
    "correction_id",
    "time_since_last_correction",
    "steps_since_last_correction",
    "error_at_trigger",
    "correction_interval",
    "body_position_error_at_trigger",
    "body_velocity_error_at_trigger",
    "body_triggered_correction",
    "pre_correction_acceleration",
    "pre_correction_jerk",
    "pre_correction_nearest_neighbor_distance",
    "pre_correction_relative_velocity",
    # sample selection: why this row was logged (see calibration/comparer.py)
    "system_position_error_pre_check",
    "sample_role",
    "sampling_weight",
    # situational events: per-body
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
    "temporary_capture_partner_id",
    "strong_mass_ratio_interaction",
    # situational events: system-wide, projected per-body
    "closest_pair_switch",
    "involved_in_closest_pair_switch",
    "hierarchical_configuration",
    "involved_in_hierarchical_tight_pair",
    "near_symmetric_configuration",
    "chaotic_scattering",
    # baseline: the complement of every anomaly flag above
    "stable",
    "quiescent_regime",
)

# Columns that hold True/False situational-event flags. Written out as 0/1
# rather than left as pandas' default "True"/"False" strings: PapaParse's
# dynamicTyping (used by outputs/plot_render.html) only recognizes lowercase
# 'true'/'false' (or all-caps 'TRUE'/'FALSE'), NOT the mixed-case "True"/
# "False" that pandas.to_csv() writes by default -- so those columns would
# silently round-trip into the browser as plain strings, making every
# `=== true` comparison in the dashboard fail and every situation appear
# unflagged even when the underlying data is correct. Writing 0/1 makes the
# column parse as a plain number everywhere (pandas, Excel, PapaParse, any
# other CSV reader) with no reliance on a specific true/false spelling.
BOOLEAN_COLUMNS: tuple[str, ...] = (
    "body_triggered_correction",
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
    "stable",
    "quiescent_regime",
)

# Situational-event columns worth logging as discrete EVENTS (see
# save_situation_log below), i.e. everything BOOLEAN_COLUMNS has except
# "stable" and "body_triggered_correction". "stable" is the instantaneous,
# un-flagged default state -- most samples are stable, so logging every
# stable sample as an "event" would turn a sparse, scannable log back into
# a dense one. "quiescent_regime" IS kept: unlike "stable", it only fires
# on a genuinely sustained calm stretch (see calibration/events.py), which
# is exactly the kind of noteworthy interval a situations log should call
# out. "body_triggered_correction" is excluded because it is not a
# situational event from calibration/events.py at all -- it is the
# ownership flag save_calibration_csv uses to decide which bodies get a
# row for a given correction, and every remaining row in this log is (by
# construction, per save_calibration_csv) already one where it is True.
SITUATION_EVENT_COLUMNS: tuple[str, ...] = tuple(
    col for col in BOOLEAN_COLUMNS if col not in ("stable", "body_triggered_correction")
)

# Extra per-row context copied alongside each logged situation, so a
# situations-log row is meaningful without needing to cross-reference the
# full per-body metrics CSV.
SITUATION_LOG_CONTEXT_COLUMNS: tuple[str, ...] = (
    "position_error",
    "velocity_error",
    "acceleration",
    "jerk",
    "nearest_neighbor_id",
    "nearest_neighbor_distance",
    "relative_velocity",
    "system_min_distance",
    "system_min_distance_pair",
    "temporary_capture_partner_id",
    "pre_correction_acceleration",
    "pre_correction_jerk",
    "pre_correction_nearest_neighbor_distance",
    "pre_correction_relative_velocity",
)

SITUATION_LOG_COLUMNS: tuple[str, ...] = (
    "simulation_id",
    "body_id",
    "body_name",
    "time",
    "situation",
) + SITUATION_LOG_CONTEXT_COLUMNS


def validate_schema(df: pd.DataFrame) -> None:
    """
    Verify that a DataFrame contains exactly the required calibration
    columns (including `body_id`) before it is written to disk.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def save_calibration_csv(df: pd.DataFrame, csv_dir: str, simulation_id: str) -> list[str]:
    """
    Write a per-body calibration comparison DataFrame to CSV under
    `csv_dir`, splitting it into ONE FILE PER BODY (grouped by `body_id`)
    rather than a single per-system file, enforcing the required column
    schema and order on each file.

    Only rows chosen by calibration/comparer.py's sampling policy are
    written (everything else is dropped to keep files small):

      * POSITIVES -- `sample_role == "correction"` rows where this body
        was one of the bodies that actually triggered the correction
        (`body_triggered_correction` is True). A correction resets every
        body's state simultaneously (see
        core.leapfrog.run_leapfrog_with_correction), but the system-wide
        RMS error crossing `correction_threshold` does not mean every
        individual body's own error did too. (At least one body triggers
        every correction, since the RMS can never exceed the largest
        individual contribution, so no correction is dropped entirely.)
      * NEGATIVES -- a deliberately thinned sample of steps where no
        correction fired: `negative_pre_failure` (the last samples
        before each correction), `negative_hard` (near-misses and
        risky-looking moments, thinned), and `negative_baseline` (every
        20th sample). Written for every body. Without these a safety
        index can only be fit to "how bad were the failures", never to
        "where is the boundary between safe and unsafe". Use
        `sampling_weight` to undo the deliberate over-representation of
        hard negatives when honest base rates matter.

    Every other sampled timestep is dropped before writing, so each
    output file is a sparse log rather than a dense per-timestep series.

    Args:
        df: Long-format DataFrame produced by
            calibration.comparer.compare_trajectories, with one row per
            (timestep, body) and a `body_id` column.
        csv_dir: Output directory for CSV files (created if missing).
        simulation_id: Used to build output filenames
            (`<simulation_id>_body<NN>.csv`), one per body, keeping every
            body's log in its own file so parallel workers never write to
            the same file and downstream tooling can read a single
            body's history in isolation.

    Returns:
        List of full paths to the written CSV files, one per body_id, in
        ascending body_id order. A body whose run had zero corrections
        still gets a file, containing only the header row.

    Raises:
        ValueError: If the DataFrame is missing required columns, or has
            no `body_id` column to split on.
    """
    validate_schema(df)
    if "body_id" not in df.columns:
        raise ValueError("DataFrame must contain a 'body_id' column to write per-body logs.")

    os.makedirs(csv_dir, exist_ok=True)

    # Keep (a) correction rows for the bodies that actually triggered the
    # correction (their own error also crossed correction_threshold, not
    # just the system-wide RMS) and (b) the sampled negative examples
    # chosen by calibration/comparer.py's sampling policy, for EVERY body
    # ("Leapfrog was fine here" is true for all of them). Bodies that did
    # not own a correction are dropped from that correction's row: the
    # system failed, but nothing says this body would have, so labelling
    # it either way would add noise. See this function's docstring.
    is_correction = (df["sample_role"] == "correction") & df["body_triggered_correction"]
    is_negative = df["sample_role"].astype(str).str.startswith("negative")
    out_df = df[is_correction | is_negative].copy()

    # Normalize boolean columns to 0/1 before writing (see BOOLEAN_COLUMNS'
    # docstring for why: avoids a True/"True" capitalization mismatch with
    # downstream CSV readers, notably PapaParse in outputs/plot_render.html).
    for col in BOOLEAN_COLUMNS:
        if col in out_df.columns:
            out_df[col] = out_df[col].astype(bool).astype(int)

    paths: list[str] = []
    for body_id in sorted(df["body_id"].unique()):
        body_df = out_df[out_df["body_id"] == body_id]
        path = os.path.join(csv_dir, f"{simulation_id}_body{int(body_id):02d}.csv")
        body_df.to_csv(path, columns=list(REQUIRED_COLUMNS), index=False)
        paths.append(path)
    return paths


def save_situation_log(df: pd.DataFrame, csv_dir: str, simulation_id: str) -> list[str]:
    """
    Write a PER-BODY situations log: a sparse, event-style CSV that lists
    only the (time, situation) pairs where a situational-event flag from
    calibration/events.py was actually active for that body, instead of
    every timestep (see save_calibration_csv for the full dense log).

    This is deliberately a separate file from save_calibration_csv's dense
    per-timestep log: the dense log answers "what was every metric doing at
    every sampled time", while this log answers "what actually happened to
    this body, and when", which is much faster to scan for a body that only
    has a handful of interesting moments across a long run.

    Args:
        df: Long-format DataFrame produced by
            calibration.comparer.compare_trajectories, with one row per
            (timestep, body) and a `body_id` column.
        csv_dir: Output directory for CSV files (created if missing).
        simulation_id: Used to build output filenames
            (`<simulation_id>_body<NN>_situations.csv`), one per body.

    Returns:
        List of full paths to the written per-body situations CSVs, in
        ascending body_id order. A body with zero situational events still
        gets a file, containing only the header row.

    Raises:
        ValueError: If the DataFrame is missing `body_id` or any of the
            columns SITUATION_EVENT_COLUMNS / SITUATION_LOG_CONTEXT_COLUMNS
            requires.
    """
    if "body_id" not in df.columns:
        raise ValueError("DataFrame must contain a 'body_id' column to write per-body logs.")
    required = ("body_name", "time") + SITUATION_EVENT_COLUMNS + SITUATION_LOG_CONTEXT_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing columns required for a situations log: {missing}")

    os.makedirs(csv_dir, exist_ok=True)

    id_vars = ["simulation_id", "body_id", "body_name", "time"] + list(
        SITUATION_LOG_CONTEXT_COLUMNS
    )
    melted = df.melt(
        id_vars=id_vars,
        value_vars=list(SITUATION_EVENT_COLUMNS),
        var_name="situation",
        value_name="active",
    )
    melted = melted[melted["active"].astype(bool)].drop(columns=["active"])
    melted = melted.sort_values(["body_id", "time", "situation"]).reset_index(drop=True)

    paths: list[str] = []
    for body_id in sorted(df["body_id"].unique()):
        body_events = melted[melted["body_id"] == body_id]
        path = os.path.join(csv_dir, f"{simulation_id}_body{int(body_id):02d}_situations.csv")
        body_events.to_csv(path, columns=list(SITUATION_LOG_COLUMNS), index=False)
        paths.append(path)
    return paths
