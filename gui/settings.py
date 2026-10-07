"""
gui/settings.py

The single source of truth for the GUI's options: one `Flag` per validate_adaptive.py
command-line flag (plus the GUI-only `only_experiment`).

Each flag has two defaults:
    cli_default   what validate_adaptive.py uses when the flag is omitted. Used to build the
                  equivalent command line (only non-default flags are printed) and to read a
                  pasted command.
    latest        the settings of the most recent run, which the GUI shows at start-up.

The flags of that run were never recorded, so `latest` was reconstructed from its outputs
(outputs/validation/): rewind trigger scores, checkpoint spacing / queue depth / chosen depth in
the *_rewinds.csv files, the estimated error at which budget resyncs fire in *_resyncs.csv, the
score at which IAS15 is released, and the cost ratio of the adaptive run to pure Leapfrog.
Flags marked `inferred=True` are the ones that could not be pinned down exactly (shown with a
diamond in the GUI). Fix them with "Paste command..." if you know the real command.

Once the GUI has run something, the settings of the last run are saved to
outputs/gui_runs/last_settings.json and used as the defaults from then on.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from dataclasses import dataclass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(PROJECT_ROOT, "outputs", "gui_runs")
LAST_SETTINGS_PATH = os.path.join(RUNS_DIR, "last_settings.json")

INDEX_SCALE_MODES = ("sigmoid", "minmax", "log_minmax")  # core.safety_index.INDEX_SCALE_MODES
DEFAULT_WORKERS = 16  # calibration.runner.DEFAULT_WORKERS


@dataclass(frozen=True)
class Flag:
    dest: str
    cli: str | None  # None = GUI-only option
    kind: str  # int | float | optfloat | optint | bool | choice | intchoice
    cli_default: object
    latest: object
    group: str
    label: str
    help: str
    choices: tuple = ()
    inferred: bool = False
    aliases: tuple = ()


G_SYS = "System and batch"
G_ADAPT = "Adaptive switching"
G_REWIND = "Rewinding"
G_FP = "False-positive detection"
G_ROLL = "Rollback analysis"
G_DRIFT = "Drift repair"
G_GUI = "This window only"
GROUPS = (G_SYS, G_ADAPT, G_REWIND, G_FP, G_ROLL, G_DRIFT, G_GUI)

FLAGS: tuple[Flag, ...] = (
    # ---- system and batch (flags shared with main.py) ----
    Flag("n_experiments", "--n-experiments", "int", 16, 32, G_SYS, "Experiments",
         "Number of experiments to generate and run. Experiment i uses seed + i. You can watch any "
         "finished experiment in 3D while the others are still running."),
    Flag("workers", "--workers", "int", DEFAULT_WORKERS, DEFAULT_WORKERS, G_SYS, "Workers",
         "Parallel worker processes (one experiment each). Keep at or below your core count.",
         inferred=True),
    Flag("seed", "--seed", "int", 10000, 956, G_SYS, "Seed",
         "Base random seed; experiment i uses seed + i. Use a range disjoint from calibration runs."),
    Flag("dt", "--dt", "float", 1.0e-3, 1.0e-3, G_SYS, "dt", "Leapfrog integration timestep."),
    Flag("total_time", "--total-time", "float", 5.0, 200.0, G_SYS, "Total time",
         "Simulated duration per experiment. 200 means 200,000 steps, which takes minutes per "
         "experiment with the drift-repair options on."),
    Flag("correction_threshold", "--correction-threshold", "float", 1.0e-2, 1.0e-2, G_SYS,
         "Correction threshold",
         "RMS position error that defines a failed sample (the calibration run would have had to "
         "snap Leapfrog back to IAS15). With 'Correct' on, it is also where snapping happens."),
    Flag("three_body", "--three-body", "bool", False, True, G_SYS, "Three-body systems only",
         "Generate only 3-body systems. Without it, experiments cycle through 2, 3, 4 and 5 bodies."),
    Flag("plot", "--plot", "bool", False, False, G_SYS, "Save comparison plot",
         "After the run, save validate_adaptive's comparison PNG for the first experiment."),
    # ---- adaptive switching ----
    Flag("safety_threshold", "--safety-threshold", "float", 0.5, 0.25, G_ADAPT, "Safety threshold",
         "0-1 safety score at or above which the adaptive method uses IAS15. Higher = tolerate more "
         "risk, use IAS15 less."),
    Flag("release_fraction", "--release-fraction", "float", 0.8, 0.6, G_ADAPT, "Release fraction",
         "Hysteresis: once in IAS15, return to Leapfrog only when the score drops below "
         "release fraction x safety threshold (in (0, 1]; 1.0 = no hysteresis)."),
    Flag("ias15_hold_steps", "--ias15-hold-steps", "int", 0, 0, G_ADAPT, "IAS15 hold steps",
         "Steps IAS15 keeps being used after the score has dropped below the release threshold.",
         inferred=True),
    Flag("index_scale", "--index-scale", "choice", "sigmoid", "sigmoid", G_ADAPT, "Index scale",
         "How the raw safety index becomes a 0-1 score: 'sigmoid' for a logit index; 'minmax' / "
         "'log_minmax' need index lo and hi.", choices=INDEX_SCALE_MODES, inferred=True),
    Flag("index_lo", "--index-lo", "optfloat", None, None, G_ADAPT, "Index lo",
         "Raw-index value mapped to score 0 (minmax / log_minmax only)."),
    Flag("index_hi", "--index-hi", "optfloat", None, None, G_ADAPT, "Index hi",
         "Raw-index value mapped to score 1 (minmax / log_minmax only)."),
    Flag("smoke_test_index", "--smoke-test-index", "bool", False, False, G_ADAPT, "Smoke-test index",
         "Use a MEANINGLESS stand-in index (log10 relative velocity). Only to test the pipeline."),
    Flag("situation_index", "--situation-index", "bool", False, False, G_ADAPT, "Situation-weight index",
         "Use the additive situation-weight index (core/situations.py) instead of the fitted formula. "
         "Keep the index scale on sigmoid."),
    Flag("correct", "--correct", "bool", False, True, G_ADAPT, "Online correction",
         "At every sample, if the RMS position error vs IAS15 exceeds the correction threshold, snap "
         "the state back to the IAS15 reference (adaptive run and Leapfrog baseline), as in main.py."),
    # ---- rewinding ----
    Flag("rewind", "--rewind", "bool", False, True, G_REWIND, "Rewind on trigger",
         "Keep checkpoints in RAM; on a trigger restore an earlier one and continue with IAS15 from "
         "there instead of switching at the current state."),
    Flag("rewind_back", "--rewind-back", "int", 1, 4, G_REWIND, "Rewind back",
         "Checkpoints back from the latest to restore (1 = second-latest, 0 = latest). Must be "
         "< checkpoint count."),
    Flag("checkpoint_count", "--checkpoint-count", "int", 5, 8, G_REWIND, "Checkpoint count",
         "Number of checkpoints kept in RAM."),
    Flag("checkpoint_interval", "--checkpoint-interval", "int", 10, 40, G_REWIND, "Checkpoint interval",
         "Steps (of dt) between checkpoints."),
    # ---- false positives ----
    Flag("detect_false_positives", "--detect-false-positives", "bool", False, False, G_FP,
         "Detect false positives",
         "Classify every IAS15 stretch with a shadow Leapfrog (detection only). Costs one extra "
         "Leapfrog step per IAS15 step."),
    Flag("false_positive_threshold", "--false-positive-threshold", "optfloat", None, None, G_FP,
         "False-positive threshold",
         "RMS difference below which a stretch counts as a false positive (empty = the correction "
         "threshold)."),
    # ---- rollback analysis ----
    Flag("analyze_rollback", "--analyze-rollback", "bool", False, True, G_ROLL, "Analyze rollback depth",
         "Count how many errors a deeper rollback could have prevented (needs Rewind). Uses the IAS15 "
         "reference as an oracle; does not change the integration."),
    Flag("rollback_clean_threshold", "--rollback-clean-threshold", "float", 1e-9, 1e-9, G_ROLL,
         "Clean threshold",
         "Largest RMS deviation from the exact state that still counts as an adequate rewind target.",
         inferred=True),
    Flag("rollback_min_gain", "--rollback-min-gain", "float", 0.5, 0.5, G_ROLL, "Minimum gain",
         "A rewind target is effective if it removes at least this fraction (0-1] of the un-rewound "
         "state's error."),
    # ---- drift repair ----
    Flag("drift_budget", "--drift-budget", "float", 0.0, 1e-4, G_DRIFT, "Drift budget",
         "Accumulated-drift budget as a fraction of the correction threshold (0 = off). A dt/2 shadow "
         "Leapfrog estimates the drift; when it reaches the budget the state is resynced to the "
         "Richardson extrapolation."),
    Flag("resync_on_switch", "--resync-on-switch", "bool", False, True, G_DRIFT, "Resync on switch",
         "Apply the Richardson repair to the state IAS15 takes over from, so Leapfrog error is not "
         "frozen into the IAS15 stretch."),
    Flag("rewind_to_anchor", "--rewind-to-anchor", "bool", False, False, G_DRIFT, "Rewind to anchor",
         "With Rewind: restore the last verified state (the anchor) instead of a checkpoint. Long "
         "IAS15 stretches; combine with a drift budget."),
    Flag("richardson_levels", "--richardson-levels", "intchoice", 2, 3, G_DRIFT, "Richardson levels",
         "2: dt/2 shadow, 4th-order repair. 3: also a dt/4 shadow, 6th-order repair (about 7 force "
         "evaluations per Leapfrog step instead of 3).", choices=(2, 3), inferred=True),
    Flag("compensated_sums", "--compensated-sums", "bool", False, False, G_DRIFT, "Compensated (Kahan) sums",
         "Kahan-compensate the position / velocity sums in the Leapfrog step and the shadows. Best "
         "with Richardson levels 3.", inferred=True, aliases=("--kahan",)),
    Flag("chaos_control", "--ias15-stepped", "bool", False, False, G_DRIFT, "IAS15 stepped (chaos control)",
         "Also run pure IAS15 forced to stop at every dt: a like-for-like timing baseline and the "
         "chaos-floor control.", aliases=("--chaos-control",)),
    # ---- GUI only ----
    Flag("only_experiment", None, "optint", None, None, G_GUI, "Run only experiment #",
         "Run just this experiment index (0-based) instead of all of them. Handy for a quick look at "
         "one system. Empty = run all."),
)

FLAG_BY_DEST = {f.dest: f for f in FLAGS}


def latest_defaults() -> dict:
    """Settings of the latest (reconstructed) run."""
    return {f.dest: f.latest for f in FLAGS}


def cli_defaults() -> dict:
    """What validate_adaptive.py uses when a flag is omitted."""
    return {f.dest: f.cli_default for f in FLAGS}


# --------------------------------------------------------------------------- text <-> value
def coerce(flag: Flag, value):
    """Convert a GUI string (or an already typed value) to the flag's type. Raises ValueError."""
    kind = flag.kind
    if kind == "bool":
        return bool(value)
    if isinstance(value, str):
        value = value.strip()
    if kind in ("optfloat", "optint") and (value is None or value == ""):
        return None
    if value is None or value == "":
        raise ValueError("a value is required")
    if kind in ("int", "optint"):
        return int(float(value)) if float(value) == int(float(value)) else _bad_int(value)
    if kind in ("float", "optfloat"):
        return float(value)
    if kind == "intchoice":
        v = int(value)
        if v not in flag.choices:
            raise ValueError(f"must be one of {list(flag.choices)}")
        return v
    if kind == "choice":
        if value not in flag.choices:
            raise ValueError(f"must be one of {list(flag.choices)}")
        return value
    raise ValueError(f"unknown kind {kind}")


def _bad_int(value):
    raise ValueError(f"must be a whole number, got {value}")


def validate(v: dict) -> list[str]:
    """The checks validate_adaptive.py applies (argparse types + main()). Returns messages."""
    e: list[str] = []

    def need(cond, msg):
        if not cond:
            e.append(msg)

    need(v["n_experiments"] >= 1, "Experiments must be at least 1.")
    need(v["workers"] >= 1, "Workers must be at least 1.")
    need(v["dt"] > 0, "dt must be positive.")
    need(v["total_time"] > 0, "Total time must be positive.")
    need(v["correction_threshold"] > 0, "Correction threshold must be positive.")
    need(0.0 <= v["safety_threshold"] <= 1.0, "Safety threshold must be between 0 and 1.")
    need(0.0 < v["release_fraction"] <= 1.0, "Release fraction must be in (0, 1].")
    need(v["ias15_hold_steps"] >= 0, "IAS15 hold steps must be >= 0.")
    need(v["rewind_back"] >= 0, "Rewind back must be >= 0.")
    need(v["checkpoint_count"] >= 1, "Checkpoint count must be >= 1.")
    need(v["checkpoint_interval"] >= 1, "Checkpoint interval must be >= 1.")
    need(v["rollback_clean_threshold"] > 0, "Clean threshold must be positive.")
    need(0.0 < v["rollback_min_gain"] <= 1.0, "Minimum gain must be in (0, 1].")
    need(v["drift_budget"] >= 0, "Drift budget must be >= 0.")
    fp = v["false_positive_threshold"]
    need(fp is None or fp > 0, "False-positive threshold must be positive.")
    if v["index_scale"] in ("minmax", "log_minmax"):
        lo, hi = v["index_lo"], v["index_hi"]
        if lo is None or hi is None:
            e.append(f"Index scale '{v['index_scale']}' needs both index lo and index hi.")
        else:
            need(hi > lo, "Index hi must be greater than index lo.")
            if v["index_scale"] == "log_minmax":
                need(lo > 0, "log_minmax needs index lo > 0.")
    if v["smoke_test_index"] and v["situation_index"]:
        e.append("Choose one of smoke-test index and situation-weight index, not both.")
    if v["analyze_rollback"] and not v["rewind"]:
        e.append("Analyze rollback needs Rewind.")
    if v["rewind_to_anchor"] and not v["rewind"]:
        e.append("Rewind to anchor needs Rewind.")
    if v["rewind"] and v["checkpoint_count"] < v["rewind_back"] + 1:
        e.append(f"Checkpoint count ({v['checkpoint_count']}) must be at least rewind back + 1 "
                 f"({v['rewind_back'] + 1}).")
    steps = v["dt"] > 0 and v["total_time"] / v["dt"]
    if steps and steps < 1:
        e.append("Total time must be at least one dt.")
    only = v["only_experiment"]
    if only is not None:
        need(0 <= only < v["n_experiments"], f"Run-only experiment must be between 0 and {v['n_experiments'] - 1}.")
    return e


# --------------------------------------------------------------------------- command line
def _fmt(x) -> str:
    return repr(x) if isinstance(x, float) else str(x)


def to_command(v: dict) -> str:
    """The validate_adaptive.py command that reproduces these settings (non-default flags only)."""
    parts = ["python", "validate_adaptive.py"]
    for f in FLAGS:
        if f.cli is None:
            continue
        val = v[f.dest]
        if val == f.cli_default:
            continue
        if f.kind == "bool":
            if val:
                parts.append(f.cli)
        elif val is not None:
            parts += [f.cli, _fmt(val)]
    return " ".join(parts)


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # raise instead of sys.exit
        raise ValueError(message)


def from_command(text: str) -> dict:
    """
    Read a validate_adaptive.py command line (anything before the first --flag is ignored, so a
    pasted `python validate_adaptive.py --seed 5 ...` works). Omitted flags take their CLI default.
    Raises ValueError on unknown flags or bad values.
    """
    tokens = shlex.split(text.replace("\\\n", " ").replace("`\n", " "), posix=True)
    while tokens and not tokens[0].startswith("--"):
        tokens.pop(0)
    parser = _Parser(add_help=False, allow_abbrev=False)
    for f in FLAGS:
        if f.cli is None:
            continue
        names = (f.cli,) + f.aliases
        if f.kind == "bool":
            parser.add_argument(*names, dest=f.dest, action="store_true", default=bool(f.cli_default))
        elif f.kind in ("int", "optint"):
            parser.add_argument(*names, dest=f.dest, type=int, default=f.cli_default)
        elif f.kind in ("float", "optfloat"):
            parser.add_argument(*names, dest=f.dest, type=float, default=f.cli_default)
        elif f.kind == "intchoice":
            parser.add_argument(*names, dest=f.dest, type=int, choices=f.choices, default=f.cli_default)
        else:
            parser.add_argument(*names, dest=f.dest, choices=f.choices, default=f.cli_default)
    ns = parser.parse_args(tokens)
    values = cli_defaults()
    values.update(vars(ns))
    return values


# --------------------------------------------------------------------------- persistence
def save_json(path: str, values: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(values, fh, indent=2)


def load_json(path: str) -> dict:
    """Load settings, ignoring unknown keys and filling missing ones from the latest-run defaults."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    values = latest_defaults()
    for dest, val in raw.items():
        if dest in FLAG_BY_DEST:
            values[dest] = val
    return values


def startup_defaults() -> tuple[dict, str]:
    """(settings, source) shown when the window opens: the last GUI run if there is one."""
    if os.path.isfile(LAST_SETTINGS_PATH):
        try:
            return load_json(LAST_SETTINGS_PATH), "the last run of this GUI"
        except (OSError, ValueError):
            pass
    return latest_defaults(), "the latest validate_adaptive.py run (reconstructed from its outputs)"
