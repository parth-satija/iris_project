"""
run_sweeps.py

Runs validate_adaptive.py repeatedly, changing ONE parameter at a time around a base configuration, and keeps each
run's output in its own folder: outputs/sweeps/<parameter>_<value>/ (validation_summary.csv + csv/). These are the
inputs of figures 38 (threshold sweep) and 39 (checkpoint_count / checkpoint_interval / rewind_back sensitivity) in
plot_report_figures.py.

    python run_sweeps.py --base-args "<the flags of your latest run>" --groups threshold
    python run_sweeps.py --base-args "..." --groups checkpoint_count checkpoint_interval rewind_back
    python run_sweeps.py --base-args "..." --dry-run          # print the commands only

--base-args is your usual validate_adaptive.py command line WITHOUT the swept flag (if it contains it, it is replaced).
Leave --analyze-rollback, --detect-false-positives and --chaos-control out of it: the sweeps only need errors and the
IAS15 share, and those flags make every run much slower.  It must contain --rewind for the three rewind sweeps.

The default values are in SWEEPS below; override one with e.g. --values threshold=0.1,0.2,0.4.
Finished runs are skipped (--force to redo them).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# group -> (validate_adaptive.py flag, default values)
SWEEPS: dict[str, tuple[str, list[str]]] = {
    "threshold": ("--safety-threshold", ["0.05", "0.1", "0.134", "0.2", "0.3", "0.5"]),
    "checkpoint_count": ("--checkpoint-count", ["3", "5", "8", "12"]),
    "checkpoint_interval": ("--checkpoint-interval", ["5", "10", "20", "40"]),
    "rewind_back": ("--rewind-back", ["0", "1", "2", "3"]),
}


def strip_flag(tokens: list[str], flag: str) -> list[str]:
    """Remove `flag value` / `flag=value` from a token list."""
    out, skip = [], False
    for tok in tokens:
        if skip:
            skip = False
            continue
        if tok == flag:
            skip = True
            continue
        if tok.startswith(flag + "="):
            continue
        out.append(tok)
    return out


def flag_value(tokens: list[str], flag: str, default: int) -> int:
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens):
            return int(tokens[i + 1])
        if tok.startswith(flag + "="):
            return int(tok.split("=", 1)[1])
    return default


def main() -> None:
    ap = argparse.ArgumentParser(description="One-parameter-at-a-time sweeps around a base validate_adaptive.py run.")
    ap.add_argument("--base-args", required=True, help="validate_adaptive.py flags of the base run, as one quoted string")
    ap.add_argument("--groups", nargs="+", choices=sorted(SWEEPS), default=sorted(SWEEPS))
    ap.add_argument("--values", nargs="*", default=[], metavar="GROUP=v1,v2", help="override a group's values")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "outputs", "sweeps"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run points that already have a validation_summary.csv")
    args = ap.parse_args()

    overrides = {}
    for item in args.values:
        group, _, vals = item.partition("=")
        if group not in SWEEPS or not vals:
            sys.exit(f"bad --values entry {item!r}; expected GROUP=v1,v2 with GROUP in {sorted(SWEEPS)}")
        overrides[group] = vals.split(",")

    base = shlex.split(args.base_args)
    if any(g != "threshold" for g in args.groups) and "--rewind" not in base:
        sys.exit("--base-args must contain --rewind for the checkpoint_count / checkpoint_interval / rewind_back sweeps")
    base_count = flag_value(base, "--checkpoint-count", 5)
    base_back = flag_value(base, "--rewind-back", 1)

    jobs = []
    for group in args.groups:
        flag, default_values = SWEEPS[group]
        for value in overrides.get(group, default_values):
            if group == "rewind_back" and int(value) >= base_count:
                print(f"skip {group}={value}: needs --checkpoint-count > {value} (base: {base_count})")
                continue
            if group == "checkpoint_count" and int(value) < base_back + 1:
                print(f"skip {group}={value}: needs at least --rewind-back + 1 = {base_back + 1}")
                continue
            out_dir = os.path.join(args.out_root, f"{group}_{value}")
            cmd = [sys.executable, os.path.join(ROOT, "validate_adaptive.py"), *strip_flag(base, flag), flag, value,
                   "--out-dir", out_dir]
            jobs.append((group, value, out_dir, cmd))

    print(f"{len(jobs)} run(s) planned")
    for k, (group, value, out_dir, cmd) in enumerate(jobs, start=1):
        done = os.path.isfile(os.path.join(out_dir, "validation_summary.csv"))
        print(f"\n[{k}/{len(jobs)}] {group}={value} -> {out_dir}" + ("  (already done, skipping)" if done and not args.force else ""))
        print("   " + " ".join(shlex.quote(c) for c in cmd))
        if args.dry_run or (done and not args.force):
            continue
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"   FAILED (exit code {result.returncode}); continuing with the next run")


if __name__ == "__main__":
    main()
