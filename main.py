"""
main.py

Entry point for the IRIS calibration experiment. Running:

    python main.py

will:
    1. Generate 16 randomized experiment configurations.
    2. Launch 16 parallel calibration experiments (Leapfrog vs IAS15).
    3. Save each experiment's per-timestep comparison metrics as CSV.
    4. Print a summary: experiments completed, runtime, average position
       error, average energy drift.

An optional validation trajectory plot can be produced with --plot.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from calibration.generator import DEFAULT_CORRECTION_THRESHOLD, generate_calibration_batch
from calibration.runner import DEFAULT_WORKERS, run_calibration_batch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "experiments", "configs")
CSV_DIR = os.path.join(PROJECT_ROOT, "outputs", "csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

N_EXPERIMENTS = 16


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for main.py."""
    parser = argparse.ArgumentParser(description="IRIS calibration experiment runner")
    parser.add_argument(
        "--n-experiments",
        type=int,
        default=N_EXPERIMENTS,
        help=f"Number of experiments to generate and run (default: {N_EXPERIMENTS}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel worker processes (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducible experiment generation (default: 42).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0e-3,
        help="Leapfrog integration timestep (default: 1e-3).",
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=5.0,
        help="Total simulated duration per experiment (default: 5.0).",
    )
    parser.add_argument(
        "--correction-threshold",
        type=float,
        default=DEFAULT_CORRECTION_THRESHOLD,
        help=(
            "RMS position-error threshold above which the Leapfrog state is "
            f"snapped back to the IAS15 reference state (default: {DEFAULT_CORRECTION_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--three-body",
        action="store_true",
        help=(
            "Generate only 3-body systems. Results go to their OWN folders "
            "(experiments/configs_3body, outputs/csv_3body) so they never mix with a "
            "regular 2-5 body run. Train on them with: python fit_safety_models.py --three-body"
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate an optional validation trajectory plot after running.",
    )
    return parser.parse_args()


def make_validation_plot(csv_dir: str, figures_dir: str) -> str | None:
    """
    Produce a single optional validation plot: position error over time
    for the first available experiment CSV, purely as a sanity-check
    visualization (not required for the calibration data itself).

    Args:
        csv_dir: Directory containing per-experiment CSV files.
        figures_dir: Directory to save the plot into.

    Returns:
        Path to the saved figure, or None if no CSV files were found.
    """
    import glob

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_files:
        return None

    # Each CSV is now a single BODY's log (see calibration/logger.py), named
    # "<simulation_id>_body<NN>.csv". Use the first one found as a sanity check.
    df = pd.read_csv(csv_files[0])
    sim_id = df["simulation_id"].iloc[0]
    body_name = df["body_name"].iloc[0] if "body_name" in df.columns else "body"
    file_stem = os.path.splitext(os.path.basename(csv_files[0]))[0]

    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(df["time"], df["position_error"])
    axes[0].set_ylabel("Position error")
    axes[0].set_title(f"Leapfrog vs IAS15 calibration: {sim_id} ({body_name})")
    axes[0].set_yscale("log")

    axes[1].plot(df["time"], df["energy_drift"])
    axes[1].set_ylabel("Energy drift")
    axes[1].set_xlabel("Time")

    fig.tight_layout()
    out_path = os.path.join(figures_dir, f"{file_stem}_validation.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    """Run the full IRIS calibration experiment pipeline."""
    args = parse_args()

    start_time = time.perf_counter()

    # 3-body runs get their own directories so they never mix with (or overwrite) a
    # regular 2-5 body dataset: configs and CSVs share simulation ids across runs.
    suffix = "_3body" if args.three_body else ""
    configs_dir = CONFIGS_DIR + suffix
    csv_dir = CSV_DIR + suffix

    print(f"Generating {args.n_experiments} experiment configurations"
          f"{' (3-body systems only)' if args.three_body else ''}...")
    config_paths = generate_calibration_batch(
        n_experiments=args.n_experiments,
        configs_dir=configs_dir,
        base_seed=args.seed,
        dt=args.dt,
        total_time=args.total_time,
        correction_threshold=args.correction_threshold,
        body_counts=(3,) if args.three_body else None,
    )
    print(f"Generated {len(config_paths)} configurations in {configs_dir}\n")

    print(f"Launching {len(config_paths)} calibration experiments across {args.workers} workers...")
    print(f"CSV output: {csv_dir}")
    results = run_calibration_batch(
        config_paths=config_paths,
        csv_dir=csv_dir,
        n_workers=args.workers,
    )

    elapsed = time.perf_counter() - start_time

    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    avg_pos_error = (
        float(np.mean([r.mean_position_error for r in successful])) if successful else float("nan")
    )
    avg_energy_drift = (
        float(np.mean([r.mean_energy_drift for r in successful])) if successful else float("nan")
    )
    total_corrections = sum(r.n_corrections for r in successful if r.n_corrections is not None)

    print("\n===== IRIS Calibration Summary =====")
    print(f"Experiments completed: {len(successful)}/{len(results)}")
    if failed:
        print(f"Experiments failed:    {len(failed)}")
        for r in failed:
            print(f"  - {r.simulation_id}: {r.error}")
    print(f"Total runtime:         {elapsed:.2f} s")
    print(f"Avg position error:    {avg_pos_error:.6e}")
    print(f"Avg energy drift:      {avg_energy_drift:.6e}")
    print(f"Total corrections:     {total_corrections} (threshold={args.correction_threshold:g})")
    print("=====================================\n")

    if args.plot:
        plot_path = make_validation_plot(csv_dir, FIGURES_DIR)
        if plot_path:
            print(f"Validation plot saved to: {plot_path}")
        else:
            print("No CSV files found; skipped validation plot.")


if __name__ == "__main__":
    main()
