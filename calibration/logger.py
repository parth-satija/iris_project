"""
calibration/logger.py

Responsible for persisting calibration comparison results (Pandas
DataFrames produced by calibration/comparer.py) to CSV files under
outputs/csv/, with a fixed, validated column schema.

Keeping this as its own module (rather than inlining `.to_csv()` calls
elsewhere) gives a single place to enforce the CSV schema and file
naming convention, which matters for the parallel runner writing many
files concurrently.
"""

from __future__ import annotations

import os

import pandas as pd

# The exact, required column order for every calibration CSV file.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "simulation_id",
    "seed",
    "timestep",
    "body_count",
    "time",
    "mass_ratio",
    "min_distance",
    "relative_velocity",
    "acceleration",
    "jerk",
    "position_error",
    "velocity_error",
    "total_energy",
    "energy_drift",
    "angular_momentum",
    "angular_momentum_drift",
    "correction_id",
    "time_since_last_correction",
    "steps_since_last_correction",
    "error_at_trigger",
    "correction_interval",
)


def validate_schema(df: pd.DataFrame) -> None:
    """
    Verify that a DataFrame contains exactly the required calibration
    columns before it is written to disk.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def save_calibration_csv(df: pd.DataFrame, csv_dir: str, simulation_id: str) -> str:
    """
    Write a calibration comparison DataFrame to CSV under `csv_dir`,
    enforcing the required column schema and order.

    Args:
        df: DataFrame produced by calibration.comparer.compare_trajectories.
        csv_dir: Output directory for CSV files (created if missing).
        simulation_id: Used to build the output filename
            (`<simulation_id>.csv`), keeping one file per experiment so
            parallel workers never write to the same file.

    Returns:
        The full path to the written CSV file.

    Raises:
        ValueError: If the DataFrame is missing required columns.
    """
    validate_schema(df)
    os.makedirs(csv_dir, exist_ok=True)
    path = os.path.join(csv_dir, f"{simulation_id}.csv")
    df.to_csv(path, columns=list(REQUIRED_COLUMNS), index=False)
    return path
