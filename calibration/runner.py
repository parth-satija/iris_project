"""
calibration/runner.py

Drives many calibration experiments in parallel using
concurrent.futures.ProcessPoolExecutor. Each worker process independently:

    1. Loads one experiment's JSON configuration.
    2. Runs the custom Leapfrog integrator.
    3. Runs the IAS15 reference integrator (REBOUND).
    4. Compares the two trajectories into a metrics DataFrame.
    5. Saves that DataFrame to its own CSV file.

Because every worker writes to a distinct file (named after its unique
simulation_id), there is no risk of workers overwriting or interleaving
each other's output.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from calibration.comparer import compare_trajectories
from calibration.generator import ExperimentConfig, load_experiment_config
from calibration.logger import save_calibration_csv
from core.leapfrog import run_leapfrog_with_correction
from core.physics import Body, System
from core.rebound_reference import run_ias15

DEFAULT_WORKERS: int = 16


@dataclass
class WorkerResult:
    """
    Summary of a single completed worker's experiment, returned to the
    parent process for aggregation and progress reporting.

    Attributes:
        simulation_id: Identifier of the completed experiment.
        csv_path: Path to the CSV file the worker wrote.
        mean_position_error: Mean position_error across all sampled
            timesteps for this experiment.
        mean_energy_drift: Mean (signed) energy_drift across all sampled
            timesteps for this experiment.
        n_corrections: Number of online correction events applied during
            this experiment's Leapfrog run (0 if none were needed).
        elapsed_seconds: Wall-clock time this worker took to run.
        error: Error message if the worker failed, else None.
    """

    simulation_id: str
    csv_path: str | None
    mean_position_error: float | None
    mean_energy_drift: float | None
    n_corrections: int | None
    elapsed_seconds: float
    error: str | None = None


def _config_to_system(config: ExperimentConfig) -> System:
    """
    Convert a loaded ExperimentConfig's initial conditions into a
    core.physics.System ready to hand to either integrator.

    Args:
        config: The experiment configuration to convert.

    Returns:
        A populated System instance.
    """
    bodies = [
        Body(
            mass=config.masses[i],
            position=np.array(config.positions[i], dtype=np.float64),
            velocity=np.array(config.velocities[i], dtype=np.float64),
            name=f"body_{i}",
        )
        for i in range(config.body_count)
    ]
    return System(bodies=bodies, g=config.g, softening=config.softening)


def run_single_experiment(config_path: str, csv_dir: str) -> WorkerResult:
    """
    Execute one full calibration experiment end to end: load config, run
    both integrators, compare, and save the CSV. This is the function
    submitted to each ProcessPoolExecutor worker, so it must be a plain,
    top-level, picklable function (no closures, no lambdas).

    Args:
        config_path: Path to the experiment's JSON configuration file.
        csv_dir: Directory to write this experiment's output CSV into.

    Returns:
        A WorkerResult summarizing what happened (success or failure).
    """
    start = time.perf_counter()
    config = load_experiment_config(config_path)

    try:
        system = _config_to_system(config)

        # IAS15 must run first: run_leapfrog_with_correction needs the full
        # IAS15 reference trajectory available up front so it has something
        # to snap back to the moment its own error exceeds the threshold.
        ias15_traj = run_ias15(
            system,
            total_time=config.total_time,
            sample_interval=config.sample_interval,
        )

        leapfrog_traj, corrections = run_leapfrog_with_correction(
            system,
            dt=config.dt,
            total_time=config.total_time,
            reference_positions=ias15_traj.positions,
            reference_velocities=ias15_traj.velocities,
            reference_times=ias15_traj.times,
            error_threshold=config.correction_threshold,
        )

        df = compare_trajectories(config, leapfrog_traj, ias15_traj, corrections)
        csv_path = save_calibration_csv(df, csv_dir, config.simulation_id)

        elapsed = time.perf_counter() - start
        return WorkerResult(
            simulation_id=config.simulation_id,
            csv_path=csv_path,
            mean_position_error=float(df["position_error"].mean()),
            mean_energy_drift=float(df["energy_drift"].mean()),
            n_corrections=len(corrections),
            elapsed_seconds=elapsed,
        )
    except Exception as exc:  # noqa: BLE001 - report failures without killing the pool
        elapsed = time.perf_counter() - start
        return WorkerResult(
            simulation_id=config.simulation_id,
            csv_path=None,
            mean_position_error=None,
            mean_energy_drift=None,
            n_corrections=None,
            elapsed_seconds=elapsed,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_calibration_batch(
    config_paths: list[str],
    csv_dir: str,
    n_workers: int = DEFAULT_WORKERS,
    show_progress: bool = True,
) -> list[WorkerResult]:
    """
    Run a batch of calibration experiments in parallel using a
    ProcessPoolExecutor, printing progress as each experiment completes.

    Args:
        config_paths: List of paths to JSON experiment configuration files.
        csv_dir: Directory to write per-experiment CSV output into.
        n_workers: Number of worker processes to use (default 16).
        show_progress: If True, print a line per completed experiment.

    Returns:
        List of WorkerResult objects, one per experiment, in COMPLETION
        order (not necessarily the same order as config_paths).
    """
    results: list[WorkerResult] = []
    total = len(config_paths)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(run_single_experiment, path, csv_dir): path for path in config_paths
        }

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            if show_progress:
                if result.error is None:
                    print(
                        f"[{completed}/{total}] OK   {result.simulation_id} "
                        f"(pos_err={result.mean_position_error:.3e}, "
                        f"energy_drift={result.mean_energy_drift:.3e}, "
                        f"corrections={result.n_corrections}, "
                        f"{result.elapsed_seconds:.2f}s)"
                    )
                else:
                    print(
                        f"[{completed}/{total}] FAIL {result.simulation_id}: {result.error}"
                    )

    return results
