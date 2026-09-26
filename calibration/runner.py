"""
calibration/runner.py

Drives many calibration experiments in parallel using
concurrent.futures.ProcessPoolExecutor. Each worker process independently:

    1. Loads one experiment's JSON configuration.
    2. Runs the custom Leapfrog integrator.
    3. Runs the IAS15 reference integrator (REBOUND).
    4. Compares the two trajectories into a per-body metrics DataFrame.
    5. Saves that DataFrame as one CSV file PER BODY (not one file per
       system) via calibration.logger.save_calibration_csv.
    6. (Currently disabled -- see the commented-out call in
       run_single_experiment below.) Would otherwise save a second, sparse
       PER-BODY situations log via calibration.logger.save_situation_log.

Because every worker writes to files named after its unique
simulation_id (plus a per-body suffix), there is no risk of workers
overwriting or interleaving each other's output.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from calibration.comparer import compare_trajectories
from calibration.generator import ExperimentConfig, load_experiment_config
from calibration.logger import save_calibration_csv

# Per-body situations log is currently disabled (see run_single_experiment) --
# kept importable here, commented out, so re-enabling is a one-line change:
# from calibration.logger import SITUATION_EVENT_COLUMNS, save_situation_log
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
        csv_paths: Paths to the per-body CSV files the worker wrote (one
            per body; logs are generated per body, not per system).
        situation_log_paths: Paths to the per-body situations-log CSV
            files the worker wrote (one per body; sparse, event-style
            logs -- see calibration.logger.save_situation_log). Currently
            always None: generating this log is disabled, see
            run_single_experiment.
        mean_position_error: Mean position_error across all sampled
            timesteps and all bodies for this experiment.
        mean_energy_drift: Mean (signed) energy_drift across all sampled
            timesteps for this experiment.
        n_corrections: Number of online correction events applied during
            this experiment's Leapfrog run (0 if none were needed).
        n_situation_events: Total number of situational-event activations
            across every body and every situation type in this experiment
            (i.e. the combined row count of every per-body situations log).
            Currently always None: computing this is disabled along with
            the situations log itself, see run_single_experiment.
        elapsed_seconds: Wall-clock time this worker took to run.
        error: Error message if the worker failed, else None.
    """

    simulation_id: str
    csv_paths: list[str] | None
    situation_log_paths: list[str] | None
    mean_position_error: float | None
    mean_energy_drift: float | None
    n_corrections: int | None
    n_situation_events: int | None
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
    both integrators, compare, and save one CSV per body. This is the
    function submitted to each ProcessPoolExecutor worker, so it must be
    a plain, top-level, picklable function (no closures, no lambdas).

    Args:
        config_path: Path to the experiment's JSON configuration file.
        csv_dir: Directory to write this experiment's per-body output
            CSVs into.

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
        csv_paths = save_calibration_csv(df, csv_dir, config.simulation_id)

        # Per-body situations log: disabled for now. Uncomment these two
        # lines (and the matching import above) to re-enable.
        # situation_log_paths = save_situation_log(df, csv_dir, config.simulation_id)
        # n_situation_events = int(df[list(SITUATION_EVENT_COLUMNS)].to_numpy().sum())
        situation_log_paths = None
        n_situation_events = None

        elapsed = time.perf_counter() - start
        return WorkerResult(
            simulation_id=config.simulation_id,
            csv_paths=csv_paths,
            situation_log_paths=situation_log_paths,
            mean_position_error=float(df["position_error"].mean()),
            mean_energy_drift=float(df["energy_drift"].mean()),
            n_corrections=len(corrections),
            n_situation_events=n_situation_events,
            elapsed_seconds=elapsed,
        )
    except Exception as exc:  # noqa: BLE001 - report failures without killing the pool
        elapsed = time.perf_counter() - start
        return WorkerResult(
            simulation_id=config.simulation_id,
            csv_paths=None,
            situation_log_paths=None,
            mean_position_error=None,
            mean_energy_drift=None,
            n_corrections=None,
            n_situation_events=None,
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
        csv_dir: Directory to write per-experiment, per-body CSV output into.
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
                    n_files = len(result.csv_paths) if result.csv_paths else 0
                    print(
                        f"[{completed}/{total}] OK   {result.simulation_id} "
                        f"(bodies={n_files}, "
                        f"pos_err={result.mean_position_error:.3e}, "
                        f"energy_drift={result.mean_energy_drift:.3e}, "
                        f"corrections={result.n_corrections}, "
                        # situations count omitted while the situations log is disabled
                        f"{result.elapsed_seconds:.2f}s)"
                    )
                else:
                    print(
                        f"[{completed}/{total}] FAIL {result.simulation_id}: {result.error}"
                    )

    return results
