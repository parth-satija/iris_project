"""
core/leapfrog_cpp.py

C++ versions of core.leapfrog.run_leapfrog / run_leapfrog_with_correction (same signatures, same return types),
so the pure-Leapfrog baseline is timed with the same compiled force kernel as the adaptive loop in
core/adaptive_cpp.py. One ctypes call per run. Falls back to the Python/Numba versions if the library is missing.
"""

from __future__ import annotations

import ctypes as C

import numpy as np

from core import adaptive_cpp as _ac
from core.leapfrog import CorrectionEvent, Trajectory, run_leapfrog, run_leapfrog_with_correction
from core.physics import System

_P_D = C.POINTER(C.c_double)
_P_I64 = C.POINTER(C.c_int64)


class _LCfg(C.Structure):
    _fields_ = [
        ("abi_version", C.c_int32), ("n", C.c_int32),
        ("g", C.c_double), ("softening", C.c_double), ("dt", C.c_double),
        ("n_steps", C.c_int64), ("steps_per_sample", C.c_int64),
        ("masses", _P_D), ("pos0", _P_D), ("vel0", _P_D),
        ("ref_pos", _P_D), ("ref_vel", _P_D), ("ref_samples", C.c_int64), ("error_threshold", C.c_double),
        ("out_capacity", C.c_int64),
        ("out_times", _P_D), ("out_pos", _P_D), ("out_vel", _P_D),
        ("out_pre_pos", _P_D), ("out_pre_vel", _P_D), ("out_pre_acc", _P_D), ("out_pre_jerk", _P_D),
        ("out_corr_sample", _P_I64), ("out_corr_step", _P_I64), ("out_corr_error", _P_D),
    ]


_ready = False


def _lib():
    global _ready
    lib = _ac._load()
    if lib is None:
        return None
    if not _ready:
        lib.iris_run_leapfrog.argtypes = [C.POINTER(_LCfg), _P_I64, C.c_char_p, C.c_int]
        lib.iris_run_leapfrog.restype = C.c_int
        _ready = True
    return lib


def _d(a: np.ndarray):
    return a.ctypes.data_as(_P_D)


def _call(lib, cfg: _LCfg) -> int:
    n_corr = C.c_int64(0)
    err = C.create_string_buffer(512)
    if lib.iris_run_leapfrog(C.byref(cfg), C.byref(n_corr), err, len(err)) != 0:
        raise RuntimeError(f"iris_run_leapfrog failed: {err.value.decode()}")
    return int(n_corr.value)


def _base_cfg(system: System, dt: float, n_steps: int, sps: int, T: int, keep: list):
    masses = np.ascontiguousarray(system.masses(), dtype=np.float64)
    pos0 = np.ascontiguousarray(system.positions(), dtype=np.float64)
    vel0 = np.ascontiguousarray(system.velocities(), dtype=np.float64)
    n = system.n
    times = np.empty(T)
    pos = np.empty((T, n, 3))
    vel = np.empty((T, n, 3))
    keep += [masses, pos0, vel0, times, pos, vel]
    cfg = _LCfg()
    cfg.abi_version = _ac.ABI_VERSION
    cfg.n, cfg.g, cfg.softening, cfg.dt = n, system.g, system.softening, dt
    cfg.n_steps, cfg.steps_per_sample = n_steps, sps
    cfg.masses, cfg.pos0, cfg.vel0 = _d(masses), _d(pos0), _d(vel0)
    cfg.out_capacity = T
    cfg.out_times, cfg.out_pos, cfg.out_vel = _d(times), _d(pos), _d(vel)
    return cfg, masses, times, pos, vel


def run_leapfrog_fast(system: System, dt: float, total_time: float, sample_interval: float | None = None) -> Trajectory:
    """Drop-in for core.leapfrog.run_leapfrog."""
    lib = _lib() if 2 <= system.n <= _ac.MAX_BODIES else None
    if lib is None:
        return run_leapfrog(system, dt, total_time, sample_interval)
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if total_time <= 0:
        raise ValueError(f"total_time must be positive, got {total_time}")
    if sample_interval is None:
        sample_interval = dt
    spf = sample_interval / dt
    if abs(spf - round(spf)) > 1e-9:
        raise ValueError(
            f"sample_interval ({sample_interval}) must be an integer multiple of dt ({dt}); got steps_per_sample={spf}")
    sps = int(round(spf))
    n_steps = int(round(total_time / dt))
    T = n_steps // sps + 1
    keep: list = []
    cfg, masses, times, pos, vel = _base_cfg(system, dt, n_steps, sps, T, keep)
    _call(lib, cfg)
    return Trajectory(times=times, positions=pos, velocities=vel, masses=masses)


def run_leapfrog_with_correction_fast(
    system: System,
    dt: float,
    total_time: float,
    reference_positions: np.ndarray,
    reference_velocities: np.ndarray,
    reference_times: np.ndarray,
    error_threshold: float,
) -> tuple[Trajectory, list[CorrectionEvent]]:
    """Drop-in for core.leapfrog.run_leapfrog_with_correction."""
    lib = _lib() if 2 <= system.n <= _ac.MAX_BODIES else None
    if lib is None:
        return run_leapfrog_with_correction(system, dt, total_time, reference_positions, reference_velocities,
                                            reference_times, error_threshold)
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if total_time <= 0:
        raise ValueError(f"total_time must be positive, got {total_time}")
    if error_threshold <= 0:
        raise ValueError(f"error_threshold must be positive, got {error_threshold}")
    if reference_times.shape[0] < 1 or abs(float(reference_times[0])) > 1e-9:
        raise ValueError("reference_times must be non-empty and start at 0.0")
    sample_interval = dt if reference_times.shape[0] < 2 else float(reference_times[1] - reference_times[0])
    spf = sample_interval / dt
    if abs(spf - round(spf)) > 1e-9:
        raise ValueError(
            f"reference sample_interval ({sample_interval}) must be an integer multiple of dt ({dt}); got steps_per_sample={spf}")
    sps = int(round(spf))
    n_steps = int(round(total_time / dt))
    T = n_steps // sps + 1
    if reference_positions.shape[0] < T:
        raise ValueError(f"reference has {reference_positions.shape[0]} samples but the run needs {T}")
    n = system.n

    keep: list = []
    cfg, masses, times, pos, vel = _base_cfg(system, dt, n_steps, sps, T, keep)
    ref_pos = np.ascontiguousarray(reference_positions[:T], dtype=np.float64)
    ref_vel = np.ascontiguousarray(reference_velocities[:T], dtype=np.float64)
    pre_pos, pre_vel, pre_acc, pre_jerk = (np.empty((T, n, 3)) for _ in range(4))
    c_sample = np.empty(T, dtype=np.int64)
    c_step = np.empty(T, dtype=np.int64)
    c_err = np.empty(T)
    cfg.ref_pos, cfg.ref_vel, cfg.ref_samples, cfg.error_threshold = _d(ref_pos), _d(ref_vel), T, float(error_threshold)
    cfg.out_pre_pos, cfg.out_pre_vel, cfg.out_pre_acc, cfg.out_pre_jerk = _d(pre_pos), _d(pre_vel), _d(pre_acc), _d(pre_jerk)
    cfg.out_corr_sample = c_sample.ctypes.data_as(_P_I64)
    cfg.out_corr_step = c_step.ctypes.data_as(_P_I64)
    cfg.out_corr_error = _d(c_err)
    n_corr = _call(lib, cfg)

    corrections: list[CorrectionEvent] = []
    last_step, last_time = 0, 0.0
    for k in range(n_corr):
        s, step = int(c_sample[k]), int(c_step[k])
        t = step * dt
        corrections.append(CorrectionEvent(
            correction_id=k + 1, sample_index=s, time=t, step_index=step, error_at_trigger=float(c_err[k]),
            steps_since_last_correction=step - last_step, time_since_last_correction=t - last_time,
            correction_interval=t - last_time,
            corrected_positions=reference_positions[s].copy(), corrected_velocities=reference_velocities[s].copy(),
            pre_correction_positions=pre_pos[s].copy(), pre_correction_velocities=pre_vel[s].copy(),
            pre_correction_acceleration=pre_acc[s].copy(), pre_correction_jerk=pre_jerk[s].copy(),
        ))
        last_step, last_time = step, t

    trajectory = Trajectory(
        times=times, positions=pos, velocities=vel, masses=masses,
        pre_check_positions=pre_pos, pre_check_velocities=pre_vel,
        pre_check_acceleration=pre_acc, pre_check_jerk=pre_jerk,
    )
    return trajectory, corrections
