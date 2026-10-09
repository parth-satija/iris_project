"""
core/adaptive_cpp.py

Drop-in front end for the C++ port of the adaptive loop (cpp/src/iris_adaptive.cpp).

    from core.adaptive_cpp import run_adaptive_fast
    trajectory, log = run_adaptive_fast(system, dt, ..., same arguments as core.adaptive.run_adaptive)

The whole run is ONE ctypes call: nothing crosses the Python/C++ boundary per step. Features that are validation-only
analysis (detect_false_positives, analyze_rollback) and custom raw-index callables are not in C++; for those
run_adaptive_fast silently delegates to the pure-Python run_adaptive (check `cpp_available()` / `supports()`).

Library lookup: $IRIS_CPP_LIB, else cpp/build/**/iris_adaptive.{dll,so,dylib}.
"""

from __future__ import annotations

import ctypes as C
import glob
import os
import sys
from typing import Callable

import numpy as np

from core.adaptive import AdaptiveLog, ResyncEvent, RewindEvent, SwitchEvent, run_adaptive
from core.leapfrog import Trajectory
from core.physics import System
from core.safety_index import IndexScale, SafetyFeatures, raw_safety_index, smoke_test_raw_index

ABI_VERSION = 2
MAX_BODIES = 16
N_FLAGS = 18

# Must match IRIS_SIT_* in cpp/include/iris_adaptive.h
SITUATION_FLAG_IDS = {
    "ejection_escape": 0,
    "involved_in_hierarchical_tight_pair": 1,
    "near_collision": 2,
    "hierarchical_configuration": 3,
    "rapid_approach": 4,
    "strong_mass_ratio_interaction": 5,
    "acceleration_spike": 6,
    "involved_in_closest_pair_switch": 7,
    "closest_pair_switch": 8,
    "temporary_capture": 9,
    "chaotic_scattering": 10,
    "close_encounter": 11,
    "rapid_separation": 12,
    "high_relative_velocity_encounter": 13,
    "quiescent_regime": 14,
    "strong_acceleration": 15,
    "high_jerk": 16,
    "near_symmetric_configuration": 17,
}

# Coefficients of core.safety_index.raw_safety_index (log10 features). tests/test_adaptive_cpp.py checks that these
# still reproduce the Python formula, so a refit that is pasted into safety_index.py but not here is caught.
FORMULA_COEF = (0.7838, 0.6963, 0.2168, 0.4218, -0.2614, -2.0146, -2.5698)

_SCALE_IDS = {"sigmoid": 0, "minmax": 1, "log_minmax": 2}
_P_D = C.POINTER(C.c_double)


class _Cfg(C.Structure):
    _fields_ = [
        ("abi_version", C.c_int32), ("n", C.c_int32),
        ("g", C.c_double), ("softening", C.c_double), ("dt", C.c_double),
        ("n_steps", C.c_int64), ("steps_per_sample", C.c_int64),
        ("masses", _P_D), ("pos0", _P_D), ("vel0", _P_D),
        ("index_kind", C.c_int32), ("scale_mode", C.c_int32),
        ("scale_lo", C.c_double), ("scale_hi", C.c_double),
        ("safety_threshold", C.c_double), ("release_fraction", C.c_double),
        ("formula_coef", C.c_double * 7),
        ("sit_intercept", C.c_double), ("n_sit_flags", C.c_int32),
        ("sit_flag_id", C.c_int32 * N_FLAGS), ("sit_weight", C.c_double * N_FLAGS),
        ("ref_pos", _P_D), ("ref_vel", _P_D), ("ref_samples", C.c_int64), ("error_threshold", C.c_double),
        ("rewind", C.c_int32), ("checkpoint_count", C.c_int32), ("checkpoint_interval", C.c_int32),
        ("rewind_back", C.c_int32), ("rewind_to_anchor", C.c_int32), ("ias15_hold_steps", C.c_int32),
        ("drift_budget", C.c_double), ("drift_error_threshold", C.c_double),
        ("resync_on_switch", C.c_int32), ("richardson_levels", C.c_int32),
        ("compensated_sums", C.c_int32), ("reserved0", C.c_int32),
        ("ias15_dt_guess", C.c_double),
        ("out_capacity", C.c_int64),
        ("out_times", _P_D), ("out_pos", _P_D), ("out_vel", _P_D), ("out_pre_pos", _P_D), ("out_pre_vel", _P_D),
        ("out_raw", _P_D), ("out_score", _P_D), ("out_ias15_frac", _P_D),
        ("out_active", C.POINTER(C.c_uint8)), ("out_nsw", C.POINTER(C.c_int64)),
    ]


class _Counts(C.Structure):
    _fields_ = [(n, C.c_int64) for n in (
        "n_steps", "n_ias15_steps", "n_corrections", "n_rewinds", "n_rewound_steps", "n_resyncs",
        "n_switch_events", "n_samples", "lf_steps_executed", "ias15_steps_executed", "ias15_substeps", "n_resets")]


class _SwitchEv(C.Structure):
    _fields_ = [("step", C.c_int64), ("time", C.c_double), ("score", C.c_double), ("to_ias15", C.c_int32), ("pad", C.c_int32)]


class _RewindEv(C.Structure):
    _fields_ = [("step", C.c_int64), ("to_step", C.c_int64), ("time", C.c_double), ("to_time", C.c_double),
                ("score", C.c_double), ("depth", C.c_int32), ("pad", C.c_int32)]


class _ResyncEv(C.Structure):
    _fields_ = [("step", C.c_int64), ("time", C.c_double), ("estimated_error", C.c_double),
                ("kind", C.c_int32), ("pad", C.c_int32)]


_lib = None
_lib_error: str | None = None


def _find_library() -> str | None:
    env = os.environ.get("IRIS_CPP_LIB")
    if env:
        return env
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    names = {"win32": ["iris_adaptive.dll", "libiris_adaptive.dll"], "darwin": ["libiris_adaptive.dylib"]}.get(
        sys.platform, ["libiris_adaptive.so"])
    for name in names:
        hits = glob.glob(os.path.join(root, "cpp", "build", "**", name), recursive=True)
        if hits:
            return max(hits, key=os.path.getmtime)
    return None


def _load():
    global _lib, _lib_error
    if _lib is not None or _lib_error is not None:
        return _lib
    path = _find_library()
    if path is None:
        _lib_error = "iris_adaptive library not found (build cpp/ with CMake, or set IRIS_CPP_LIB)"
        return None
    try:
        lib = C.CDLL(path)
        lib.iris_abi_version.restype = C.c_int
        if lib.iris_abi_version() != ABI_VERSION:
            raise RuntimeError(f"ABI mismatch: library {lib.iris_abi_version()} vs wrapper {ABI_VERSION}")
        lib.iris_rebound_version.restype = C.c_char_p
        lib.iris_run.argtypes = [C.POINTER(_Cfg), C.POINTER(C.c_void_p), C.c_char_p, C.c_int]
        lib.iris_run.restype = C.c_int
        lib.iris_get_counts.argtypes = [C.c_void_p, C.POINTER(_Counts)]
        lib.iris_get_events.argtypes = [C.c_void_p, C.c_int, C.c_void_p]
        lib.iris_free.argtypes = [C.c_void_p]
        _lib = lib
    except Exception as exc:  # noqa: BLE001
        _lib_error = f"could not load {path}: {exc}"
    return _lib


def cpp_available() -> bool:
    return _load() is not None


def cpp_unavailable_reason() -> str | None:
    _load()
    return _lib_error


def rebound_c_version() -> str | None:
    lib = _load()
    return lib.iris_rebound_version().decode() if lib else None


def supports(raw_index_fn: Callable, n_bodies: int, detect_false_positives: bool, analyze_rollback: bool) -> bool:
    if detect_false_positives or analyze_rollback or not (2 <= n_bodies <= MAX_BODIES):
        return False
    if raw_index_fn is raw_safety_index or raw_index_fn is smoke_test_raw_index:
        return True
    return type(raw_index_fn).__name__ == "SituationIndex"


def _dptr(a: np.ndarray):
    return a.ctypes.data_as(_P_D)


def run_adaptive_fast(
    system: System,
    dt: float,
    total_time: float,
    sample_interval: float,
    safety_threshold: float,
    index_scale: IndexScale,
    raw_index_fn: Callable[[SafetyFeatures], np.ndarray],
    release_fraction: float = 0.8,
    reference: Trajectory | None = None,
    error_threshold: float | None = None,
    rewind: bool = False,
    checkpoint_count: int = 5,
    checkpoint_interval: int = 10,
    ias15_hold_steps: int = 0,
    rewind_back: int = 1,
    detect_false_positives: bool = False,
    false_positive_threshold: float | None = None,
    analyze_rollback: bool = False,
    oracle: Trajectory | None = None,
    rollback_error_threshold: float | None = None,
    rollback_clean_threshold: float = 1e-9,
    rollback_min_gain: float = 0.5,
    drift_budget: float = 0.0,
    drift_error_threshold: float | None = None,
    rewind_to_anchor: bool = False,
    resync_on_switch: bool = False,
    richardson_levels: int = 2,
    compensated_sums: bool = False,
    ias15_dt_guess: float = 1.0e-3,
) -> tuple[Trajectory, AdaptiveLog]:
    """Same contract as core.adaptive.run_adaptive; runs the loop in C++ when it can, else in Python."""
    python_kwargs = dict(
        system=system, dt=dt, total_time=total_time, sample_interval=sample_interval,
        safety_threshold=safety_threshold, index_scale=index_scale, raw_index_fn=raw_index_fn,
        release_fraction=release_fraction, reference=reference, error_threshold=error_threshold, rewind=rewind,
        checkpoint_count=checkpoint_count, checkpoint_interval=checkpoint_interval, ias15_hold_steps=ias15_hold_steps,
        rewind_back=rewind_back, detect_false_positives=detect_false_positives,
        false_positive_threshold=false_positive_threshold, analyze_rollback=analyze_rollback, oracle=oracle,
        rollback_error_threshold=rollback_error_threshold, rollback_clean_threshold=rollback_clean_threshold,
        rollback_min_gain=rollback_min_gain, drift_budget=drift_budget, drift_error_threshold=drift_error_threshold,
        rewind_to_anchor=rewind_to_anchor, resync_on_switch=resync_on_switch, richardson_levels=richardson_levels,
        compensated_sums=compensated_sums,
    )
    lib = _load()
    if lib is None or not supports(raw_index_fn, system.n, detect_false_positives, analyze_rollback):
        return run_adaptive(**python_kwargs)

    # --- validation identical to run_adaptive (so both paths reject the same inputs) ---
    if dt <= 0 or total_time <= 0:
        raise ValueError("dt and total_time must be positive")
    if not 0.0 <= safety_threshold <= 1.0:
        raise ValueError(f"safety_threshold must be in [0, 1], got {safety_threshold}")
    if not 0.0 < release_fraction <= 1.0:
        raise ValueError(f"release_fraction must be in (0, 1], got {release_fraction}")
    if (reference is None) != (error_threshold is None):
        raise ValueError("reference and error_threshold must be given together (or both omitted)")
    if rewind and checkpoint_count < rewind_back + 1:
        raise ValueError("checkpoint_count must be >= rewind_back + 1")
    if rewind and checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be >= 1")
    if rewind_to_anchor and not rewind:
        raise ValueError("rewind_to_anchor needs rewind=True")
    if drift_budget > 0 and (drift_error_threshold is None or drift_error_threshold <= 0):
        raise ValueError("drift_budget > 0 needs a positive drift_error_threshold")
    if richardson_levels not in (2, 3):
        raise ValueError(f"richardson_levels must be 2 or 3, got {richardson_levels}")
    spf = sample_interval / dt
    if abs(spf - round(spf)) > 1e-9 or spf < 1:
        raise ValueError(f"sample_interval ({sample_interval}) must be a positive integer multiple of dt ({dt})")
    sps = int(round(spf))
    n_steps = int(round(total_time / dt))
    T = n_steps // sps + 1
    if reference is not None and reference.times.shape[0] < T:
        raise ValueError(f"reference has {reference.times.shape[0]} samples but the run needs {T}")

    n = system.n
    masses = np.ascontiguousarray(system.masses(), dtype=np.float64)
    pos0 = np.ascontiguousarray(system.positions(), dtype=np.float64)
    vel0 = np.ascontiguousarray(system.velocities(), dtype=np.float64)

    cfg = _Cfg()
    cfg.abi_version = ABI_VERSION
    cfg.n, cfg.g, cfg.softening, cfg.dt = n, system.g, system.softening, dt
    cfg.n_steps, cfg.steps_per_sample = n_steps, sps
    cfg.masses, cfg.pos0, cfg.vel0 = _dptr(masses), _dptr(pos0), _dptr(vel0)
    cfg.scale_mode = _SCALE_IDS[index_scale.mode]
    cfg.scale_lo = float(index_scale.lo) if index_scale.lo is not None else 0.0
    cfg.scale_hi = float(index_scale.hi) if index_scale.hi is not None else 1.0
    cfg.safety_threshold, cfg.release_fraction = safety_threshold, release_fraction
    for i, v in enumerate(FORMULA_COEF):
        cfg.formula_coef[i] = v
    if raw_index_fn is raw_safety_index:
        cfg.index_kind = 0
    elif raw_index_fn is smoke_test_raw_index:
        cfg.index_kind = 1
    else:  # SituationIndex: weights in the dict's own order (fixes the float accumulation order)
        cfg.index_kind = 2
        cfg.sit_intercept = float(raw_index_fn.intercept)
        items = list(raw_index_fn.weights.items())
        cfg.n_sit_flags = len(items)
        for k, (name, w) in enumerate(items):
            cfg.sit_flag_id[k] = SITUATION_FLAG_IDS[name]
            cfg.sit_weight[k] = float(w)

    correcting = reference is not None
    if correcting:
        ref_pos = np.ascontiguousarray(reference.positions[:T], dtype=np.float64)
        ref_vel = np.ascontiguousarray(reference.velocities[:T], dtype=np.float64)
        cfg.ref_pos, cfg.ref_vel, cfg.ref_samples = _dptr(ref_pos), _dptr(ref_vel), T
        cfg.error_threshold = float(error_threshold)
    cfg.rewind, cfg.checkpoint_count, cfg.checkpoint_interval = int(rewind), checkpoint_count, checkpoint_interval
    cfg.rewind_back, cfg.rewind_to_anchor, cfg.ias15_hold_steps = rewind_back, int(rewind_to_anchor), ias15_hold_steps
    cfg.drift_budget = float(drift_budget)
    cfg.drift_error_threshold = float(drift_error_threshold) if drift_error_threshold is not None else 0.0
    cfg.resync_on_switch, cfg.richardson_levels, cfg.compensated_sums = int(resync_on_switch), richardson_levels, int(compensated_sums)
    cfg.ias15_dt_guess = ias15_dt_guess

    times = np.empty(T)
    pos = np.empty((T, n, 3))
    vel = np.empty((T, n, 3))
    raw = np.empty(T)
    score = np.empty(T)
    frac = np.empty(T)
    active = np.empty(T, dtype=np.uint8)
    nsw = np.empty(T, dtype=np.int64)
    cfg.out_capacity = T
    cfg.out_times, cfg.out_pos, cfg.out_vel = _dptr(times), _dptr(pos), _dptr(vel)
    cfg.out_raw, cfg.out_score, cfg.out_ias15_frac = _dptr(raw), _dptr(score), _dptr(frac)
    cfg.out_active = active.ctypes.data_as(C.POINTER(C.c_uint8))
    cfg.out_nsw = nsw.ctypes.data_as(C.POINTER(C.c_int64))
    if correcting:
        pre_pos = np.empty((T, n, 3))
        pre_vel = np.empty((T, n, 3))
        cfg.out_pre_pos, cfg.out_pre_vel = _dptr(pre_pos), _dptr(pre_vel)

    handle = C.c_void_p()
    err = C.create_string_buffer(512)
    if lib.iris_run(C.byref(cfg), C.byref(handle), err, len(err)) != 0:
        raise RuntimeError(f"iris_run failed: {err.value.decode()}")
    try:
        cnt = _Counts()
        lib.iris_get_counts(handle, C.byref(cnt))
        sw = (_SwitchEv * cnt.n_switch_events)()
        rw = (_RewindEv * cnt.n_rewinds)()
        rs = (_ResyncEv * cnt.n_resyncs)()
        lib.iris_get_events(handle, 0, C.cast(sw, C.c_void_p))
        lib.iris_get_events(handle, 1, C.cast(rw, C.c_void_p))
        lib.iris_get_events(handle, 2, C.cast(rs, C.c_void_p))
    finally:
        lib.iris_free(handle)

    trajectory = Trajectory(
        times=times, positions=pos, velocities=vel, masses=masses,
        pre_check_positions=pre_pos if correcting else None,
        pre_check_velocities=pre_vel if correcting else None,
    )
    switch_events = [SwitchEvent(e.step, e.time, "ias15" if e.to_ias15 else "leapfrog", e.score) for e in sw]
    rewind_events = [RewindEvent(e.step, e.to_step, e.time, e.to_time, e.score, depth=e.depth) for e in rw]
    resync_events = [ResyncEvent(e.step, e.time, e.estimated_error, None, None, "switch" if e.kind else "budget") for e in rs]
    log = AdaptiveLog(
        active_integrator=np.where(active == 1, "ias15", "leapfrog").astype(object),
        ias15_step_fraction=frac,
        raw_index=raw,
        safety_score=score,
        n_switches=nsw,
        switch_events=switch_events,
        n_steps=n_steps,
        n_ias15_steps=int(cnt.n_ias15_steps),
        n_corrections=int(cnt.n_corrections),
        rewind_events=rewind_events,
        n_rewinds=len(rewind_events),
        n_rewound_steps=int(cnt.n_rewound_steps),
        resync_events=resync_events,
        n_resyncs=len(resync_events),
        drift_budget=float(drift_budget),
        resync_on_switch=bool(resync_on_switch),
    )
    return trajectory, log
