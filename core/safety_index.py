"""
core/safety_index.py

The pieces of the "safety index" used by the adaptive (Leapfrog <-> IAS15)
integrator in core/adaptive.py:

  1. compute_safety_features(...)  the causal, run-time-computable inputs
                                   (the same quantities the calibration
                                   logged as pre_correction_*).
  2. raw_safety_index(features)    THE FORMULA. Intentionally left as a stub
                                   for now -- paste it in once the
                                   calibration analysis has settled on one.
  3. normalize_safety_index(...)   converts the raw index to a 0-1 score
                                   so a single 0-1 `safety_threshold` can be
                                   compared against it.

Higher score == LESS safe (more likely Leapfrog is about to need correction),
so the adaptive integrator switches to IAS15 when score >= safety_threshold.

--------------------------------------------------------------------------
Converting a raw index to 0-1
--------------------------------------------------------------------------
Which conversion is right depends on what the raw index is:

  "sigmoid"     raw index is a LOGIT / log-odds, e.g. the `safety_logit`
                printed by fit_safety_models.py. score = 1 / (1 + e^-raw),
                which is the model's estimated probability of a correction.
                This is the natural choice for a fitted classifier and needs
                no extra parameters.
  "minmax"      raw index is on some arbitrary linear scale. score =
                clip((raw - lo) / (hi - lo), 0, 1). Needs `lo` and `hi`.
  "log_minmax"  raw index is a strictly positive, heavy-tailed quantity
                (e.g. a jerk*dt^3-style error estimate). score =
                clip((log10(raw) - log10(lo)) / (log10(hi) - log10(lo)), 0, 1).
                Needs 0 < lo < hi.

FAIL-SAFE: a non-finite raw index (NaN / +-inf, e.g. from a degenerate
close approach) is treated as score 1.0 (unsafe -> use IAS15), never as
"probably fine".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.metrics import mass_ratio as system_mass_ratio
from core.metrics import nearest_neighbor_per_body

INDEX_SCALE_MODES: tuple[str, ...] = ("sigmoid", "minmax", "log_minmax")

# Floor applied before log10, identical to fit_safety_models.py so a formula
# fitted there sees the same feature values here.
_LOG_FLOOR: float = 1.0e-15


@dataclass(frozen=True)
class IndexScale:
    """
    How a raw safety index is mapped onto [0, 1].

    Attributes:
        mode: One of INDEX_SCALE_MODES (see module docstring).
        lo: Lower bound (raw-index units) mapped to 0.0. Required for
            "minmax" and "log_minmax".
        hi: Upper bound (raw-index units) mapped to 1.0. Required for
            "minmax" and "log_minmax".
    """

    mode: str = "sigmoid"
    lo: float | None = None
    hi: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in INDEX_SCALE_MODES:
            raise ValueError(f"mode must be one of {INDEX_SCALE_MODES}, got {self.mode!r}")
        if self.mode in ("minmax", "log_minmax"):
            if self.lo is None or self.hi is None:
                raise ValueError(f"mode {self.mode!r} requires both lo and hi")
            if not self.hi > self.lo:
                raise ValueError(f"hi ({self.hi}) must be greater than lo ({self.lo})")
            if self.mode == "log_minmax" and not self.lo > 0:
                raise ValueError(f"log_minmax requires lo > 0, got {self.lo}")


@dataclass
class SafetyFeatures:
    """
    Per-body inputs to the safety index at ONE instant. Every field is
    computable at run time from the integrator's own state (no IAS15
    reference needed), and is defined exactly as in the calibration data:

    Attributes:
        relative_velocity: shape-(N,) speed of each body relative to its
            nearest neighbor.
        nn_distance: shape-(N,) distance to each body's nearest neighbor.
        acceleration: shape-(N,) magnitude of each body's acceleration a(t).
        jerk: shape-(N,) magnitude of (a(t) - a(t - dt)) / dt, the backward
            finite difference at the integrator's own step. NaN when no
            previous acceleration exists (the very first step).
        mass: shape-(N,) body masses.
        mass_ratio: system-level mass ratio (core.metrics.mass_ratio).
    """

    relative_velocity: np.ndarray
    nn_distance: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    mass: np.ndarray
    mass_ratio: float
    # Optional raw state, filled in by compute_safety_features. Only the stateful
    # situation index (core/situations.py) uses these.
    positions: np.ndarray | None = None
    velocities: np.ndarray | None = None
    acceleration_vector: np.ndarray | None = None
    step: int | None = None

    def logs(self) -> dict[str, np.ndarray]:
        """
        log10 versions of the features, under the same names and with the
        same floor as fit_safety_models.py, so a fitted `safety_logit`
        formula can be pasted into raw_safety_index() verbatim:

            log_rel_velocity, log_jerk, log_acceleration, log_nn_distance,
            log_mass, log_mass_ratio, log_v_over_r
        """

        def lg(x) -> np.ndarray:
            return np.log10(np.maximum(np.asarray(x, dtype=np.float64), _LOG_FLOOR))

        out = {
            "log_rel_velocity": lg(self.relative_velocity),
            "log_jerk": lg(self.jerk),
            "log_acceleration": lg(self.acceleration),
            "log_nn_distance": lg(self.nn_distance),
            "log_mass": lg(self.mass),
            "log_mass_ratio": lg(np.full_like(self.mass, self.mass_ratio, dtype=np.float64)),
        }
        out["log_v_over_r"] = out["log_rel_velocity"] - out["log_nn_distance"]
        return out


def compute_safety_features(
    positions: np.ndarray,
    velocities: np.ndarray,
    acc: np.ndarray,
    acc_prev: np.ndarray | None,
    dt: float,
    masses: np.ndarray,
    step: int | None = None,
) -> SafetyFeatures:
    """
    Compute the safety-index inputs for the CURRENT state.

    Args:
        positions: shape-(N, 3) positions at the current time.
        velocities: shape-(N, 3) velocities at the current time.
        acc: shape-(N, 3) accelerations a(t) at the current state.
        acc_prev: shape-(N, 3) accelerations one step (dt) earlier, or None
            if there is no history yet (jerk is then NaN).
        dt: The integrator step separating acc_prev and acc.
        masses: shape-(N,) body masses.

    Returns:
        A SafetyFeatures for this instant.
    """
    n = positions.shape[0]
    nn_distance, relative_velocity, _ = nearest_neighbor_per_body(positions, velocities)
    if acc_prev is None:
        jerk = np.full(n, np.nan)
    else:
        jerk = np.linalg.norm((acc - acc_prev) / dt, axis=1)
    return SafetyFeatures(
        relative_velocity=relative_velocity,
        nn_distance=nn_distance,
        acceleration=np.linalg.norm(acc, axis=1),
        jerk=jerk,
        mass=np.asarray(masses, dtype=np.float64),
        mass_ratio=float(system_mass_ratio(masses)),
        positions=positions,
        velocities=velocities,
        acceleration_vector=acc,
        step=step,
    )


# --------------------------------------------------------------------------
# THE FORMULA -- deliberately not filled in yet
# --------------------------------------------------------------------------
def raw_safety_index(features: SafetyFeatures) -> np.ndarray:
    """
    The raw (un-normalized) safety index, one value per body. HIGHER MEANS
    LESS SAFE. The system score used for switching is the max over bodies.

    This is the ridge-logistic `safety_logit` printed by
    `python fit_safety_models.py --three-body` (causal features only, C=0.001,
    fit on 3-body calibration data only). It is a LOGIT, so use the default
    --index-scale sigmoid; the 0-1 score is then 1 / (1 + exp(-safety_logit)).

    Fit quality (grouped 5-fold CV): AUC 0.744 +- 0.174, weighted AUC 0.791.
    Only the 0.80-recall threshold is usable (safety_threshold ~0.134, about
    19% of ordinary steps sent to IAS15); higher-recall thresholds land at
    logits of -15 or lower with a ~97% false-positive rate, i.e. the index does
    not separate at high recall.

    Caveat: the training set is an enriched sample (hard negatives are
    over-represented), so the sigmoid output is NOT a calibrated probability
    of needing a correction. Choose --safety-threshold from the recall /
    false-positive trade-off, not from 0.5.

    Coefficients are on log10 features with the same 1e-15 floor used in
    fit_safety_models.py (see SafetyFeatures.logs()), so they apply verbatim.

    Previous version, ridge fit on MIXED 2-5 body data (CV AUC 0.907):

        -5.263 - 4.979*log_rel_velocity + 7.484*log_jerk - 6.372*log_acceleration
        + 8.370*log_nn_distance - 0.088*log_mass - 0.273*log_mass_ratio
    """
    f = features.logs()
    return (
        +0.7838
        +0.6963 * f["log_rel_velocity"]
        +0.2168 * f["log_jerk"]
        +0.4218 * f["log_acceleration"]
        -0.2614 * f["log_nn_distance"]
        -2.0146 * f["log_mass"]
        -2.5698 * f["log_mass_ratio"]
    )


def smoke_test_raw_index(features: SafetyFeatures) -> np.ndarray:
    """
    MEANINGLESS stand-in index (log10 of relative velocity) used only by
    --smoke-test-index to exercise the adaptive pipeline end to end before a
    real formula exists. Results obtained with it say nothing about the
    method.
    """
    return features.logs()["log_rel_velocity"]


_INDEX_SOURCES: dict[str, Callable[[SafetyFeatures], np.ndarray]] = {
    "formula": raw_safety_index,
    "smoke": smoke_test_raw_index,
}


def resolve_raw_index_fn(source: str) -> Callable[[SafetyFeatures], np.ndarray]:
    """
    Map a picklable source name ("formula" or "smoke") to the raw-index
    function. Worker processes receive the NAME, not the function, so
    nothing unpicklable crosses the process boundary.
    """
    if source == "situations":
        # Stateful (keeps history), so build a FRESH instance per call: one per run.
        from core.situations import SituationIndex

        return SituationIndex()
    try:
        return _INDEX_SOURCES[source]
    except KeyError:
        raise ValueError(
            f"unknown index source {source!r}; expected one of {list(_INDEX_SOURCES) + ['situations']}"
        )


def check_index_available(raw_index_fn: Callable[[SafetyFeatures], np.ndarray]) -> tuple[bool, str]:
    """
    Cheaply test whether a raw-index function is usable, by calling it on a
    dummy 3-body feature set. Lets the entry point fail fast, once, with a
    clear message instead of once per worker.

    Returns:
        (ok, message); message is empty when ok.
    """
    self_test = getattr(raw_index_fn, "self_test", None)
    if callable(self_test):  # stateful indices can't be probed with a bare feature set
        return self_test()
    dummy = SafetyFeatures(
        relative_velocity=np.array([1.0, 2.0, 3.0]),
        nn_distance=np.array([1.0, 1.0, 1.0]),
        acceleration=np.array([1.0, 1.0, 1.0]),
        jerk=np.array([1.0, 1.0, 1.0]),
        mass=np.array([1.0, 2.0, 3.0]),
        mass_ratio=3.0,
    )
    try:
        raw_index_fn(dummy)
    except NotImplementedError as exc:
        return False, str(exc)
    return True, ""


# --------------------------------------------------------------------------
# Raw index -> 0..1
# --------------------------------------------------------------------------
def normalize_safety_index(raw: np.ndarray, scale: IndexScale) -> np.ndarray:
    """
    Map raw safety-index values onto [0, 1] (higher == less safe).

    Non-finite inputs (NaN, +-inf) map to 1.0 (fail-safe: treat as unsafe).

    Args:
        raw: Array (any shape) of raw index values.
        scale: How to convert; see IndexScale / the module docstring.

    Returns:
        Array of the same shape with values in [0, 1].
    """
    x = np.asarray(raw, dtype=np.float64)
    finite = np.isfinite(x)
    safe_x = np.where(finite, x, 0.0)

    if scale.mode == "sigmoid":
        z = np.clip(safe_x, -700.0, 700.0)
        # numerically stable logistic: never exponentiates a large positive number
        out = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    elif scale.mode == "minmax":
        out = np.clip((safe_x - scale.lo) / (scale.hi - scale.lo), 0.0, 1.0)
    else:  # log_minmax
        positive = safe_x > 0
        log_x = np.log10(np.where(positive, safe_x, 1.0))
        lo, hi = np.log10(scale.lo), np.log10(scale.hi)
        out = np.clip((log_x - lo) / (hi - lo), 0.0, 1.0)
        out = np.where(positive, out, 0.0)  # index <= 0 is as safe as it gets

    return np.where(finite, out, 1.0)


def system_safety_score(raw_per_body: np.ndarray, scale: IndexScale) -> float:
    """
    The single 0-1 score the integrator compares against safety_threshold:
    the WORST (largest) normalized per-body score, since one body going bad
    is enough to need IAS15.
    """
    scores = normalize_safety_index(raw_per_body, scale)
    return float(np.max(scores)) if scores.size else 0.0
