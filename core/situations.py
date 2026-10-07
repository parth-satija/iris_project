"""
core/situations.py

A CAUSAL, run-time version of the additive situation-weight safety index
produced by fit_situation_weights.py:

    safety_logit_i = intercept + sum( weight[s] for every situation s active for body i )

The adaptive integrator's score is the sigmoid of the WORST body's logit
(see core/safety_index.py::system_safety_score), so use --index-scale sigmoid
(the default).

--------------------------------------------------------------------------
How the flags differ from calibration/events.py
--------------------------------------------------------------------------
The weights were fitted on the flags calibration/events.py logged. That code
sees the whole trajectory; a live integrator does not. Each situation below
uses the same definition and the same constants (imported from
calibration.events), with these causal substitutions:

  * Evaluated on the SAMPLE grid (every `steps_per_sample` steps, the same
    spacing the calibration data uses), and held constant in between. So a
    score computed at sample k governs the steps until sample k+1.
  * Derivatives are BACKWARD differences over one sample interval (events.py
    uses central differences, which need the next sample).
  * Percentile / median thresholds (typical separation, "strong" acceleration,
    high jerk, ...) use the history so far instead of the whole run, and are
    refreshed every REFRESH samples. Percentile-based flags stay off for the
    first WARMUP samples, when the history is too short to mean anything.
  * The trailing window is sized as in events.py: 10% of the run's sample
    count (known from total_time / sample_interval).
  * "Temporary capture" compares against the pairs bound at sample 0.

These are approximations. The weights were NOT refitted on the causal flags,
so the score's quality is an empirical question: judge it by the error counts
of validate_adaptive.py, not by the AUC printed by fit_situation_weights.py.

Weights below are pasted from fit_situation_weights.py (ridge, C=10,
sampling_weight fit, 48 experiments). Re-paste them after refitting.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from calibration.events import (
    ACCEL_SPIKE_PERCENTILE,
    CHAOTIC_SWITCH_COUNT_THRESHOLD,
    CLOSE_ENCOUNTER_FACTOR,
    CLOSE_ENCOUNTER_PERCENTILE,
    HIERARCHICAL_RATIO_THRESHOLD,
    HIGH_JERK_PERCENTILE,
    HIGH_VELOCITY_PERCENTILE,
    MASS_RATIO_THRESHOLD,
    MIN_WINDOW,
    NEAR_COLLISION_FACTOR,
    NEAR_COLLISION_PERCENTILE,
    NEAR_SYMMETRIC_CV_THRESHOLD,
    RAPID_APPROACH_PERCENTILE,
    RAPID_SEPARATION_PERCENTILE,
    RECENT_WINDOW_FRACTION,
    STRONG_ACCEL_PERCENTILE,
)
from core.physics import compute_accelerations

SITUATION_INTERCEPT: float = -12.8307

SITUATION_WEIGHTS: dict[str, float] = {
    "ejection_escape": +6.8358,
    "involved_in_hierarchical_tight_pair": +4.6114,
    "near_collision": +4.3797,
    "hierarchical_configuration": -2.5397,
    "rapid_approach": -2.1626,
    "strong_mass_ratio_interaction": +2.1378,
    "acceleration_spike": +1.4949,
    "involved_in_closest_pair_switch": +1.0325,
    "closest_pair_switch": +0.9224,
    "temporary_capture": +0.8707,
    "chaotic_scattering": +0.4028,
    "close_encounter": +0.3190,
    "rapid_separation": -0.2541,
    "high_relative_velocity_encounter": -0.2416,
    "quiescent_regime": -0.1651,
    "strong_acceleration": -0.0496,
    "high_jerk": +0.0433,
    "near_symmetric_configuration": -0.0423,
}


class SituationIndex:
    """
    Stateful raw-index callable for core.adaptive.run_adaptive. Create ONE
    instance per run (core.safety_index.resolve_raw_index_fn("situations")
    does): run_adaptive calls configure() once, then the instance once per
    integrator step with that step's SafetyFeatures.
    """

    REFRESH = 50  # recompute percentile thresholds every this many samples
    WARMUP = 10  # samples before percentile-based flags are trusted

    def __init__(self, intercept: float = SITUATION_INTERCEPT, weights: dict | None = None) -> None:
        self.intercept = float(intercept)
        self.weights = dict(SITUATION_WEIGHTS if weights is None else weights)
        self._ready = False

    # ------------------------------------------------------------------
    def configure(
        self,
        masses: np.ndarray,
        g: float,
        softening: float,
        dt: float,
        steps_per_sample: int,
        n_steps: int,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> None:
        """Called once by run_adaptive with the run's constants and the t=0 state."""
        self.m = np.asarray(masses, dtype=np.float64)
        self.n = int(self.m.shape[0])
        self.g = float(g)
        self.stride = int(steps_per_sample)
        self.dt_s = self.stride * float(dt)
        self.n_samples = n_steps // self.stride + 1
        self.window = max(MIN_WINDOW, int(round(RECENT_WINDOW_FRACTION * self.n_samples)))

        self.iu = np.triu_indices(self.n, k=1)
        self.P = int(self.iu[0].shape[0])
        self.pair_idx = np.zeros((self.n, self.n), dtype=int)
        self.pair_idx[self.iu] = np.arange(self.P)
        self.pair_idx[(self.iu[1], self.iu[0])] = np.arange(self.P)

        cap = self.n_samples + 1
        self.h_pd = np.zeros((cap, self.P))  # pair distance
        self.h_ps = np.zeros((cap, self.P))  # pair relative speed
        self.h_dd = np.zeros((cap, self.P))  # d(pair distance)/dt
        self.h_am = np.zeros((cap, self.n))  # |acceleration|
        self.h_jk = np.zeros((cap, self.n))  # |jerk|
        self.h_dl = np.zeros((cap, self.n))  # |a(t)| - |a(t-1)| spike measure

        self.held = np.full(self.n, self.intercept)
        self.th: dict[str, float] = {}
        self._ready = True
        if self.P == 0:
            return

        acc0 = compute_accelerations(positions, self.m, g=self.g, softening=softening)
        dist0, speed0 = self._pairwise(positions, velocities)
        pd0, ps0 = dist0[self.iu], speed0[self.iu]
        self.h_pd[0], self.h_ps[0], self.h_am[0] = pd0, ps0, np.linalg.norm(acc0, axis=1)
        self.prev_pd, self.prev_acc, self.prev_am = pd0, acc0.copy(), self.h_am[0].copy()
        self.prev_min_idx = int(np.argmin(pd0))
        self.primordial = self._bound(pd0, ps0)

        self.dfc_hist: deque = deque(maxlen=self.window + 1)
        self.dfc_hist.append(self._dist_from_com(positions))
        self.last_close = np.full(self.P, -1, dtype=int)
        self.last_unstable = np.full(self.n, -1, dtype=int)
        self.switches: deque = deque()

    # ------------------------------------------------------------------
    @staticmethod
    def _pairwise(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dist = np.linalg.norm(pos[np.newaxis, :, :] - pos[:, np.newaxis, :], axis=-1)
        speed = np.linalg.norm(vel[np.newaxis, :, :] - vel[:, np.newaxis, :], axis=-1)
        return dist, speed

    def _bound(self, pdist: np.ndarray, pspeed: np.ndarray) -> np.ndarray:
        iu0, iu1 = self.iu
        energy = 0.5 * pspeed**2 - self.g * (self.m[iu0] + self.m[iu1]) / np.maximum(pdist, 1e-300)
        return energy < 0.0

    def _dist_from_com(self, pos: np.ndarray) -> np.ndarray:
        com = np.sum(pos * self.m[:, None], axis=0) / np.sum(self.m)
        return np.linalg.norm(pos - com, axis=1)

    def _refresh(self, s: int) -> None:
        """Recompute the history-based thresholds from samples 0..s."""
        pdh = self.h_pd[: s + 1]
        d_ref = float(np.median(pdh))
        d_ref = d_ref if d_ref > 0 else 1.0
        p_coll, p_close = np.percentile(pdh, [NEAR_COLLISION_PERCENTILE, CLOSE_ENCOUNTER_PERCENTILE])
        self.th = {
            "close": max(CLOSE_ENCOUNTER_FACTOR * d_ref, float(p_close)),
            "coll": max(NEAR_COLLISION_FACTOR * d_ref, float(p_coll)),
            "vhigh": float(np.percentile(self.h_ps[: s + 1], HIGH_VELOCITY_PERCENTILE)),
            "approach": float(np.percentile(self.h_dd[1 : s + 1], RAPID_APPROACH_PERCENTILE)),
            "sep": float(np.percentile(self.h_dd[1 : s + 1], RAPID_SEPARATION_PERCENTILE)),
            "accel": float(np.percentile(self.h_am[: s + 1], STRONG_ACCEL_PERCENTILE)),
            "jerk": float(np.percentile(self.h_jk[1 : s + 1], HIGH_JERK_PERCENTILE)),
            "spike": float(np.percentile(self.h_dl[1 : s + 1], ACCEL_SPIKE_PERCENTILE)),
        }

    # ------------------------------------------------------------------
    def _update(self, pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, s: int) -> None:
        """Evaluate every situation at sample s and refresh the held score."""
        n, w_win = self.n, self.window
        iu0, iu1 = self.iu
        mm = self.m

        dist, speed = self._pairwise(pos, vel)
        pdist, pspeed = dist[self.iu], speed[self.iu]
        am = np.linalg.norm(acc, axis=1)
        dd = (pdist - self.prev_pd) / self.dt_s
        jk = np.linalg.norm(acc - self.prev_acc, axis=1) / self.dt_s
        dl = np.abs(am - self.prev_am)
        self.h_pd[s], self.h_ps[s], self.h_am[s] = pdist, pspeed, am
        self.h_dd[s], self.h_jk[s], self.h_dl[s] = dd, jk, dl
        self.prev_pd, self.prev_acc, self.prev_am = pdist, acc.copy(), am

        if s < 50 or s % self.REFRESH == 0 or not self.th:
            self._refresh(s)
        th = self.th
        warm = s >= self.WARMUP

        idx = np.arange(n)
        d2 = dist.copy()
        np.fill_diagonal(d2, np.inf)
        nn = np.argmin(d2, axis=1)
        nn_dist = d2[idx, nn]
        p_nn = self.pair_idx[idx, nn]

        close_enc = nn_dist <= th["close"]
        near_coll = nn_dist <= th["coll"]
        strong_acc = warm & (am >= th["accel"])
        high_jerk = warm & (jk >= th["jerk"])
        spike = warm & (dl >= th["spike"])

        rapid_app = warm & (dd[p_nn] <= th["approach"])
        recent_close = (self.last_close[p_nn] >= 0) & (s - self.last_close[p_nn] <= w_win)
        rapid_sep = warm & (dd[p_nn] >= th["sep"]) & recent_close
        hi_rv_enc = warm & close_enc & (pspeed[p_nn] >= th["vhigh"])
        ratio = np.maximum(mm, mm[nn]) / np.maximum(np.minimum(mm, mm[nn]), 1e-300)
        strong_mass = close_enc & (ratio > MASS_RATIO_THRESHOLD)
        self.last_close[pdist < th["close"]] = s  # after use: "recent" excludes the current sample

        # ejection / escape: positive specific energy and moving outward vs `window` samples ago
        dfc = self._dist_from_com(pos)
        self.dfc_hist.append(dfc)
        vcom = np.sum(vel * mm[:, None], axis=0) / np.sum(mm)
        ke = 0.5 * np.sum((vel - vcom) ** 2, axis=1)
        pe = self.g * np.sum(mm[None, :] / np.maximum(d2, 1e-300), axis=1)
        ejection = ((ke - pe) > 0.0) & (dfc > self.dfc_hist[0])

        # temporary capture: bound now, not bound at sample 0
        captured = self._bound(pdist, pspeed) & ~self.primordial
        temp_capture = np.zeros(n, dtype=bool)
        temp_capture[iu0[captured]] = True
        temp_capture[iu1[captured]] = True

        # closest-pair switch and chaotic scattering
        min_idx = int(np.argmin(pdist))
        switch = min_idx != self.prev_min_idx
        inv_switch = np.zeros(n, dtype=bool)
        if switch:
            for p in (min_idx, self.prev_min_idx):
                inv_switch[iu0[p]] = True
                inv_switch[iu1[p]] = True
            self.switches.append(s)
        self.prev_min_idx = min_idx
        while self.switches and self.switches[0] <= s - w_win:
            self.switches.popleft()
        chaotic = len(self.switches) >= CHAOTIC_SWITCH_COUNT_THRESHOLD

        # hierarchical configuration (N >= 3)
        hier = False
        inv_hier = np.zeros(n, dtype=bool)
        if n >= 3:
            i, j = int(iu0[min_idx]), int(iu1[min_idx])
            others = [k for k in range(n) if k not in (i, j)]
            d_outer = min(min(dist[i, k], dist[j, k]) for k in others)
            if d_outer > 0 and pdist[min_idx] / d_outer < HIERARCHICAL_RATIO_THRESHOLD:
                hier = True
                inv_hier[i] = inv_hier[j] = True

        # near-symmetric configuration (N >= 3)
        sym = False
        if n >= 3 and self.P > 1:
            mean_d = float(np.mean(pdist))
            sym = mean_d > 0 and float(np.std(pdist)) / mean_d < NEAR_SYMMETRIC_CV_THRESHOLD

        unstable = (
            strong_acc | high_jerk | spike | close_enc | rapid_app | rapid_sep | hi_rv_enc
            | near_coll | ejection | temp_capture | strong_mass | inv_hier | chaotic | inv_switch
        )
        recently_unstable = (self.last_unstable >= 0) & (s - self.last_unstable <= w_win)
        quiescent = ~unstable & ~recently_unstable
        self.last_unstable[unstable] = s

        flags = {
            "strong_acceleration": strong_acc,
            "high_jerk": high_jerk,
            "acceleration_spike": spike,
            "close_encounter": close_enc,
            "rapid_approach": rapid_app,
            "rapid_separation": rapid_sep,
            "high_relative_velocity_encounter": hi_rv_enc,
            "near_collision": near_coll,
            "ejection_escape": ejection,
            "temporary_capture": temp_capture,
            "strong_mass_ratio_interaction": strong_mass,
            "closest_pair_switch": switch,
            "involved_in_closest_pair_switch": inv_switch,
            "hierarchical_configuration": hier,
            "involved_in_hierarchical_tight_pair": inv_hier,
            "near_symmetric_configuration": sym,
            "chaotic_scattering": chaotic,
            "quiescent_regime": quiescent,
        }
        raw = np.full(n, self.intercept)
        for name, weight in self.weights.items():
            raw += weight * np.asarray(flags[name], dtype=np.float64)
        self.held = raw
        self.last_flags = flags

    # ------------------------------------------------------------------
    def snapshot_state(self) -> dict:
        """
        Copy of the index's memory, for core.adaptive's rewinding. The per-sample
        history arrays (h_*) are deliberately NOT copied: entries beyond the
        restored sample are always rewritten by _update() before they are read
        (every threshold uses only the slice [: s + 1] after h[s] was written).
        """
        state: dict = {"held": self.held.copy(), "th": dict(self.th)}
        if self.P > 0:
            state.update(
                prev_pd=self.prev_pd.copy(),
                prev_acc=self.prev_acc.copy(),
                prev_am=self.prev_am.copy(),
                prev_min_idx=self.prev_min_idx,
                dfc_hist=deque(self.dfc_hist, maxlen=self.dfc_hist.maxlen),
                last_close=self.last_close.copy(),
                last_unstable=self.last_unstable.copy(),
                switches=deque(self.switches),
            )
        return state

    def restore_state(self, state: dict) -> None:
        """Inverse of snapshot_state(). Copies again, so one snapshot can be restored repeatedly."""
        self.held = state["held"].copy()
        self.th = dict(state["th"])
        if self.P > 0:
            self.prev_pd = state["prev_pd"].copy()
            self.prev_acc = state["prev_acc"].copy()
            self.prev_am = state["prev_am"].copy()
            self.prev_min_idx = state["prev_min_idx"]
            self.dfc_hist = deque(state["dfc_hist"], maxlen=state["dfc_hist"].maxlen)
            self.last_close = state["last_close"].copy()
            self.last_unstable = state["last_unstable"].copy()
            self.switches = deque(state["switches"])

    def __call__(self, features) -> np.ndarray:
        """Per-body raw index (a logit) for this integrator step."""
        if not self._ready:
            raise RuntimeError(
                "SituationIndex.configure() was not called; use it through "
                "core.adaptive.run_adaptive."
            )
        step = features.step
        if step is None or features.positions is None:
            raise ValueError("SituationIndex needs SafetyFeatures.step/positions/velocities/acceleration_vector")
        if self.P > 0 and step % self.stride == 0:
            self._update(features.positions, features.velocities, features.acceleration_vector, step // self.stride)
        return self.held.copy()

    def self_test(self) -> tuple[bool, str]:
        """Smoke-check the index on a tiny synthetic system (used by check_index_available)."""
        try:
            rng = np.random.default_rng(0)
            pos = rng.normal(size=(4, 3))
            vel = 0.1 * rng.normal(size=(4, 3))
            masses = np.array([1.0, 2.0, 3.0, 4.0])
            probe = SituationIndex(self.intercept, self.weights)
            probe.configure(masses, 1.0, 0.0, 1e-3, 2, 60, pos, vel)
            for s in range(1, 30):
                p = pos + 0.01 * s * rng.normal(size=pos.shape)
                acc = compute_accelerations(p, masses, g=1.0)
                probe._update(p, vel, acc, s)
            if not np.all(np.isfinite(probe.held)):
                return False, "SituationIndex produced non-finite values in its self-test"
        except Exception as exc:  # noqa: BLE001
            return False, f"SituationIndex self-test failed: {type(exc).__name__}: {exc}"
        return True, ""
