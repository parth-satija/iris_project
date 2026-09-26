"""
calibration/events.py

Detects "interesting dynamical situations" from a full trajectory and
expresses them as PER-BODY time series, so that calibration/comparer.py
can attach them to each body's own log rows (see calibration/logger.py,
which now writes one CSV per body instead of one CSV per system).

Some situations are intrinsically about a *pair* or the *whole system*
(e.g. closest-pair switch, hierarchical configuration, near-symmetric
configuration, chaotic scattering). Those are still computed once per
timestep, but are then broadcast onto every body's row as context, PLUS
a body-specific "involved_in_*" boolean that says whether that
particular body is one of the bodies driving the situation at that
timestep. This is what lets a per-body log "contain" a system-level
situation without needing a separate per-system file.

All thresholds are derived from the trajectory itself (medians /
percentiles) wherever a sensible universal constant does not exist, so
the detector adapts to the scale of each experiment (which varies a lot
across the randomized IRIS configs) rather than relying on one fixed
number that would be wrong for most systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.physics import compute_accelerations

# ---------------------------------------------------------------------------
# Tunable detection thresholds. These are deliberately factors/percentiles
# rather than absolute physical constants, since IRIS experiments span a
# wide range of masses/radii/timescales (see calibration/generator.py).
# ---------------------------------------------------------------------------

CLOSE_ENCOUNTER_FACTOR: float = 0.25       # nearest-neighbor dist < factor * median pairwise dist
NEAR_COLLISION_FACTOR: float = 0.05        # nearest-neighbor dist < factor * median pairwise dist
CLOSE_ENCOUNTER_PERCENTILE: float = 10.0   # statistical fallback: bottom 10% of pairwise dist
NEAR_COLLISION_PERCENTILE: float = 2.0     # statistical fallback: bottom 2% of pairwise dist
HIGH_VELOCITY_PERCENTILE: float = 90.0     # relative velocity considered "large"
RAPID_APPROACH_PERCENTILE: float = 10.0    # d(dist)/dt considered "rapidly decreasing"
RAPID_SEPARATION_PERCENTILE: float = 90.0  # d(dist)/dt considered "rapidly increasing"
STRONG_ACCEL_PERCENTILE: float = 90.0      # |a_i| considered "unusually large"
HIGH_JERK_PERCENTILE: float = 90.0         # |j_i| considered "unusually large"
ACCEL_SPIKE_PERCENTILE: float = 90.0       # |a_i(t) - a_i(t-1)| considered a "spike"
HIERARCHICAL_RATIO_THRESHOLD: float = 0.3  # tight_pair_dist / dist_to_outer_body
NEAR_SYMMETRIC_CV_THRESHOLD: float = 0.15  # coefficient of variation of pairwise distances
MASS_RATIO_THRESHOLD: float = 5.0          # max(m)/min(m) considered "strong" for a pair
CHAOTIC_SWITCH_COUNT_THRESHOLD: int = 3    # closest-pair switches within window => chaotic
RECENT_WINDOW_FRACTION: float = 0.1        # fraction of total samples used as a trailing window
MIN_WINDOW: int = 2


@dataclass
class EventsResult:
    """
    Per-body and per-system situational-event time series, all aligned to
    the same `times` array used to build them.

    Per-body arrays have shape (T, N); per-system arrays have shape (T,).
    Boolean arrays use numpy bool_ dtype; id/index arrays use int (with
    -1 meaning "not applicable").
    """

    # --- per-body, shape (T, N) ---
    acceleration_magnitude: np.ndarray
    jerk_magnitude: np.ndarray
    strong_acceleration: np.ndarray
    high_jerk: np.ndarray
    acceleration_spike: np.ndarray
    nearest_neighbor_id: np.ndarray
    nearest_neighbor_distance: np.ndarray
    nearest_neighbor_relative_velocity: np.ndarray
    close_encounter: np.ndarray
    rapid_approach: np.ndarray
    rapid_separation: np.ndarray
    high_relative_velocity_encounter: np.ndarray
    near_collision: np.ndarray
    ejection_escape: np.ndarray
    temporary_capture: np.ndarray
    temporary_capture_partner_id: np.ndarray
    strong_mass_ratio_interaction: np.ndarray
    involved_in_hierarchical_tight_pair: np.ndarray
    involved_in_near_symmetric_configuration: np.ndarray
    involved_in_chaotic_scattering: np.ndarray
    involved_in_closest_pair_switch: np.ndarray
    stable: np.ndarray
    quiescent_regime: np.ndarray

    # --- per-system, shape (T,) ---
    system_min_distance: np.ndarray
    system_min_distance_pair_i: np.ndarray
    system_min_distance_pair_j: np.ndarray
    closest_pair_switch: np.ndarray
    hierarchical_configuration: np.ndarray
    hierarchical_tight_pair_i: np.ndarray
    hierarchical_tight_pair_j: np.ndarray
    hierarchical_outer_body: np.ndarray
    near_symmetric_configuration: np.ndarray
    chaotic_scattering: np.ndarray

    # thresholds actually used, kept for transparency/debugging/logging
    thresholds: dict = field(default_factory=dict)


def _rolling_any(flags_2d: np.ndarray, window: int) -> np.ndarray:
    """
    For a boolean array of shape (T, K), return a same-shape boolean array
    where result[t, k] is True if flags_2d[max(0, t-window):t, k] contains
    any True (i.e. "was this True at some point in the trailing window,
    NOT counting the current sample").

    Args:
        flags_2d: shape-(T, K) boolean array.
        window: Trailing window size in samples (>= 1).

    Returns:
        shape-(T, K) boolean array.
    """
    t, k = flags_2d.shape
    if t == 0:
        return flags_2d.copy()
    df = pd.DataFrame(flags_2d.astype(np.int8))
    # shift(1) excludes the current sample; rolling max over the window
    # after the shift gives "was True at any point strictly before t,
    # within the last `window` samples".
    rolled = df.shift(1).rolling(window=window, min_periods=1).max().fillna(0.0)
    return rolled.to_numpy().astype(bool)


def compute_body_events(
    times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    g: float,
    softening: float = 0.0,
    near_collision_distance: float | None = None,
) -> EventsResult:
    """
    Compute every per-body and per-system situational event flag across a
    full trajectory.

    Args:
        times: shape-(T,) sample times.
        positions: shape-(T, N, 3) positions.
        velocities: shape-(T, N, 3) velocities.
        masses: shape-(N,) body masses.
        g: Gravitational constant used for the trajectory.
        softening: Softening length used for the trajectory.
        near_collision_distance: Optional absolute distance below which a
            near-collision is flagged. If None, defaults to
            `NEAR_COLLISION_FACTOR * median(pairwise distance)`.

    Returns:
        An EventsResult with every per-body / per-system time series.
    """
    t_n = positions.shape[0]
    n = positions.shape[1]

    # ---- pairwise distance / relative-velocity tensors, shape (T, N, N) ----
    diff = positions[:, :, np.newaxis, :] - positions[:, np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)  # (T, N, N)
    vel_diff = velocities[:, :, np.newaxis, :] - velocities[:, np.newaxis, :, :]
    rel_speed = np.linalg.norm(vel_diff, axis=-1)  # (T, N, N)

    iu = np.triu_indices(n, k=1)
    n_pairs = iu[0].shape[0]
    pair_dist = dist[:, iu[0], iu[1]]  # (T, P)
    pair_speed = rel_speed[:, iu[0], iu[1]]  # (T, P)

    d_ref = float(np.median(pair_dist)) if pair_dist.size else 1.0
    d_ref = d_ref if d_ref > 0 else 1.0
    v_ref_high = (
        float(np.percentile(pair_speed, HIGH_VELOCITY_PERCENTILE)) if pair_speed.size else 0.0
    )

    # Close-encounter / near-collision thresholds combine a fixed factor of
    # the median separation with a statistical (percentile) fallback. Using
    # a fixed factor ALONE silently produces zero flagged samples for any
    # low/moderate-eccentricity orbit, since such orbits never get 4x closer
    # than their own median separation. Taking the max of the two is exactly
    # equivalent to "flag if EITHER condition says this is unusually close",
    # so every trajectory that has any variation in separation at all will
    # have its bottom ~10% (close encounter) / ~2% (near collision) of
    # samples flagged, even when the absolute factor never fires.
    close_encounter_threshold = max(
        CLOSE_ENCOUNTER_FACTOR * d_ref,
        float(np.percentile(pair_dist, CLOSE_ENCOUNTER_PERCENTILE)) if pair_dist.size else 0.0,
    )
    near_collision_threshold = (
        float(near_collision_distance)
        if near_collision_distance is not None
        else max(
            NEAR_COLLISION_FACTOR * d_ref,
            float(np.percentile(pair_dist, NEAR_COLLISION_PERCENTILE)) if pair_dist.size else 0.0,
        )
    )

    window = max(MIN_WINDOW, int(round(RECENT_WINDOW_FRACTION * t_n)))

    # ---- d(dist)/dt per pair, via non-uniform-safe gradient ----
    if t_n > 1:
        ddist_dt = np.gradient(pair_dist, times, axis=0)
    else:
        ddist_dt = np.zeros_like(pair_dist)

    approach_threshold = (
        float(np.percentile(ddist_dt, RAPID_APPROACH_PERCENTILE)) if ddist_dt.size else 0.0
    )
    separation_threshold = (
        float(np.percentile(ddist_dt, RAPID_SEPARATION_PERCENTILE)) if ddist_dt.size else 0.0
    )

    pair_close = pair_dist < close_encounter_threshold  # (T, P)

    # ---- accelerations / jerks, per body, shape (T, N, 3) ----
    accelerations = np.zeros_like(positions)
    for idx in range(t_n):
        accelerations[idx] = compute_accelerations(
            positions[idx], masses, g=g, softening=softening
        )
    accel_mag = np.linalg.norm(accelerations, axis=-1)  # (T, N)

    if t_n > 1:
        jerks = np.gradient(accelerations, times, axis=0)
    else:
        jerks = np.zeros_like(accelerations)
    jerk_mag = np.linalg.norm(jerks, axis=-1)  # (T, N)

    accel_delta = np.zeros_like(accel_mag)
    if t_n > 1:
        accel_delta[1:] = np.abs(accel_mag[1:] - accel_mag[:-1])

    accel_threshold = float(np.percentile(accel_mag, STRONG_ACCEL_PERCENTILE)) if accel_mag.size else 0.0
    jerk_threshold = float(np.percentile(jerk_mag, HIGH_JERK_PERCENTILE)) if jerk_mag.size else 0.0
    spike_threshold = (
        float(np.percentile(accel_delta, ACCEL_SPIKE_PERCENTILE)) if accel_delta.size else 0.0
    )

    # Percentile-derived thresholds use >= so that, when a trajectory has a
    # near-discrete/plateaued distribution (e.g. an almost-circular orbit
    # where |a_i| is nearly constant), the values sitting exactly at the
    # percentile boundary are still flagged instead of silently excluded by
    # a strict ">" comparison.
    strong_acceleration = accel_mag >= accel_threshold
    high_jerk = jerk_mag >= jerk_threshold
    acceleration_spike = accel_delta >= spike_threshold

    # ---- nearest neighbor per body ----
    dist_self_inf = dist.copy()
    for i in range(n):
        dist_self_inf[:, i, i] = np.inf
    nearest_neighbor_id = np.argmin(dist_self_inf, axis=2) if n > 1 else np.full((t_n, n), -1)
    nearest_neighbor_distance = (
        np.min(dist_self_inf, axis=2) if n > 1 else np.full((t_n, n), np.inf)
    )

    close_encounter = nearest_neighbor_distance <= close_encounter_threshold
    near_collision = nearest_neighbor_distance <= near_collision_threshold

    # map (i, j) with i<j -> pair index p, for quick lookup
    pair_index = {}
    for p in range(n_pairs):
        pair_index[(int(iu[0][p]), int(iu[1][p]))] = p

    def _pair_of(i: int, j: int) -> int:
        return pair_index[(i, j)] if i < j else pair_index[(j, i)]

    rapid_approach = np.zeros((t_n, n), dtype=bool)
    rapid_separation = np.zeros((t_n, n), dtype=bool)
    high_relative_velocity_encounter = np.zeros((t_n, n), dtype=bool)
    strong_mass_ratio_interaction = np.zeros((t_n, n), dtype=bool)
    nearest_neighbor_relative_velocity = np.zeros((t_n, n))

    recently_close_pair = _rolling_any(pair_close, window) if n_pairs > 0 else pair_close

    for i in range(n):
        if n < 2:
            break
        for t in range(t_n):
            j = int(nearest_neighbor_id[t, i])
            p = _pair_of(i, j)
            nearest_neighbor_relative_velocity[t, i] = pair_speed[t, p]
            rapid_approach[t, i] = ddist_dt[t, p] <= approach_threshold
            rapid_separation[t, i] = (
                ddist_dt[t, p] >= separation_threshold and recently_close_pair[t, p]
            )
            high_relative_velocity_encounter[t, i] = (
                close_encounter[t, i] and pair_speed[t, p] >= v_ref_high
            )
            mass_pair_ratio = max(masses[i], masses[j]) / max(min(masses[i], masses[j]), 1e-300)
            strong_mass_ratio_interaction[t, i] = (
                close_encounter[t, i] and mass_pair_ratio > MASS_RATIO_THRESHOLD
            )

    # ---- ejection / escape, per body ----
    com = np.sum(positions * masses[np.newaxis, :, np.newaxis], axis=1) / np.sum(masses)
    dist_from_com = np.linalg.norm(positions - com[:, np.newaxis, :], axis=-1)  # (T, N)
    vel_com = np.sum(velocities * masses[np.newaxis, :, np.newaxis], axis=1) / np.sum(masses)

    specific_energy = np.zeros((t_n, n))
    for i in range(n):
        v_rel_com = velocities[:, i, :] - vel_com  # (T, 3)
        ke = 0.5 * np.sum(v_rel_com**2, axis=1)
        pe = np.zeros(t_n)
        for j in range(n):
            if j == i:
                continue
            pe += g * masses[j] / np.maximum(dist[:, i, j], 1e-300)
        specific_energy[:, i] = ke - pe

    ejection_escape = np.zeros((t_n, n), dtype=bool)
    for i in range(n):
        for t in range(t_n):
            t0 = max(0, t - window)
            trend_positive = dist_from_com[t, i] > dist_from_com[t0, i]
            ejection_escape[t, i] = bool(specific_energy[t, i] > 0.0 and trend_positive)

    # ---- temporary capture, per pair then broadcast to bodies ----
    temporary_capture = np.zeros((t_n, n), dtype=bool)
    temporary_capture_partner_id = np.full((t_n, n), -1, dtype=int)
    if n_pairs > 0:
        pair_specific_energy = 0.5 * pair_speed**2 - g * (
            masses[iu[0]] + masses[iu[1]]
        )[np.newaxis, :] / np.maximum(pair_dist, 1e-300)
        pair_bound = pair_specific_energy < 0.0
        primordially_bound = pair_bound[0] if t_n > 0 else np.zeros(n_pairs, dtype=bool)
        pair_temporary_capture = pair_bound & (~primordially_bound[np.newaxis, :])
        for p in range(n_pairs):
            i, j = int(iu[0][p]), int(iu[1][p])
            mask = pair_temporary_capture[:, p]
            temporary_capture[mask, i] = True
            temporary_capture[mask, j] = True
            temporary_capture_partner_id[mask, i] = j
            temporary_capture_partner_id[mask, j] = i

    # ---- closest pair identity / switch (per system) ----
    if n_pairs > 0:
        min_pair_idx = np.argmin(pair_dist, axis=1)
        system_min_distance = pair_dist[np.arange(t_n), min_pair_idx]
        system_min_distance_pair_i = iu[0][min_pair_idx]
        system_min_distance_pair_j = iu[1][min_pair_idx]
        closest_pair_switch = np.zeros(t_n, dtype=bool)
        closest_pair_switch[1:] = min_pair_idx[1:] != min_pair_idx[:-1]
    else:
        system_min_distance = np.full(t_n, np.inf)
        system_min_distance_pair_i = np.full(t_n, -1, dtype=int)
        system_min_distance_pair_j = np.full(t_n, -1, dtype=int)
        closest_pair_switch = np.zeros(t_n, dtype=bool)

    involved_in_closest_pair_switch = np.zeros((t_n, n), dtype=bool)
    for t in range(1, t_n):
        if closest_pair_switch[t]:
            for i, j in (
                (system_min_distance_pair_i[t], system_min_distance_pair_j[t]),
                (system_min_distance_pair_i[t - 1], system_min_distance_pair_j[t - 1]),
            ):
                if i >= 0:
                    involved_in_closest_pair_switch[t, i] = True
                    involved_in_closest_pair_switch[t, j] = True

    # ---- hierarchical configuration (needs N >= 3) ----
    hierarchical_configuration = np.zeros(t_n, dtype=bool)
    hierarchical_tight_pair_i = np.full(t_n, -1, dtype=int)
    hierarchical_tight_pair_j = np.full(t_n, -1, dtype=int)
    hierarchical_outer_body = np.full(t_n, -1, dtype=int)
    involved_in_hierarchical_tight_pair = np.zeros((t_n, n), dtype=bool)

    if n >= 3 and n_pairs > 0:
        for t in range(t_n):
            p_min = int(np.argmin(pair_dist[t]))
            i, j = int(iu[0][p_min]), int(iu[1][p_min])
            tight_dist = pair_dist[t, p_min]
            others = [k for k in range(n) if k not in (i, j)]
            d_outer = min(min(dist[t, i, k], dist[t, j, k]) for k in others)
            if d_outer <= 0:
                continue
            ratio = tight_dist / d_outer
            if ratio < HIERARCHICAL_RATIO_THRESHOLD:
                hierarchical_configuration[t] = True
                hierarchical_tight_pair_i[t] = i
                hierarchical_tight_pair_j[t] = j
                outer_k = min(others, key=lambda k: min(dist[t, i, k], dist[t, j, k]))
                hierarchical_outer_body[t] = outer_k
                involved_in_hierarchical_tight_pair[t, i] = True
                involved_in_hierarchical_tight_pair[t, j] = True

    # ---- near-symmetric configuration (needs N >= 3) ----
    near_symmetric_configuration = np.zeros(t_n, dtype=bool)
    if n >= 3 and n_pairs > 1:
        mean_pair_dist = np.mean(pair_dist, axis=1)
        std_pair_dist = np.std(pair_dist, axis=1)
        cv = np.divide(
            std_pair_dist,
            mean_pair_dist,
            out=np.full(t_n, np.inf),
            where=mean_pair_dist > 0,
        )
        near_symmetric_configuration = cv < NEAR_SYMMETRIC_CV_THRESHOLD

    involved_in_near_symmetric_configuration = np.broadcast_to(
        near_symmetric_configuration[:, np.newaxis], (t_n, n)
    ).copy()

    # ---- chaotic scattering: many closest-pair switches within a window ----
    switch_count_in_window = (
        pd.Series(closest_pair_switch.astype(int))
        .rolling(window=window, min_periods=1)
        .sum()
        .to_numpy()
    )
    chaotic_scattering = switch_count_in_window >= CHAOTIC_SWITCH_COUNT_THRESHOLD
    involved_in_chaotic_scattering = np.broadcast_to(
        chaotic_scattering[:, np.newaxis], (t_n, n)
    ).copy()

    # ---- stable / quiescent regime: the deliberate complement of every
    # anomaly flag above, i.e. "nothing interesting is happening to this
    # body right now". near_symmetric_configuration is intentionally NOT
    # part of the negation: it is itself an "easy" regime rather than a
    # difficulty indicator, so being near-symmetric does not disqualify a
    # body from also being stable.
    unstable = (
        strong_acceleration
        | high_jerk
        | acceleration_spike
        | close_encounter
        | rapid_approach
        | rapid_separation
        | high_relative_velocity_encounter
        | near_collision
        | ejection_escape
        | temporary_capture
        | strong_mass_ratio_interaction
        | involved_in_hierarchical_tight_pair
        | involved_in_chaotic_scattering
        | involved_in_closest_pair_switch
    )
    stable = ~unstable

    # quiescent_regime additionally requires the trailing window to have
    # been free of every anomaly too, i.e. a SUSTAINED calm stretch rather
    # than a single non-flagged sample -- the direct counterpart to
    # chaotic_scattering (which likewise requires repeated events within a
    # window, not just one). This is the interval a hybrid scheduler could
    # safely stay on the cheap Leapfrog integrator without checking in.
    recently_unstable = _rolling_any(unstable, window)
    quiescent_regime = stable & ~recently_unstable

    thresholds = {
        "d_ref_median_pairwise_distance": d_ref,
        "close_encounter_threshold": close_encounter_threshold,
        "near_collision_threshold": near_collision_threshold,
        "high_relative_velocity_threshold": v_ref_high,
        "rapid_approach_ddist_dt_threshold": approach_threshold,
        "rapid_separation_ddist_dt_threshold": separation_threshold,
        "strong_acceleration_threshold": accel_threshold,
        "high_jerk_threshold": jerk_threshold,
        "acceleration_spike_threshold": spike_threshold,
        "hierarchical_ratio_threshold": HIERARCHICAL_RATIO_THRESHOLD,
        "near_symmetric_cv_threshold": NEAR_SYMMETRIC_CV_THRESHOLD,
        "mass_ratio_threshold": MASS_RATIO_THRESHOLD,
        "chaotic_switch_count_threshold": CHAOTIC_SWITCH_COUNT_THRESHOLD,
        "window_samples": window,
    }

    return EventsResult(
        acceleration_magnitude=accel_mag,
        jerk_magnitude=jerk_mag,
        strong_acceleration=strong_acceleration,
        high_jerk=high_jerk,
        acceleration_spike=acceleration_spike,
        nearest_neighbor_id=nearest_neighbor_id,
        nearest_neighbor_distance=nearest_neighbor_distance,
        nearest_neighbor_relative_velocity=nearest_neighbor_relative_velocity,
        close_encounter=close_encounter,
        rapid_approach=rapid_approach,
        rapid_separation=rapid_separation,
        high_relative_velocity_encounter=high_relative_velocity_encounter,
        near_collision=near_collision,
        ejection_escape=ejection_escape,
        temporary_capture=temporary_capture,
        temporary_capture_partner_id=temporary_capture_partner_id,
        strong_mass_ratio_interaction=strong_mass_ratio_interaction,
        involved_in_hierarchical_tight_pair=involved_in_hierarchical_tight_pair,
        involved_in_near_symmetric_configuration=involved_in_near_symmetric_configuration,
        involved_in_chaotic_scattering=involved_in_chaotic_scattering,
        involved_in_closest_pair_switch=involved_in_closest_pair_switch,
        stable=stable,
        quiescent_regime=quiescent_regime,
        system_min_distance=system_min_distance,
        system_min_distance_pair_i=system_min_distance_pair_i,
        system_min_distance_pair_j=system_min_distance_pair_j,
        closest_pair_switch=closest_pair_switch,
        hierarchical_configuration=hierarchical_configuration,
        hierarchical_tight_pair_i=hierarchical_tight_pair_i,
        hierarchical_tight_pair_j=hierarchical_tight_pair_j,
        hierarchical_outer_body=hierarchical_outer_body,
        near_symmetric_configuration=near_symmetric_configuration,
        chaotic_scattering=chaotic_scattering,
        thresholds=thresholds,
    )
