// situations.hpp -- C++ port of core/situations.py::SituationIndex (causal situation-weight index).
//
// Evaluated once per SAMPLE (every steps_per_sample steps) and held in between, exactly like the Python
// class, so its cost is amortised over `stride` steps. Constants are those of calibration/events.py.
// Numerics follow numpy: np.percentile (linear interpolation, numpy's _lerp), np.median, population std.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "iris_adaptive.h"

namespace iris {

// calibration/events.py constants
constexpr double CLOSE_ENCOUNTER_FACTOR = 0.25;
constexpr double NEAR_COLLISION_FACTOR = 0.05;
constexpr double CLOSE_ENCOUNTER_PERCENTILE = 10.0;
constexpr double NEAR_COLLISION_PERCENTILE = 2.0;
constexpr double HIGH_VELOCITY_PERCENTILE = 90.0;
constexpr double RAPID_APPROACH_PERCENTILE = 10.0;
constexpr double RAPID_SEPARATION_PERCENTILE = 90.0;
constexpr double STRONG_ACCEL_PERCENTILE = 90.0;
constexpr double HIGH_JERK_PERCENTILE = 90.0;
constexpr double ACCEL_SPIKE_PERCENTILE = 90.0;
constexpr double HIERARCHICAL_RATIO_THRESHOLD = 0.3;
constexpr double NEAR_SYMMETRIC_CV_THRESHOLD = 0.15;
constexpr double MASS_RATIO_THRESHOLD = 5.0;
constexpr int CHAOTIC_SWITCH_COUNT_THRESHOLD = 3;
constexpr double RECENT_WINDOW_FRACTION = 0.1;
constexpr int MIN_WINDOW = 2;

// numpy.percentile(a, q) with the default 'linear' method. `a` is consumed (reordered).
inline double np_percentile(std::vector<double>& a, double q) {
    const size_t n = a.size();
    const double vi = double(n - 1) * (q / 100.0);
    const size_t lo = size_t(std::floor(vi));
    const double gamma = vi - double(lo);
    std::nth_element(a.begin(), a.begin() + lo, a.end());
    const double va = a[lo];
    if (gamma == 0.0 || lo + 1 >= n) return va;
    const double vb = *std::min_element(a.begin() + lo + 1, a.end());
    const double diff = vb - va;
    return gamma >= 0.5 ? vb - diff * (1.0 - gamma) : va + diff * gamma;
}

inline double np_median(std::vector<double>& a) {
    const size_t n = a.size();
    const size_t mid = n / 2;
    std::nth_element(a.begin(), a.begin() + mid, a.end());
    const double hi = a[mid];
    if (n % 2 == 1) return hi;
    const double lo = *std::max_element(a.begin(), a.begin() + mid);
    return (lo + hi) / 2.0;
}

struct SitThresholds {
    double close = 0, coll = 0, vhigh = 0, approach = 0, sep = 0, accel = 0, jerk = 0, spike = 0;
};

// Everything that must be restored on a rewind (the h_* histories are NOT part of it: replay rewrites them).
struct SitState {
    std::vector<double> held;
    SitThresholds th;
    bool has_th = false;
    std::vector<double> prev_pd, prev_acc, prev_am;
    int prev_min_idx = 0;
    std::vector<int64_t> last_close, last_unstable;
    std::vector<int64_t> switches;
};

class SituationIndex {
public:
    // Outputs of the latest update
    double zmax = 0.0;           // max over bodies of the raw logit (+inf if any non-finite)
    double raw_finite_max = 0.0; // max over finite raw values (NaN if none)

    void configure(int n_, const double* masses, double g_, double softening, double dt, int64_t stride_,
                   int64_t n_steps, const double* pos, const double* vel, double intercept_,
                   int n_flags, const int* flag_id, const double* flag_w) {
        n = n_;
        m.assign(masses, masses + n);
        msum = 0.0;
        for (int i = 0; i < n; ++i) msum += m[i];
        g = g_;
        stride = stride_;
        dt_s = double(stride) * dt;
        n_samples = n_steps / stride + 1;
        window = std::max<int64_t>(MIN_WINDOW, int64_t(std::nearbyint(RECENT_WINDOW_FRACTION * double(n_samples))));
        intercept = intercept_;
        order_id.assign(flag_id, flag_id + n_flags);
        order_w.assign(flag_w, flag_w + n_flags);

        iu0.clear(); iu1.clear();
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) { iu0.push_back(i); iu1.push_back(j); }
        P = int(iu0.size());
        pair_idx.assign(size_t(n) * n, 0);
        for (int p = 0; p < P; ++p) { pair_idx[iu0[p] * n + iu1[p]] = p; pair_idx[iu1[p] * n + iu0[p]] = p; }

        const size_t cap = size_t(n_samples) + 1;
        h_pd.assign(cap * P, 0.0); h_ps.assign(cap * P, 0.0); h_dd.assign(cap * P, 0.0);
        h_am.assign(cap * n, 0.0); h_jk.assign(cap * n, 0.0); h_dl.assign(cap * n, 0.0);
        h_dfc.assign(cap * n, 0.0);

        st.held.assign(n, intercept);
        st.has_th = false;
        zmax = intercept;
        raw_finite_max = intercept;

        std::vector<double> acc0(size_t(n) * 3), dist(size_t(n) * n), speed(size_t(n) * n);
        accel_basic(pos, softening, acc0.data());
        pairwise(pos, vel, dist.data(), speed.data());
        st.prev_pd.resize(P);
        std::vector<double> ps0(P);
        for (int p = 0; p < P; ++p) { st.prev_pd[p] = dist[iu0[p] * n + iu1[p]]; ps0[p] = speed[iu0[p] * n + iu1[p]]; }
        for (int p = 0; p < P; ++p) { h_pd[p] = st.prev_pd[p]; h_ps[p] = ps0[p]; }
        st.prev_acc = acc0;
        st.prev_am.resize(n);
        for (int i = 0; i < n; ++i) {
            st.prev_am[i] = std::sqrt(acc0[3 * i] * acc0[3 * i] + acc0[3 * i + 1] * acc0[3 * i + 1] + acc0[3 * i + 2] * acc0[3 * i + 2]);
            h_am[i] = st.prev_am[i];
        }
        st.prev_min_idx = int(std::min_element(st.prev_pd.begin(), st.prev_pd.end()) - st.prev_pd.begin());
        primordial.assign(P, 0);
        bound(st.prev_pd.data(), ps0.data(), primordial.data());
        dist_from_com(pos, &h_dfc[0]);
        st.last_close.assign(P, -1);
        st.last_unstable.assign(n, -1);
        st.switches.clear();
        softening_ = softening;
    }

    void snapshot(SitState& out) const { out = st; }
    void restore(const SitState& in) {
        st = in;
        refresh_outputs();
    }

    // Called with the post-step state; only does work on sample steps.
    inline void maybe_update(int64_t step, const double* pos, const double* vel, const double* acc) {
        if (step % stride == 0) update(pos, vel, acc, step / stride);
    }

private:
    int n = 0, P = 0;
    std::vector<double> m;
    double msum = 0, g = 1, dt_s = 1, intercept = 0, softening_ = 0;
    int64_t stride = 1, n_samples = 0, window = 2;
    std::vector<int> order_id;
    std::vector<double> order_w;
    std::vector<int> iu0, iu1, pair_idx;
    std::vector<double> h_pd, h_ps, h_dd, h_am, h_jk, h_dl, h_dfc;
    std::vector<uint8_t> primordial;
    SitState st;
    std::vector<double> scratch;

    void accel_basic(const double* pos, double eps, double* out) const {
        const double eps2 = eps * eps;
        for (int i = 0; i < n; ++i) {
            double sx = 0, sy = 0, sz = 0;
            for (int j = 0; j < n; ++j) {
                if (j == i) continue;
                const double dx = pos[3 * j] - pos[3 * i], dy = pos[3 * j + 1] - pos[3 * i + 1], dz = pos[3 * j + 2] - pos[3 * i + 2];
                const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                const double inv = 1.0 / std::pow(dist * dist + eps2, 1.5);
                sx += m[j] * dx * inv; sy += m[j] * dy * inv; sz += m[j] * dz * inv;
            }
            out[3 * i] = g * sx; out[3 * i + 1] = g * sy; out[3 * i + 2] = g * sz;
        }
    }

    void pairwise(const double* pos, const double* vel, double* dist, double* speed) const {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                const double dx = pos[3 * j] - pos[3 * i], dy = pos[3 * j + 1] - pos[3 * i + 1], dz = pos[3 * j + 2] - pos[3 * i + 2];
                const double wx = vel[3 * j] - vel[3 * i], wy = vel[3 * j + 1] - vel[3 * i + 1], wz = vel[3 * j + 2] - vel[3 * i + 2];
                dist[i * n + j] = std::sqrt(dx * dx + dy * dy + dz * dz);
                speed[i * n + j] = std::sqrt(wx * wx + wy * wy + wz * wz);
            }
    }

    void bound(const double* pdist, const double* pspeed, uint8_t* out) const {
        for (int p = 0; p < P; ++p) {
            const double e = 0.5 * pspeed[p] * pspeed[p] - g * (m[iu0[p]] + m[iu1[p]]) / std::max(pdist[p], 1e-300);
            out[p] = e < 0.0;
        }
    }

    void dist_from_com(const double* pos, double* out) const {
        double cx = 0, cy = 0, cz = 0;
        for (int i = 0; i < n; ++i) { cx += pos[3 * i] * m[i]; cy += pos[3 * i + 1] * m[i]; cz += pos[3 * i + 2] * m[i]; }
        cx /= msum; cy /= msum; cz /= msum;
        for (int i = 0; i < n; ++i) {
            const double dx = pos[3 * i] - cx, dy = pos[3 * i + 1] - cy, dz = pos[3 * i + 2] - cz;
            out[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    void refresh_outputs() {
        double zm = -std::numeric_limits<double>::infinity();
        double fm = std::numeric_limits<double>::quiet_NaN();
        for (int i = 0; i < n; ++i) {
            const double v = st.held[i];
            if (!std::isfinite(v)) { zm = std::numeric_limits<double>::infinity(); continue; }
            if (v > zm) zm = v;
            if (std::isnan(fm) || v > fm) fm = v;
        }
        zmax = zm;
        raw_finite_max = fm;
    }

    void refresh_thresholds(int64_t s) {
        const size_t cnt = size_t(s + 1) * P;
        std::vector<double>& a = scratch;
        a.assign(h_pd.begin(), h_pd.begin() + cnt);
        double d_ref = np_median(a);
        if (!(d_ref > 0)) d_ref = 1.0;
        a.assign(h_pd.begin(), h_pd.begin() + cnt);
        const double p_coll = np_percentile(a, NEAR_COLLISION_PERCENTILE);
        a.assign(h_pd.begin(), h_pd.begin() + cnt);
        const double p_close = np_percentile(a, CLOSE_ENCOUNTER_PERCENTILE);
        st.th.close = std::max(CLOSE_ENCOUNTER_FACTOR * d_ref, p_close);
        st.th.coll = std::max(NEAR_COLLISION_FACTOR * d_ref, p_coll);
        a.assign(h_ps.begin(), h_ps.begin() + cnt);
        st.th.vhigh = np_percentile(a, HIGH_VELOCITY_PERCENTILE);
        a.assign(h_dd.begin() + P, h_dd.begin() + cnt);
        st.th.approach = np_percentile(a, RAPID_APPROACH_PERCENTILE);
        a.assign(h_dd.begin() + P, h_dd.begin() + cnt);
        st.th.sep = np_percentile(a, RAPID_SEPARATION_PERCENTILE);
        a.assign(h_am.begin(), h_am.begin() + size_t(s + 1) * n);
        st.th.accel = np_percentile(a, STRONG_ACCEL_PERCENTILE);
        a.assign(h_jk.begin() + n, h_jk.begin() + size_t(s + 1) * n);
        st.th.jerk = np_percentile(a, HIGH_JERK_PERCENTILE);
        a.assign(h_dl.begin() + n, h_dl.begin() + size_t(s + 1) * n);
        st.th.spike = np_percentile(a, ACCEL_SPIKE_PERCENTILE);
        st.has_th = true;
    }

    void update(const double* pos, const double* vel, const double* acc, int64_t s) {
        constexpr double INF = std::numeric_limits<double>::infinity();
        const int N = n;
        double dist[IRIS_MAX_BODIES * IRIS_MAX_BODIES], speed[IRIS_MAX_BODIES * IRIS_MAX_BODIES];
        pairwise(pos, vel, dist, speed);
        constexpr int MP = IRIS_MAX_BODIES * (IRIS_MAX_BODIES - 1) / 2;
        double pdist[MP], pspeed[MP], dd[MP], am[IRIS_MAX_BODIES], jk[IRIS_MAX_BODIES], dl[IRIS_MAX_BODIES];
        for (int p = 0; p < P; ++p) { pdist[p] = dist[iu0[p] * N + iu1[p]]; pspeed[p] = speed[iu0[p] * N + iu1[p]]; }
        for (int i = 0; i < N; ++i) {
            am[i] = std::sqrt(acc[3 * i] * acc[3 * i] + acc[3 * i + 1] * acc[3 * i + 1] + acc[3 * i + 2] * acc[3 * i + 2]);
            const double ax = (acc[3 * i] - st.prev_acc[3 * i]), ay = (acc[3 * i + 1] - st.prev_acc[3 * i + 1]), az = (acc[3 * i + 2] - st.prev_acc[3 * i + 2]);
            jk[i] = std::sqrt(ax * ax + ay * ay + az * az) / dt_s;
            dl[i] = std::fabs(am[i] - st.prev_am[i]);
        }
        for (int p = 0; p < P; ++p) dd[p] = (pdist[p] - st.prev_pd[p]) / dt_s;
        for (int p = 0; p < P; ++p) { h_pd[s * P + p] = pdist[p]; h_ps[s * P + p] = pspeed[p]; h_dd[s * P + p] = dd[p]; }
        for (int i = 0; i < N; ++i) { h_am[s * N + i] = am[i]; h_jk[s * N + i] = jk[i]; h_dl[s * N + i] = dl[i]; }
        std::copy(pdist, pdist + P, st.prev_pd.begin());
        st.prev_acc.assign(acc, acc + 3 * N);
        std::copy(am, am + N, st.prev_am.begin());

        if (s < 50 || s % 50 == 0 || !st.has_th) refresh_thresholds(s);
        const SitThresholds& th = st.th;
        const bool warm = s >= 10;

        int nn[IRIS_MAX_BODIES], p_nn[IRIS_MAX_BODIES];
        double nn_dist[IRIS_MAX_BODIES];
        for (int i = 0; i < N; ++i) {
            int best = -1; double bd = INF;
            for (int j = 0; j < N; ++j) {
                if (j == i) continue;
                const double d = dist[i * N + j];
                if (best < 0 || d < bd) { bd = d; best = j; }
            }
            nn[i] = best; nn_dist[i] = bd; p_nn[i] = pair_idx[i * N + best];
        }

        constexpr int MB = IRIS_MAX_BODIES;
        uint8_t close_enc[MB], near_coll[MB], strong_acc[MB], high_jerk[MB], spike[MB], rapid_app[MB], rapid_sep[MB],
            hi_rv[MB], strong_mass[MB], ejection[MB], unstable[MB], quiescent[MB];
        uint8_t temp_capture[MB] = {0}, inv_switch[MB] = {0}, inv_hier[MB] = {0};
        for (int i = 0; i < N; ++i) {
            close_enc[i] = nn_dist[i] <= th.close;
            near_coll[i] = nn_dist[i] <= th.coll;
            strong_acc[i] = warm && am[i] >= th.accel;
            high_jerk[i] = warm && jk[i] >= th.jerk;
            spike[i] = warm && dl[i] >= th.spike;
            const int p = p_nn[i];
            rapid_app[i] = warm && dd[p] <= th.approach;
            const bool recent_close = st.last_close[p] >= 0 && (s - st.last_close[p] <= window);
            rapid_sep[i] = warm && dd[p] >= th.sep && recent_close;
            hi_rv[i] = warm && close_enc[i] && pspeed[p] >= th.vhigh;
            const double mi = m[i], mj = m[nn[i]];
            const double ratio = std::max(mi, mj) / std::max(std::min(mi, mj), 1e-300);
            strong_mass[i] = close_enc[i] && ratio > MASS_RATIO_THRESHOLD;
        }
        for (int p = 0; p < P; ++p)
            if (pdist[p] < th.close) st.last_close[p] = s;

        // ejection / escape
        double* dfc = &h_dfc[size_t(s) * N];
        dist_from_com(pos, dfc);
        const int64_t oldest = std::max<int64_t>(0, s - window);
        const double* dfc_old = &h_dfc[size_t(oldest) * N];
        double vcx = 0, vcy = 0, vcz = 0;
        for (int i = 0; i < N; ++i) { vcx += vel[3 * i] * m[i]; vcy += vel[3 * i + 1] * m[i]; vcz += vel[3 * i + 2] * m[i]; }
        vcx /= msum; vcy /= msum; vcz /= msum;
        for (int i = 0; i < N; ++i) {
            const double wx = vel[3 * i] - vcx, wy = vel[3 * i + 1] - vcy, wz = vel[3 * i + 2] - vcz;
            const double ke = 0.5 * (wx * wx + wy * wy + wz * wz);
            double pe = 0.0;
            for (int j = 0; j < N; ++j) {
                const double d2 = (j == i) ? INF : dist[i * N + j];
                pe += m[j] / std::max(d2, 1e-300);
            }
            pe *= g;
            ejection[i] = ((ke - pe) > 0.0) && (dfc[i] > dfc_old[i]);
        }

        // temporary capture
        uint8_t bnd[MP];
        bound(pdist, pspeed, bnd);
        for (int p = 0; p < P; ++p)
            if (bnd[p] && !primordial[p]) { temp_capture[iu0[p]] = 1; temp_capture[iu1[p]] = 1; }

        // closest-pair switch / chaotic scattering
        const int min_idx = int(std::min_element(pdist, pdist + P) - pdist);
        const bool sw = min_idx != st.prev_min_idx;
        if (sw) {
            for (int p : {min_idx, st.prev_min_idx}) { inv_switch[iu0[p]] = 1; inv_switch[iu1[p]] = 1; }
            st.switches.push_back(s);
        }
        st.prev_min_idx = min_idx;
        {
            size_t k = 0;
            while (k < st.switches.size() && st.switches[k] <= s - window) ++k;
            if (k) st.switches.erase(st.switches.begin(), st.switches.begin() + k);
        }
        const bool chaotic = int(st.switches.size()) >= CHAOTIC_SWITCH_COUNT_THRESHOLD;

        // hierarchical configuration
        bool hier = false;
        if (N >= 3) {
            const int i = iu0[min_idx], j = iu1[min_idx];
            double d_outer = INF;
            for (int k = 0; k < N; ++k) {
                if (k == i || k == j) continue;
                d_outer = std::min(d_outer, std::min(dist[i * N + k], dist[j * N + k]));
            }
            if (d_outer > 0 && pdist[min_idx] / d_outer < HIERARCHICAL_RATIO_THRESHOLD) { hier = true; inv_hier[i] = inv_hier[j] = 1; }
        }
        // near-symmetric configuration
        bool sym = false;
        if (N >= 3 && P > 1) {
            double mean_d = 0.0;
            for (int p = 0; p < P; ++p) mean_d += pdist[p];
            mean_d /= P;
            double var = 0.0;
            for (int p = 0; p < P; ++p) var += (pdist[p] - mean_d) * (pdist[p] - mean_d);
            var /= P;
            sym = mean_d > 0 && std::sqrt(var) / mean_d < NEAR_SYMMETRIC_CV_THRESHOLD;
        }

        for (int i = 0; i < N; ++i) {
            unstable[i] = strong_acc[i] | high_jerk[i] | spike[i] | close_enc[i] | rapid_app[i] | rapid_sep[i] | hi_rv[i] |
                          near_coll[i] | ejection[i] | temp_capture[i] | strong_mass[i] | inv_hier[i] | uint8_t(chaotic) | inv_switch[i];
            const bool recently = st.last_unstable[i] >= 0 && (s - st.last_unstable[i] <= window);
            quiescent[i] = !unstable[i] && !recently;
            if (unstable[i]) st.last_unstable[i] = s;
        }

        // raw logit per body, accumulated in the Python dict order
        for (int i = 0; i < N; ++i) {
            double raw = intercept;
            for (size_t k = 0; k < order_id.size(); ++k) {
                double f = 0.0;
                switch (order_id[k]) {
                    case IRIS_SIT_EJECTION_ESCAPE: f = ejection[i]; break;
                    case IRIS_SIT_INVOLVED_HIERARCHICAL_TIGHT_PAIR: f = inv_hier[i]; break;
                    case IRIS_SIT_NEAR_COLLISION: f = near_coll[i]; break;
                    case IRIS_SIT_HIERARCHICAL_CONFIGURATION: f = hier; break;
                    case IRIS_SIT_RAPID_APPROACH: f = rapid_app[i]; break;
                    case IRIS_SIT_STRONG_MASS_RATIO_INTERACTION: f = strong_mass[i]; break;
                    case IRIS_SIT_ACCELERATION_SPIKE: f = spike[i]; break;
                    case IRIS_SIT_INVOLVED_CLOSEST_PAIR_SWITCH: f = inv_switch[i]; break;
                    case IRIS_SIT_CLOSEST_PAIR_SWITCH: f = sw; break;
                    case IRIS_SIT_TEMPORARY_CAPTURE: f = temp_capture[i]; break;
                    case IRIS_SIT_CHAOTIC_SCATTERING: f = chaotic; break;
                    case IRIS_SIT_CLOSE_ENCOUNTER: f = close_enc[i]; break;
                    case IRIS_SIT_RAPID_SEPARATION: f = rapid_sep[i]; break;
                    case IRIS_SIT_HIGH_RELATIVE_VELOCITY_ENCOUNTER: f = hi_rv[i]; break;
                    case IRIS_SIT_QUIESCENT_REGIME: f = quiescent[i]; break;
                    case IRIS_SIT_STRONG_ACCELERATION: f = strong_acc[i]; break;
                    case IRIS_SIT_HIGH_JERK: f = high_jerk[i]; break;
                    case IRIS_SIT_NEAR_SYMMETRIC_CONFIGURATION: f = sym; break;
                    default: break;
                }
                raw += order_w[k] * f;
            }
            st.held[i] = raw;
        }
        refresh_outputs();
    }
};

}  // namespace iris
