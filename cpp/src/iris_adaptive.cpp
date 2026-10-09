// iris_adaptive.cpp -- C++ port of the main loop of core/adaptive.py::run_adaptive.
//
// Ported (same semantics, same order of operations as the Python):
//   * Leapfrog (velocity Verlet) <-> IAS15 switching with release hysteresis and hold steps
//   * rewinding (checkpoint ring, rewind_back, rewind_to_anchor, forced IAS15 replay, anchor bookkeeping)
//   * drift budget, resync-on-switch, Richardson levels 2/3, Kahan-compensated sums
//   * optional snapping to a reference trajectory (validate_adaptive.py --correct)
//   * safety index: fitted formula, smoke stand-in, and the stateful situations index
// NOT ported (validation-only analysis that needs an oracle; the Python wrapper falls back to Python for them):
//   detect_false_positives, analyze_rollback.
//
// Where the speed comes from (see README.md for measurements):
//   * no interpreter, no ctypes, no allocation in the step loop; everything is stack / preallocated
//   * compile-time body count (templates for N = 2..5): force loops fully unrolled, state in registers/L1
//   * pairwise-symmetric forces (N(N-1)/2 sqrt+div); the nearest-neighbour search of the safety features is
//     a by-product of the same pass (no second O(N^2) sweep)
//   * safety score compared in RAW-INDEX space against a precomputed threshold: no sigmoid, no exp, per step
//   * IAS15: ONE persistent REBOUND simulation (no create/free per switch), stepped through its step callback
//     with a bit-exact replica of exact_finish_time (no signal()/gettimeofday/heartbeat per dt)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

#ifdef _MSC_VER
#define restrict __restrict
#else
#define restrict __restrict__
#endif
extern "C" {
#include "rebound.h"
extern REB_API const char* reb_version_str;
}

#include "iris_adaptive.h"
#include "situations.hpp"

namespace iris {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kInvLn10 = 0.43429448190325182765;

// ---------------------------------------------------------------------------------------------
// raw index -> 0..1 score (core/safety_index.py::normalize_safety_index) and its inverse threshold
// ---------------------------------------------------------------------------------------------
struct ScoreMap {
    int mode = IRIS_SCALE_SIGMOID;
    double lo = 0, hi = 1;
    double operator()(double z) const {
        if (!std::isfinite(z)) return 1.0;  // fail-safe: non-finite -> unsafe
        double out;
        if (mode == IRIS_SCALE_SIGMOID) {
            const double zc = std::min(std::max(z, -700.0), 700.0);
            out = zc >= 0 ? 1.0 / (1.0 + std::exp(-zc)) : std::exp(zc) / (1.0 + std::exp(zc));
        } else if (mode == IRIS_SCALE_MINMAX) {
            out = std::min(std::max((z - lo) / (hi - lo), 0.0), 1.0);
        } else {
            if (z > 0) {
                const double l = std::log10(lo), h = std::log10(hi);
                out = std::min(std::max((std::log10(z) - l) / (h - l), 0.0), 1.0);
            } else {
                out = 0.0;
            }
        }
        return out;
    }
    // Smallest raw value z with map(z) >= thr (the map is non-decreasing). Lets the hot loop compare in raw space.
    double threshold_in_raw(double thr) const {
        if (thr <= 0.0) return -kInf;
        if ((*this)(1e300) < thr) return kInf;
        if ((*this)(-1e300) >= thr) return -kInf;
        double lo_ = -1e300, hi_ = 1e300;
        for (int it = 0; it < 6000; ++it) {
            const double mid = 0.5 * lo_ + 0.5 * hi_;
            if (mid == lo_ || mid == hi_) break;
            if ((*this)(mid) >= thr) hi_ = mid; else lo_ = mid;
        }
        return hi_;
    }
};

// ---------------------------------------------------------------------------------------------
// IAS15 driver: one persistent REBOUND simulation, stepped directly
// ---------------------------------------------------------------------------------------------
class Ias15 {
public:
    reb_simulation* r = nullptr;
    int64_t substeps = 0, resets = 0;
    double dt_guess = 1e-3;

    Ias15() = default;
    Ias15(const Ias15&) = delete;
    Ias15& operator=(const Ias15&) = delete;
    ~Ias15() { if (r) reb_simulation_free(r); }

    void init(int n, const double* m, double g, double softening, double dtg) {
        r = reb_simulation_create();
        r->G = g;
        r->softening = softening;
        r->exact_finish_time = 1;
        dt_guess = dtg;
        for (int i = 0; i < n; ++i) {
            reb_particle p = {};
            p.m = m[i];
            reb_simulation_add(r, p);
        }
        r->dt = dt_guess;
    }

    // Make the simulation indistinguishable from a freshly built one holding exactly this state at time t:
    // particle state, time, step-size guess, and every piece of IAS15 memory (b/e predictors and their backups,
    // compensated-summation carries). Allocation-free.
    void reset(const double* pos, const double* vel, double t) {
        const size_t n = r->N;
        for (size_t i = 0; i < n; ++i) {
            reb_particle& p = r->particles[i];
            p.x = pos[3 * i]; p.y = pos[3 * i + 1]; p.z = pos[3 * i + 2];
            p.vx = vel[3 * i]; p.vy = vel[3 * i + 1]; p.vz = vel[3 * i + 2];
        }
        auto* s = static_cast<reb_integrator_ias15_state*>(r->integrator.state);
        if (s && s->N_allocated) {
            const size_t n3 = s->N_allocated;
            std::memset(s->b, 0, 7 * n3 * sizeof(double));
            std::memset(s->e, 0, 7 * n3 * sizeof(double));
            std::memset(s->br, 0, 7 * n3 * sizeof(double));
            std::memset(s->er, 0, 7 * n3 * sizeof(double));
            std::memset(s->csx, 0, n3 * sizeof(double));
            std::memset(s->csv, 0, n3 * sizeof(double));
        }
        r->t = t;
        r->dt = dt_guess;
        r->dt_last_done = 0.0;
        ++resets;
    }

    // Replica of reb_simulation_integrate(r, tmax) with exact_finish_time = 1 (simulation.c: reb_check_exit /
    // reb_simulation_integrate_raw), minus the per-call signal(), gettimeofday(), heartbeat and status machinery.
    void advance_to(double tmax) {
        double last_full_dt = r->dt;
        r->dt_last_done = 0.0;
        bool last = false;
        for (;;) {
            if ((r->t + r->dt) >= tmax) {
                if (r->t == tmax) break;
                if (last) {
                    double tscale = 1e-12 * std::fabs(tmax);
                    if (tscale < 1e-200) tscale = 1e-12;
                    if (std::fabs(r->t - tmax) < tscale) break;
                    r->dt = tmax - r->t;
                } else {
                    last = true;
                    if (r->dt_last_done != 0.0) last_full_dt = r->dt_last_done;
                    r->dt = tmax - r->t;
                }
            } else if (last) {
                last = false;
            }
            r->integrator.callbacks.step(r, r->integrator.state);
            ++substeps;
        }
        r->dt = last_full_dt;
    }

    void read(double* pos, double* vel) const {
        const size_t n = r->N;
        for (size_t i = 0; i < n; ++i) {
            const reb_particle& p = r->particles[i];
            pos[3 * i] = p.x; pos[3 * i + 1] = p.y; pos[3 * i + 2] = p.z;
            vel[3 * i] = p.vx; vel[3 * i + 1] = p.vy; vel[3 * i + 2] = p.vz;
        }
    }
};

// ---------------------------------------------------------------------------------------------
// Result handle
// ---------------------------------------------------------------------------------------------
struct Result {
    iris_counts counts{};
    std::vector<iris_switch_event> switches;
    std::vector<iris_rewind_event> rewinds;
    std::vector<iris_resync_event> resyncs;
};

// ---------------------------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------------------------
template <int NT>
class Engine {
public:
    static constexpr int CAP = NT ? NT : IRIS_MAX_BODIES;
    struct Arr { double a[CAP][3]; };
    struct Shadow { Arr pos, vel, acc, cp, cv; };
    struct Dsh {  // the dt/2 (and dt/4) shadow Leapfrog runs; `valid` == "dshadow is not None" in Python
        bool valid = false;
        Shadow s[2];
    };
    struct Ck {   // core.adaptive._Checkpoint (minus the fields only analysis/false-positive detection use)
        int64_t step = 0;
        Arr pos, vel, acc, accp, cx, cv;
        bool has_accp = false;
        double z = 0, rawfin = kNaN;
        int64_t n_switches = 0, n_events = 0, n_ias = 0, ias_interval = 0, n_corr = 0, n_samples = 0, n_resyncs = 0;
        SitState idx;
        Dsh drift;
    };

    explicit Engine(const iris_config& cfg) : c(cfg), n(NT ? NT : cfg.n) {}

    int run(Result& res, std::string& err);
    int run_leapfrog(const iris_leapfrog_config& L, int64_t* n_corr_out, std::string& err);

private:
    const iris_config& c;
    const int n;

    // ---- constants of the run
    double dt = 0, g = 1, eps2 = 0;
    double gm[CAP], mass[CAP];
    bool comp = false;
    int levels = 2;
    double kv = 0, kj = 0, ka = 0, kd = 0, c0 = 0, mterm[CAP];
    double inv_dt = 0;

    // ---- live state
    Arr pos, vel, acc, accp, cx, cv;
    bool has_accp = false;
    double nnr2[CAP];
    int nnidx[CAP];
    Dsh dsh;
    double z = 0, rawfin = kNaN;  // worst-body raw index of the CURRENT state (decision value / finite max for logging)

    SituationIndex sit;
    Ias15 ias;
    ScoreMap smap;

    // ---------------------------------------------------------------- numerics
    template <bool NN>
    inline void forces(const Arr& p, Arr& out) {
        for (int i = 0; i < n; ++i) { out.a[i][0] = out.a[i][1] = out.a[i][2] = 0.0; }
        if (NN) for (int i = 0; i < n; ++i) { nnr2[i] = kInf; nnidx[i] = -1; }
        for (int a = 0; a < n; ++a) {
            for (int b = a + 1; b < n; ++b) {
                const double dx = p.a[b][0] - p.a[a][0], dy = p.a[b][1] - p.a[a][1], dz = p.a[b][2] - p.a[a][2];
                const double r2 = dx * dx + dy * dy + dz * dz;
                const double s = r2 + eps2;
                const double inv3 = 1.0 / (s * std::sqrt(s));
                const double fa = gm[b] * inv3, fb = gm[a] * inv3;
                out.a[a][0] += fa * dx; out.a[a][1] += fa * dy; out.a[a][2] += fa * dz;
                out.a[b][0] -= fb * dx; out.a[b][1] -= fb * dy; out.a[b][2] -= fb * dz;
                if (NN) {
                    if (r2 < nnr2[a]) { nnr2[a] = r2; nnidx[a] = b; }
                    if (r2 < nnr2[b]) { nnr2[b] = r2; nnidx[b] = a; }
                }
            }
        }
    }

    // One velocity-Verlet step of length h (core/jit_kernels.py::vv_step_kernel), in place; optionally Kahan.
    template <bool NN>
    inline void vv(Arr& p, Arr& v, Arr& a, Arr& cp, Arr& cvv, double h) {
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k) {
                const double inc = v.a[i][k] * h + ((0.5 * a.a[i][k]) * h) * h;
                if (comp) {
                    const double y = inc - cp.a[i][k];
                    const double t = p.a[i][k] + y;
                    cp.a[i][k] = (t - p.a[i][k]) - y;
                    p.a[i][k] = t;
                } else {
                    p.a[i][k] += inc;
                }
            }
        Arr na;
        forces<NN>(p, na);
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k) {
                const double inc = (0.5 * (a.a[i][k] + na.a[i][k])) * h;
                if (comp) {
                    const double y = inc - cvv.a[i][k];
                    const double t = v.a[i][k] + y;
                    cvv.a[i][k] = (t - v.a[i][k]) - y;
                    v.a[i][k] = t;
                } else {
                    v.a[i][k] += inc;
                }
            }
        a = na;
    }

    static inline void zero(Arr& x) { std::memset(&x, 0, sizeof(Arr)); }

    double pos_err(const Arr& a, const Arr& b) const {  // core.metrics.position_error
        double s = 0.0;
        for (int i = 0; i < n; ++i) {
            double q = 0.0;
            for (int k = 0; k < 3; ++k) { const double d = a.a[i][k] - b.a[i][k]; q += d * d; }
            s += q;
        }
        return std::sqrt(s / n);
    }

    void new_shadow(Dsh& d) {
        d.valid = true;
        d.s[0].pos = pos; d.s[0].vel = vel; d.s[0].acc = acc; d.s[0].cp = cx; d.s[0].cv = cv;
        if (levels >= 3) d.s[1] = d.s[0];
    }

    void advance_shadow(Dsh& d) {
        Shadow& f = d.s[0];
        for (int q = 0; q < 2; ++q) vv<false>(f.pos, f.vel, f.acc, f.cp, f.cv, 0.5 * dt);
        if (levels >= 3) {
            Shadow& qs = d.s[1];
            for (int q = 0; q < 4; ++q) vv<false>(qs.pos, qs.vel, qs.acc, qs.cp, qs.cv, 0.25 * dt);
        }
    }

    // core.adaptive._repair_one for one array
    inline void repair_one(const Arr& x, const Arr& cxx, const Arr& f, const Arr& cf, const Arr& q, const Arr& cq,
                           Arr& out, Arr& cout) const {
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k) {
                const double d_fc = (f.a[i][k] - x.a[i][k]) - (cf.a[i][k] - cxx.a[i][k]);
                double base, cb, corr;
                if (levels < 3) { base = f.a[i][k]; cb = cf.a[i][k]; corr = d_fc / 3.0; }
                else {
                    const double d_qf = (q.a[i][k] - f.a[i][k]) - (cq.a[i][k] - cf.a[i][k]);
                    base = q.a[i][k]; cb = cq.a[i][k]; corr = (19.0 * d_qf - d_fc) / 45.0;
                }
                const double t = base + corr;
                out.a[i][k] = t;
                cout.a[i][k] = ((t - base) - corr) + cb;
            }
    }

    // core.adaptive._repair: returns the estimated error; writes the repaired state.
    double repair(const Dsh& d, Arr& npos, Arr& nvel, Arr& ncx, Arr& ncv) const {
        const Shadow& f = d.s[0];
        if (!comp && levels < 3) {
            const double est = (4.0 / 3.0) * pos_err(pos, f.pos);
            for (int i = 0; i < n; ++i)
                for (int k = 0; k < 3; ++k) {
                    npos.a[i][k] = (4.0 * f.pos.a[i][k] - pos.a[i][k]) / 3.0;
                    nvel.a[i][k] = (4.0 * f.vel.a[i][k] - vel.a[i][k]) / 3.0;
                }
            ncx = cx; ncv = cv;
            return est;
        }
        const Shadow& q = levels >= 3 ? d.s[1] : d.s[0];
        repair_one(pos, cx, f.pos, f.cp, q.pos, q.cp, npos, ncx);
        repair_one(vel, cv, f.vel, f.cv, q.vel, q.cv, nvel, ncv);
        return pos_err(pos, npos);
    }

    // ---------------------------------------------------------------- safety index of the CURRENT state
    inline void eval_index(int64_t step) {
        if (c.index_kind == IRIS_INDEX_SITUATIONS) {
            sit.maybe_update(step, &pos.a[0][0], &vel.a[0][0], &acc.a[0][0]);
            z = sit.zmax;
            rawfin = sit.raw_finite_max;
            return;
        }
        double zm = -kInf, fm = kNaN;
        const bool formula = c.index_kind == IRIS_INDEX_FORMULA;
        for (int i = 0; i < n; ++i) {
            const int j = nnidx[i];
            const double wx = vel.a[i][0] - vel.a[j][0], wy = vel.a[i][1] - vel.a[j][1], wz = vel.a[i][2] - vel.a[j][2];
            const double v2 = wx * wx + wy * wy + wz * wz;
            double raw;
            if (formula) {
                const double ax = acc.a[i][0], ay = acc.a[i][1], az = acc.a[i][2];
                const double a2 = ax * ax + ay * ay + az * az;
                const double jx = (ax - accp.a[i][0]) * inv_dt, jy = (ay - accp.a[i][1]) * inv_dt, jz = (az - accp.a[i][2]) * inv_dt;
                const double j2 = jx * jx + jy * jy + jz * jz;
                // log10(max(x, 1e-15)) of a magnitude == 0.5*log10(max(x^2, 1e-30)); NaN propagates through std::max(a, b)
                raw = c0 + kv * std::log(std::max(v2, 1e-30)) + kj * std::log(std::max(j2, 1e-30)) +
                      ka * std::log(std::max(a2, 1e-30)) + kd * std::log(std::max(nnr2[i], 1e-30)) + mterm[i];
            } else {
                raw = 0.5 * kInvLn10 * std::log(std::max(v2, 1e-30));  // smoke_test_raw_index
            }
            if (!std::isfinite(raw)) { zm = kInf; continue; }
            if (raw > zm) zm = raw;
            if (std::isnan(fm) || raw > fm) fm = raw;
        }
        z = zm;
        rawfin = fm;
    }

    // ---------------------------------------------------------------- checkpoints
    std::vector<Ck> ring;
    int rhead = 0, rsize = 0;
    Ck anchor;
    inline Ck& ck_at(int k) { return ring[(rhead + k) % int(ring.size())]; }
    inline Ck& ck_new() {
        const int cap = int(ring.size());
        if (rsize < cap) { return ring[(rhead + rsize++) % cap]; }
        Ck& s = ring[rhead];
        rhead = (rhead + 1) % cap;
        return s;
    }
    // bookkeeping counters shared by fill_ck and the loop
    int64_t n_switches = 0, n_ias = 0, ias_interval = 0, n_corr = 0, ns = 0;
    Result* R = nullptr;

    void fill_ck(Ck& k, int64_t step_now) {
        k.step = step_now;
        k.pos = pos; k.vel = vel; k.acc = acc; k.accp = accp; k.cx = cx; k.cv = cv;
        k.has_accp = has_accp;
        k.z = z; k.rawfin = rawfin;
        k.n_switches = n_switches; k.n_events = int64_t(R->switches.size()); k.n_ias = n_ias;
        k.ias_interval = ias_interval; k.n_corr = n_corr; k.n_samples = ns; k.n_resyncs = int64_t(R->resyncs.size());
        if (c.index_kind == IRIS_INDEX_SITUATIONS) sit.snapshot(k.idx);
        k.drift.valid = dsh.valid;
        if (dsh.valid) {
            k.drift.s[0] = dsh.s[0];
            if (levels >= 3) k.drift.s[1] = dsh.s[1];
        }
    }

    void push_switch(int64_t step_, double t_, bool to_ias, double zv) {
        iris_switch_event e{};
        e.step = step_; e.time = t_; e.score = smap(zv); e.to_ias15 = to_ias ? 1 : 0;
        R->switches.push_back(e);
    }
};

template <int NT>
int Engine<NT>::run_leapfrog(const iris_leapfrog_config& L, int64_t* n_corr_out, std::string& err) {
    dt = L.dt; g = L.g; eps2 = L.softening * L.softening;
    comp = false;
    if (n < 2 || n > IRIS_MAX_BODIES) { err = "n must be in [2, IRIS_MAX_BODIES]"; return 1; }
    if (L.steps_per_sample < 1 || L.n_steps < 1 || !(dt > 0)) { err = "bad step counts"; return 1; }
    const int64_t sps = L.steps_per_sample;
    const int64_t T = L.n_steps / sps + 1;
    if (L.out_capacity < T) { err = "output capacity too small"; return 1; }
    const bool correcting = L.ref_pos != nullptr;
    if (correcting && (L.ref_samples < T || !L.ref_vel || !L.out_pre_pos || !L.out_pre_vel || !L.out_pre_acc ||
                       !L.out_pre_jerk || !L.out_corr_sample || !L.out_corr_step || !L.out_corr_error)) {
        err = "correction needs ref_samples >= T and all pre_check/correction output arrays";
        return 1;
    }
    for (int i = 0; i < n; ++i) gm[i] = g * L.masses[i];
    zero(pos); zero(vel); zero(acc); zero(cx); zero(cv);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) { pos.a[i][k] = L.pos0[3 * i + k]; vel.a[i][k] = L.vel0[3 * i + k]; }
    forces<false>(pos, acc);

    const size_t sz = size_t(n) * 3;
    auto put = [&](double* base, int64_t k, const Arr& a) { std::memcpy(base + size_t(k) * sz, &a.a[0][0], sz * sizeof(double)); };
    L.out_times[0] = 0.0;
    put(L.out_pos, 0, pos); put(L.out_vel, 0, vel);
    if (correcting) {
        put(L.out_pre_pos, 0, pos); put(L.out_pre_vel, 0, vel); put(L.out_pre_acc, 0, acc);
        for (size_t q = 0; q < sz; ++q) L.out_pre_jerk[q] = kNaN;
    }
    int64_t n_corr = 0, sample_idx = 0;

    if (!correcting) {  // run_leapfrog: whole samples only, exactly like the Python/Numba path
        const int64_t n_samples = L.n_steps / sps;
        for (int64_t s = 1; s <= n_samples; ++s) {
            for (int64_t q = 0; q < sps; ++q) vv<false>(pos, vel, acc, cx, cv, dt);
            L.out_times[s] = double(s * sps) * dt;
            put(L.out_pos, s, pos); put(L.out_vel, s, vel);
        }
        *n_corr_out = 0;
        return 0;
    }

    // run_leapfrog_with_correction: the last step of each sample block is taken separately so that a(t) entering it
    // is available for the pre-check jerk, (a(t+dt) - a(t)) / dt.
    int64_t step = 0;
    Arr acc_before;
    while (step < L.n_steps) {
        const int64_t block_end = std::min((step / sps + 1) * sps, L.n_steps);
        for (; step < block_end - 1; ++step) vv<false>(pos, vel, acc, cx, cv, dt);
        acc_before = acc;
        vv<false>(pos, vel, acc, cx, cv, dt);
        step = block_end;
        if (step % sps != 0) continue;
        ++sample_idx;
        Arr rp;
        std::memcpy(&rp.a[0][0], L.ref_pos + size_t(sample_idx) * sz, sz * sizeof(double));
        const double error = pos_err(pos, rp);
        put(L.out_pre_pos, sample_idx, pos); put(L.out_pre_vel, sample_idx, vel); put(L.out_pre_acc, sample_idx, acc);
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k) L.out_pre_jerk[size_t(sample_idx) * sz + 3 * i + k] = (acc.a[i][k] - acc_before.a[i][k]) / dt;
        if (error > L.error_threshold) {
            L.out_corr_sample[n_corr] = sample_idx;
            L.out_corr_step[n_corr] = step;
            L.out_corr_error[n_corr] = error;
            ++n_corr;
            pos = rp;
            std::memcpy(&vel.a[0][0], L.ref_vel + size_t(sample_idx) * sz, sz * sizeof(double));
            forces<false>(pos, acc);
        }
        L.out_times[sample_idx] = double(step) * dt;
        put(L.out_pos, sample_idx, pos); put(L.out_vel, sample_idx, vel);
    }
    *n_corr_out = n_corr;
    return 0;
}

template <int NT>
int Engine<NT>::run(Result& res, std::string& err) {
    R = &res;
    dt = c.dt; g = c.g; eps2 = c.softening * c.softening;
    inv_dt = 1.0 / dt;
    comp = c.compensated_sums != 0;
    levels = c.richardson_levels;
    smap.mode = c.scale_mode; smap.lo = c.scale_lo; smap.hi = c.scale_hi;

    if (n < 2 || n > IRIS_MAX_BODIES) { err = "n must be in [2, IRIS_MAX_BODIES]"; return 1; }
    if (c.steps_per_sample < 1 || c.n_steps < 1) { err = "bad step counts"; return 1; }
    const int64_t T = c.n_steps / c.steps_per_sample + 1;
    if (c.out_capacity < T) { err = "output capacity too small"; return 1; }
    const int64_t sps = c.steps_per_sample;
    const int64_t n_steps = c.n_steps;

    double msum_ratio_max = c.masses[0], msum_ratio_min = c.masses[0];
    for (int i = 0; i < n; ++i) {
        mass[i] = c.masses[i];
        gm[i] = g * mass[i];
        msum_ratio_max = std::max(msum_ratio_max, mass[i]);
        msum_ratio_min = std::min(msum_ratio_min, mass[i]);
    }
    // formula-index constants (per-body mass terms are constants of the run)
    const double mr = msum_ratio_max / msum_ratio_min;
    auto lg = [](double x) { return std::log10(std::max(x, 1e-15)); };
    c0 = c.formula_coef[0];
    kv = c.formula_coef[1] * 0.5 * kInvLn10;
    kj = c.formula_coef[2] * 0.5 * kInvLn10;
    ka = c.formula_coef[3] * 0.5 * kInvLn10;
    kd = c.formula_coef[4] * 0.5 * kInvLn10;
    for (int i = 0; i < n; ++i) mterm[i] = c.formula_coef[5] * lg(mass[i]) + c.formula_coef[6] * lg(mr);

    const double zsafe = smap.threshold_in_raw(c.safety_threshold);
    const double zrel = smap.threshold_in_raw(c.release_fraction * c.safety_threshold);

    // ---- initial state
    zero(pos); zero(vel); zero(acc); zero(accp); zero(cx); zero(cv);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) { pos.a[i][k] = c.pos0[3 * i + k]; vel.a[i][k] = c.vel0[3 * i + k]; }
    forces<false>(pos, acc);
    has_accp = false;

    const bool drift_on = c.drift_budget > 0 || c.resync_on_switch;
    const double drift_limit = c.drift_budget > 0 ? c.drift_budget * c.drift_error_threshold : kInf;
    if (drift_on) new_shadow(dsh);
    const bool correcting = c.ref_pos != nullptr;
    const bool rewind = c.rewind != 0;
    const int hold = c.ias15_hold_steps;

    if (c.index_kind == IRIS_INDEX_SITUATIONS) {
        std::vector<int> ids(c.sit_flag_id, c.sit_flag_id + c.n_sit_flags);
        std::vector<double> ws(c.sit_weight, c.sit_weight + c.n_sit_flags);
        sit.configure(n, c.masses, g, c.softening, dt, sps, n_steps, c.pos0, c.vel0, c.sit_intercept,
                      c.n_sit_flags, ids.data(), ws.data());
        z = sit.zmax; rawfin = sit.raw_finite_max;
    } else {
        z = -kInf; rawfin = kNaN;
    }

    ias.init(n, c.masses, g, c.softening, c.ias15_dt_guess > 0 ? c.ias15_dt_guess : 1e-3);

    // ---- outputs: sample 0
    const size_t sz = size_t(n) * 3;
    auto put_state = [&](double* base, int64_t k, const Arr& a) { std::memcpy(base + size_t(k) * sz, &a.a[0][0], sz * sizeof(double)); };
    c.out_times[0] = 0.0;
    put_state(c.out_pos, 0, pos); put_state(c.out_vel, 0, vel);
    if (c.out_pre_pos) put_state(c.out_pre_pos, 0, pos);
    if (c.out_pre_vel) put_state(c.out_pre_vel, 0, vel);
    c.out_raw[0] = kNaN; c.out_score[0] = kNaN; c.out_ias15_frac[0] = 0.0; c.out_active[0] = 0; c.out_nsw[0] = 0;
    ns = 1;

    if (rewind) {
        ring.resize(size_t(std::max(c.checkpoint_count, 1)));
        fill_ck(ck_new(), 0);
        fill_ck(anchor, 0);
    }

    bool ias_mode = false;
    int64_t hold_remaining = 0, force_until = 0;
    bool anchored_stretch = false, anchor_pending = false;
    int64_t n_rewound = 0;
    int64_t lf_exec = 0, ias_exec = 0;

    int64_t step = 1;
    while (step <= n_steps) {
        const double t_start = double(step - 1) * dt, t_end = double(step) * dt;

        // ===== 1. which integrator takes this step?
        bool want;
        if (step <= force_until) {
            want = true;
        } else if (!has_accp) {
            want = false;
        } else if (!ias_mode) {
            want = z >= zsafe;
            if (want && rewind) {
                // ---- rewind: restore an earlier checkpoint and replay with IAS15 (core.adaptive, "Rewinding")
                Ck* target;
                int depth, tidx = 0;
                if (c.rewind_to_anchor) { target = &anchor; depth = -1; }
                else {
                    tidx = std::max(rsize - 1 - c.rewind_back, 0);
                    target = &ck_at(tidx);
                    depth = rsize - 1 - tidx;
                }
                const double detect_z = z;
                const int64_t tstep = target->step;
                ns = target->n_samples;
                R->switches.resize(size_t(target->n_events));
                if (c.rewind_to_anchor) { rhead = 0; rsize = 0; Ck& s = ck_new(); s = anchor; }
                else { rsize = tidx + 1; }  // newer checkpoints belong to the discarded timeline
                n_rewound += (step - 1) - tstep;
                {
                    iris_rewind_event e{};
                    e.step = step; e.to_step = tstep; e.time = t_start; e.to_time = double(tstep) * dt;
                    e.score = smap(detect_z); e.depth = depth;
                    R->rewinds.push_back(e);
                }
                pos = target->pos; vel = target->vel; acc = target->acc; cx = target->cx; cv = target->cv;
                accp = target->accp; has_accp = target->has_accp;
                z = target->z; rawfin = target->rawfin;
                n_switches = target->n_switches; n_ias = target->n_ias; ias_interval = target->ias_interval;
                n_corr = target->n_corr;
                R->resyncs.resize(size_t(target->n_resyncs));
                const bool had_drift = target->drift.valid;
                Dsh* tdrift = &target->drift;
                dsh.valid = false;
                anchored_stretch = c.rewind_to_anchor != 0;
                if (c.index_kind == IRIS_INDEX_SITUATIONS) sit.restore(target->idx);
                if (c.resync_on_switch && had_drift) {
                    // repair the restore state before IAS15 freezes its Leapfrog error in it
                    Arr npos, nvel, ncx, ncv;
                    const double est = repair(*tdrift, npos, nvel, ncx, ncv);
                    pos = npos; vel = nvel; cx = ncx; cv = ncv;
                    forces<false>(pos, acc);
                    iris_resync_event e{};
                    e.step = tstep; e.time = double(tstep) * dt; e.estimated_error = est; e.kind = 1;
                    R->resyncs.push_back(e);
                    fill_ck(anchor, tstep);  // the repaired state is the new verified anchor
                }
                ias.reset(&pos.a[0][0], &vel.a[0][0], double(tstep) * dt);
                ias_mode = true;
                ++n_switches;
                push_switch(tstep + 1, double(tstep) * dt, true, detect_z);
                hold_remaining = hold;
                force_until = step;
                step = tstep + 1;
                continue;
            }
        } else {
            if (z >= zrel) { hold_remaining = hold; want = true; }
            else if (hold_remaining > 0) { --hold_remaining; want = true; }
            else want = false;
        }

        if (want && !ias_mode) {
            if (c.resync_on_switch && dsh.valid) {  // plain switch (no rewinding): repair the state
                Arr npos, nvel, ncx, ncv;
                const double est = repair(dsh, npos, nvel, ncx, ncv);
                pos = npos; vel = nvel; cx = ncx; cv = ncv;
                forces<false>(pos, acc);
                iris_resync_event e{};
                e.step = step - 1; e.time = t_start; e.estimated_error = est; e.kind = 1;
                R->resyncs.push_back(e);
            }
            ias.reset(&pos.a[0][0], &vel.a[0][0], t_start);
            ias_mode = true;
            hold_remaining = hold;
            ++n_switches;
            push_switch(step, t_start, true, z);
            dsh.valid = false;
            anchored_stretch = false;
        } else if (!want && ias_mode) {
            if (rewind && anchored_stretch) fill_ck(anchor, step - 1);  // IAS15 from a verified state: still clean
            anchored_stretch = false;
            if (drift_on) new_shadow(dsh);
            ias_mode = false;
            ++n_switches;
            push_switch(step, t_start, false, z);
        }

        // ===== 2. advance one dt
        accp = acc;
        has_accp = true;
        if (!ias_mode) {
            vv<true>(pos, vel, acc, cx, cv, dt);
            if (dsh.valid) advance_shadow(dsh);
            ++lf_exec;
        } else {
            ias.advance_to(t_end);
            ias.read(&pos.a[0][0], &vel.a[0][0]);
            zero(cx); zero(cv);
            forces<true>(pos, acc);
            ++n_ias; ++ias_interval; ++ias_exec;
        }

        // ===== 3. safety features / score of the new state
        eval_index(step);

        if (c.drift_budget > 0 && dsh.valid && !ias_mode) {
            Arr npos, nvel, ncx, ncv;
            const double est = repair(dsh, npos, nvel, ncx, ncv);
            if (est >= drift_limit) {
                cx = ncx; cv = ncv; pos = npos; vel = nvel;
                forces<false>(pos, acc);
                new_shadow(dsh);
                iris_resync_event e{};
                e.step = step; e.time = t_end; e.estimated_error = est; e.kind = 0;
                R->resyncs.push_back(e);
                anchor_pending = true;
            }
        }

        if (step % sps == 0) {
            if (c.out_pre_pos) put_state(c.out_pre_pos, ns, pos);
            if (c.out_pre_vel) put_state(c.out_pre_vel, ns, vel);
            if (correcting) {
                const int64_t k = ns;
                Arr rp;
                std::memcpy(&rp.a[0][0], c.ref_pos + size_t(k) * sz, sz * sizeof(double));
                if (pos_err(pos, rp) > c.error_threshold) {
                    ++n_corr;
                    anchor_pending = true;
                    pos = rp;
                    std::memcpy(&vel.a[0][0], c.ref_vel + size_t(k) * sz, sz * sizeof(double));
                    forces<false>(pos, acc);
                    zero(cx); zero(cv);
                    if (dsh.valid) new_shadow(dsh);
                    if (ias_mode) ias.reset(&pos.a[0][0], &vel.a[0][0], t_end);
                }
            }
            c.out_times[ns] = t_end;
            put_state(c.out_pos, ns, pos);
            put_state(c.out_vel, ns, vel);
            c.out_active[ns] = ias_mode ? 1 : 0;
            c.out_ias15_frac[ns] = double(ias_interval) / double(sps);
            c.out_raw[ns] = rawfin;
            c.out_score[ns] = smap(z);
            c.out_nsw[ns] = n_switches;
            ++ns;
            ias_interval = 0;
        }

        if (rewind && anchor_pending) { fill_ck(anchor, step); anchor_pending = false; }
        if (rewind && step % c.checkpoint_interval == 0) fill_ck(ck_new(), step);
        ++step;
    }

    iris_counts& k = res.counts;
    k.n_steps = n_steps; k.n_ias15_steps = n_ias; k.n_corrections = n_corr;
    k.n_rewinds = int64_t(R->rewinds.size()); k.n_rewound_steps = n_rewound;
    k.n_resyncs = int64_t(R->resyncs.size()); k.n_switch_events = int64_t(R->switches.size());
    k.n_samples = ns; k.lf_steps_executed = lf_exec; k.ias15_steps_executed = ias_exec;
    k.ias15_substeps = ias.substeps; k.n_resets = ias.resets;
    return 0;
}

}  // namespace iris

// =================================================================================================
// C ABI
// =================================================================================================
extern "C" {

IRIS_API int iris_abi_version(void) { return IRIS_ABI_VERSION; }
IRIS_API const char* iris_rebound_version(void) { return reb_version_str; }

IRIS_API int iris_run(const iris_config* cfg, void** handle, char* err, int err_len) {
    auto fail = [&](const std::string& m) {
        if (err && err_len > 0) { std::snprintf(err, size_t(err_len), "%s", m.c_str()); }
        return 1;
    };
    if (!cfg || !handle) return fail("null argument");
    *handle = nullptr;
    if (cfg->abi_version != IRIS_ABI_VERSION) return fail("ABI version mismatch between wrapper and library");
    if (cfg->richardson_levels != 2 && cfg->richardson_levels != 3) return fail("richardson_levels must be 2 or 3");
    if (cfg->rewind && (cfg->checkpoint_interval < 1 || cfg->checkpoint_count < cfg->rewind_back + 1))
        return fail("bad checkpoint settings");
    std::string e;
    try {
        auto* res = new iris::Result();
        int rc;
        switch (cfg->n) {
            case 2: { iris::Engine<2> en(*cfg); rc = en.run(*res, e); break; }
            case 3: { iris::Engine<3> en(*cfg); rc = en.run(*res, e); break; }
            case 4: { iris::Engine<4> en(*cfg); rc = en.run(*res, e); break; }
            case 5: { iris::Engine<5> en(*cfg); rc = en.run(*res, e); break; }
            default: { iris::Engine<0> en(*cfg); rc = en.run(*res, e); break; }
        }
        if (rc != 0) { delete res; return fail(e); }
        *handle = res;
        return 0;
    } catch (const std::exception& ex) {
        return fail(std::string("exception: ") + ex.what());
    }
}

IRIS_API void iris_get_counts(void* h, iris_counts* out) { *out = static_cast<iris::Result*>(h)->counts; }

IRIS_API void iris_get_events(void* h, int kind, void* out) {
    auto* r = static_cast<iris::Result*>(h);
    if (kind == IRIS_EVENTS_SWITCH) std::memcpy(out, r->switches.data(), r->switches.size() * sizeof(iris_switch_event));
    else if (kind == IRIS_EVENTS_REWIND) std::memcpy(out, r->rewinds.data(), r->rewinds.size() * sizeof(iris_rewind_event));
    else if (kind == IRIS_EVENTS_RESYNC) std::memcpy(out, r->resyncs.data(), r->resyncs.size() * sizeof(iris_resync_event));
}

IRIS_API void iris_free(void* h) { delete static_cast<iris::Result*>(h); }

IRIS_API int iris_run_leapfrog(const iris_leapfrog_config* L, int64_t* n_corrections, char* err, int err_len) {
    auto fail = [&](const std::string& m) {
        if (err && err_len > 0) { std::snprintf(err, size_t(err_len), "%s", m.c_str()); }
        return 1;
    };
    if (!L || !n_corrections) return fail("null argument");
    if (L->abi_version != IRIS_ABI_VERSION) return fail("ABI version mismatch between wrapper and library");
    iris_config dummy{};  // Engine only needs the constants it copies in run_leapfrog
    dummy.n = L->n;
    std::string e;
    try {
        int rc;
        switch (L->n) {
            case 2: { iris::Engine<2> en(dummy); rc = en.run_leapfrog(*L, n_corrections, e); break; }
            case 3: { iris::Engine<3> en(dummy); rc = en.run_leapfrog(*L, n_corrections, e); break; }
            case 4: { iris::Engine<4> en(dummy); rc = en.run_leapfrog(*L, n_corrections, e); break; }
            case 5: { iris::Engine<5> en(dummy); rc = en.run_leapfrog(*L, n_corrections, e); break; }
            default: { iris::Engine<0> en(dummy); rc = en.run_leapfrog(*L, n_corrections, e); break; }
        }
        return rc ? fail(e) : 0;
    } catch (const std::exception& ex) {
        return fail(std::string("exception: ") + ex.what());
    }
}

}  // extern "C"
