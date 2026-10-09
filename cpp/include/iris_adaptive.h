/*
 * iris_adaptive.h  --  C ABI of the C++ port of core/adaptive.py::run_adaptive.
 *
 * ONE call runs the whole simulation (all steps, switching, rewinding, repairs,
 * sampling). The caller (Python via ctypes, or bench.cpp) allocates the output
 * arrays; nothing crosses the language boundary per step.
 *
 * Layout conventions: all arrays are float64 C-contiguous; positions/velocities
 * are (n, 3); per-sample arrays are (T, ...) with T = n_steps / steps_per_sample + 1.
 */
#ifndef IRIS_ADAPTIVE_H
#define IRIS_ADAPTIVE_H

#include <stdint.h>

#ifdef _WIN32
#define IRIS_API __declspec(dllexport)
#else
#define IRIS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define IRIS_ABI_VERSION 2
#define IRIS_MAX_BODIES 16
#define IRIS_N_SITUATION_FLAGS 18

enum { IRIS_INDEX_FORMULA = 0, IRIS_INDEX_SMOKE = 1, IRIS_INDEX_SITUATIONS = 2 };
enum { IRIS_SCALE_SIGMOID = 0, IRIS_SCALE_MINMAX = 1, IRIS_SCALE_LOG_MINMAX = 2 };

/* Canonical situation-flag ids (order of SITUATION_WEIGHTS in core/situations.py at the time of writing).
 * The Python wrapper maps flag NAMES to these ids and passes the weights in the dict's own order, so the
 * floating-point accumulation order matches the Python index exactly. */
enum {
    IRIS_SIT_EJECTION_ESCAPE = 0,
    IRIS_SIT_INVOLVED_HIERARCHICAL_TIGHT_PAIR,
    IRIS_SIT_NEAR_COLLISION,
    IRIS_SIT_HIERARCHICAL_CONFIGURATION,
    IRIS_SIT_RAPID_APPROACH,
    IRIS_SIT_STRONG_MASS_RATIO_INTERACTION,
    IRIS_SIT_ACCELERATION_SPIKE,
    IRIS_SIT_INVOLVED_CLOSEST_PAIR_SWITCH,
    IRIS_SIT_CLOSEST_PAIR_SWITCH,
    IRIS_SIT_TEMPORARY_CAPTURE,
    IRIS_SIT_CHAOTIC_SCATTERING,
    IRIS_SIT_CLOSE_ENCOUNTER,
    IRIS_SIT_RAPID_SEPARATION,
    IRIS_SIT_HIGH_RELATIVE_VELOCITY_ENCOUNTER,
    IRIS_SIT_QUIESCENT_REGIME,
    IRIS_SIT_STRONG_ACCELERATION,
    IRIS_SIT_HIGH_JERK,
    IRIS_SIT_NEAR_SYMMETRIC_CONFIGURATION
};

typedef struct iris_config {
    int32_t abi_version;  /* must be IRIS_ABI_VERSION */
    int32_t n;            /* number of bodies, 2 .. IRIS_MAX_BODIES */
    double g, softening, dt;
    int64_t n_steps;
    int64_t steps_per_sample;
    const double *masses; /* (n)   */
    const double *pos0;   /* (n,3) */
    const double *vel0;   /* (n,3) */

    /* --- safety index -> 0..1 score --- */
    int32_t index_kind;   /* IRIS_INDEX_* */
    int32_t scale_mode;   /* IRIS_SCALE_* */
    double scale_lo, scale_hi;
    double safety_threshold;
    double release_fraction;
    /* formula index: raw = c[0] + c[1] lg(relv) + c[2] lg(jerk) + c[3] lg(acc) + c[4] lg(nn_dist)
     *                         + c[5] lg(mass) + c[6] lg(mass_ratio)        (lg = log10, floor 1e-15) */
    double formula_coef[7];
    /* situations index */
    double sit_intercept;
    int32_t n_sit_flags;
    int32_t sit_flag_id[IRIS_N_SITUATION_FLAGS];
    double sit_weight[IRIS_N_SITUATION_FLAGS];

    /* --- optional snapping to a reference (validation --correct); ref_pos NULL = off --- */
    const double *ref_pos;  /* (ref_samples, n, 3) */
    const double *ref_vel;
    int64_t ref_samples;
    double error_threshold;

    /* --- rewinding --- */
    int32_t rewind;
    int32_t checkpoint_count;
    int32_t checkpoint_interval;
    int32_t rewind_back;
    int32_t rewind_to_anchor;
    int32_t ias15_hold_steps;

    /* --- drift repair --- */
    double drift_budget;
    double drift_error_threshold;
    int32_t resync_on_switch;
    int32_t richardson_levels; /* 2 or 3 */
    int32_t compensated_sums;
    int32_t reserved0;

    /* --- IAS15 driver --- */
    double ias15_dt_guess;     /* initial IAS15 step guess of a fresh stretch; REBOUND's default is 1e-3 */

    /* --- outputs, allocated by the caller; capacity out_capacity samples --- */
    int64_t out_capacity;
    double *out_times;         /* (T)       */
    double *out_pos;           /* (T,n,3)   */
    double *out_vel;           /* (T,n,3)   */
    double *out_pre_pos;       /* (T,n,3) or NULL: raw states before snapping (only meaningful with ref_pos) */
    double *out_pre_vel;       /* (T,n,3) or NULL */
    double *out_raw;           /* (T)  worst-body raw index (NaN at sample 0) */
    double *out_score;         /* (T)  0..1 score           (NaN at sample 0) */
    double *out_ias15_frac;    /* (T)  fraction of IAS15 steps since the previous sample */
    uint8_t *out_active;       /* (T)  0 = leapfrog, 1 = ias15 */
    int64_t *out_nsw;          /* (T)  cumulative integrator switches */
} iris_config;

typedef struct iris_counts {
    int64_t n_steps;           /* nominal steps */
    int64_t n_ias15_steps;     /* steps taken by IAS15 in the final timeline */
    int64_t n_corrections;
    int64_t n_rewinds;
    int64_t n_rewound_steps;   /* steps discarded and replayed (extra work) */
    int64_t n_resyncs;
    int64_t n_switch_events;
    int64_t n_samples;         /* samples written */
    int64_t lf_steps_executed; /* leapfrog steps actually executed incl. replayed ones (work counter) */
    int64_t ias15_steps_executed;
    int64_t ias15_substeps;    /* REBOUND integrator steps actually taken (>= ias15_steps_executed) */
    int64_t n_resets;          /* IAS15 fresh-sim resets */
} iris_counts;

typedef struct iris_switch_event { int64_t step; double time; double score; int32_t to_ias15; int32_t pad; } iris_switch_event;
typedef struct iris_rewind_event { int64_t step; int64_t to_step; double time; double to_time; double score; int32_t depth; int32_t pad; } iris_rewind_event;
typedef struct iris_resync_event { int64_t step; double time; double estimated_error; int32_t kind; /* 0 budget, 1 switch */ int32_t pad; } iris_resync_event;

enum { IRIS_EVENTS_SWITCH = 0, IRIS_EVENTS_REWIND = 1, IRIS_EVENTS_RESYNC = 2 };

/* ---- pure Leapfrog baseline (core/leapfrog.py::run_leapfrog / run_leapfrog_with_correction) ----
 * Same velocity-Verlet step and the same force kernel as the adaptive loop, so timing comparisons are like-for-like.
 * ref_pos == NULL -> plain run_leapfrog (only the state arrays are written).
 * ref_pos != NULL -> run_leapfrog_with_correction: snaps to the reference when the RMS position error exceeds
 *                    error_threshold; the pre_check arrays (pos, vel, acc, jerk before any snap) are written. */
typedef struct iris_leapfrog_config {
    int32_t abi_version;
    int32_t n;
    double g, softening, dt;
    int64_t n_steps;
    int64_t steps_per_sample;
    const double *masses, *pos0, *vel0;
    const double *ref_pos, *ref_vel; /* (ref_samples, n, 3) or NULL */
    int64_t ref_samples;
    double error_threshold;
    int64_t out_capacity;            /* >= n_steps / steps_per_sample + 1 */
    double *out_times, *out_pos, *out_vel;
    double *out_pre_pos, *out_pre_vel, *out_pre_acc, *out_pre_jerk; /* NULL unless correcting */
    int64_t *out_corr_sample;        /* (out_capacity) sample index of each correction */
    int64_t *out_corr_step;          /* (out_capacity) integrator step of each correction */
    double *out_corr_error;          /* (out_capacity) error that triggered it */
} iris_leapfrog_config;

IRIS_API int iris_run_leapfrog(const iris_leapfrog_config *cfg, int64_t *n_corrections, char *err, int err_len);

IRIS_API int iris_abi_version(void);
IRIS_API const char *iris_rebound_version(void);
/* Returns 0 on success (handle set), nonzero on failure (message in err). */
IRIS_API int iris_run(const iris_config *cfg, void **handle, char *err, int err_len);
IRIS_API void iris_get_counts(void *handle, iris_counts *out);
/* Copies the events of one kind into out (sized from iris_get_counts: n_switch_events / n_rewinds / n_resyncs). */
IRIS_API void iris_get_events(void *handle, int kind, void *out);
IRIS_API void iris_free(void *handle);

#ifdef __cplusplus
}
#endif
#endif
