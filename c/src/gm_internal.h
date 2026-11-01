/* Internal declarations of the gridmon C core (not part of the public API).
 * Math hooks: every libm call goes through a GM_* macro so firmware can map them to TMU intrinsics
 * (e.g. -DGM_SINF=__sin is NOT a drop-in: TMU __sin takes per-unit angles; see docs/C_CORE.md). */
#ifndef GM_INTERNAL_H
#define GM_INTERNAL_H

#include <math.h>
#include <stdint.h>

#include "gridmon.h"

#ifndef GM_SINF
#define GM_SINF(x) sinf(x)
#endif
#ifndef GM_COSF
#define GM_COSF(x) cosf(x)
#endif
#ifndef GM_ATAN2F
#define GM_ATAN2F(y, x) atan2f((y), (x))
#endif
#ifndef GM_SQRTF
#define GM_SQRTF(x) sqrtf(x)
#endif
#ifndef GM_DIVF
#define GM_DIVF(a, b) ((a) / (b))
#endif
#ifndef GM_FLOORF
#define GM_FLOORF(x) floorf(x)
#endif
#ifndef GM_EXPF
#define GM_EXPF(x) expf(x)
#endif
#ifndef GM_FABSF
#define GM_FABSF(x) fabsf(x)
#endif

#if GM_REPORT_NAN || defined(GM_TRACES)
#define GM_NANF ((float)NAN)
#endif

/* wrap(x) = x - 2 pi floor((x + pi) / (2 pi)) -> [-pi, pi)   (DSP_SPEC section 0) */
static inline float gm_wrap(float x)
{
    return x - GM_K_TWO_PI * GM_FLOORF(GM_DIVF(x + GM_K_PI, GM_K_TWO_PI));
}

/* time (s) of sample index n: n / 5000 */
static inline float gm_time_of_sample(int32_t n)
{
    return GM_DIVF((float)n, GM_K_FS);
}

/* time (s) of phasor m: (5m + 4) / 5000 */
static inline float gm_time_of_phasor(int32_t m)
{
    return gm_time_of_sample(GM_STEP * m + (GM_STEP - 1));
}

/* Per-record intermediate scalars shared between the blocks (lives on the caller's stack; small). */
typedef struct {
    const int16_t *x;          /* planar record, channel c at x + c * GM_N_RECORD */
    float lsb;                 /* pu per count */
    gm_result_t *out;
#ifdef GM_TRACES
    gm_traces_t *tr;           /* NULL: no traces */
#endif
    /* phasor block */
    float v1_ref;              /* median |V1| over REF */
    uint16_t v1_ref_ok;        /* v1_ref > 0 */
    float a1_obs_min, a1_obs_max, a1_end_sum;
    int32_t m_star;            /* first argmin_OBS |V1| */
    float v2_star_re, v2_star_im;
    float u2_obs_max, u2_end_sum;
    /* frequency block */
    float df_ref;              /* f_ref - 50 (Hz) */
    float jmax;                /* max |J| over OBS (rad) */
    int32_t m_jump;            /* first argmax |J| over OBS */
    uint16_t jump_detected;
} gm_work_t;

/* blocks (one file per DSP_SPEC section) */
void gm_block_urms(gm_work_t *w);                       /* gm_rms.c      (section 1) */
void gm_block_phasors(gm_ctx_t *ctx, gm_work_t *w);     /* gm_phasor.c   (section 2) */
void gm_block_frequency(gm_ctx_t *ctx, gm_work_t *w);   /* gm_freq.c     (sections 3-5) */
void gm_block_pll(gm_work_t *w);                        /* gm_pll.c      (section 6) */
void gm_block_envelope(gm_ctx_t *ctx, gm_work_t *w);    /* gm_envelope.c (section 7) */
void gm_block_thd(gm_work_t *w);                        /* gm_thd.c      (section 8) */
void gm_block_unbalance(gm_ctx_t *ctx, gm_work_t *w);   /* gm_features.c (section 9: |V1|, |V2|, rho) */
void gm_rules(gm_result_t *out);                        /* gm_rules.c    (section 10) */
#if GM_ENABLE_GBM
void gm_gbm(gm_result_t *out);                          /* gm_gbm.c      (MODEL_FORMAT section 4) */
#endif

/* helpers (gm_features.c) */
float gm_median(float *buf, int32_t n);                 /* sorts buf[0..n-1] in place; even n: mean of middle two */
void gm_sort3(float v[3]);
void gm_set_feature(gm_result_t *out, int32_t idx, float value, uint16_t valid);
void gm_set_aux(gm_result_t *out, int32_t idx, float value, uint16_t valid);

#endif /* GM_INTERNAL_H */
