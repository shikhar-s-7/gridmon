/* gridmon portable C core (M6): grid-event monitor for 1 s three-phase LL voltage records.
 *
 * Reproduces the frozen Python reference (docs/DSP_SPEC.md, docs/FEATURES.md, src/gridmon/):
 * DSP traces -> 37 features -> priority rules (PRIMARY output, D-050) and the GBM-embedded model
 * (ADVISORY output, compiled in when GM_ENABLE_GBM != 0).
 * C99, float32 only, no malloc, no recursion, no I/O, no static mutable state: everything lives in
 * the caller's gm_ctx_t. C28x-safe types (16-bit char/int there): int16_t/int32_t/uint16_t/uint32_t.
 * Simulation-only: monitor output, never a control or protection input. API: docs/C_CORE.md.
 */
#ifndef GRIDMON_H
#define GRIDMON_H

#include <stdint.h>

#include "gm_tables.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- build options */
#ifndef GM_ENABLE_GBM
#define GM_ENABLE_GBM 1     /* 1: run the GBM advisory classifier (host/web default; C2000: see C_CORE.md) */
#endif
#ifndef GM_REPORT_NAN
#define GM_REPORT_NAN 0     /* 1 (host SIL builds): invalid features are written as NaN, like Python; 0: 0.0f */
#endif

#define GM_CORE_VERSION "0.6.0"
#define GM_N_FEATURES 37
#define GM_N_AUX 8
#define GM_N_CLASSES 7
#define GM_N_REASONS 8

/* ---- status codes (gm_process_record return value and gm_result_t.status) */
typedef enum {
    GM_OK = 0,
    GM_ERR_NULL = 1,      /* a required pointer is NULL */
    GM_ERR_NOT_INIT = 2,  /* gm_init was not called on this context */
    GM_ERR_LENGTH = 3,    /* n_per_ch != GM_N_RECORD (5000) */
    GM_ERR_LSB = 4        /* lsb_pu not > 0 */
} gm_status_t;

/* ---- result flags (gm_result_t.flags) */
#define GM_FLAG_JUMP_DETECTED   0x0001u  /* |J| > 3 deg in OBS: frequency features jump-masked */
#define GM_FLAG_DF_END_FALLBACK 0x0002u  /* END fully jump-masked: df_end uses the unmasked mean */
#define GM_FLAG_ROCOF_END_NONE  0x0004u  /* END fully ROCOF-masked: rocof_end invalid (D-051) */
#define GM_FLAG_NO_SIGNAL       0x0008u  /* v1_ref == 0: |V1|-relative features invalid */
#define GM_FLAG_THD_INVALID     0x0010u  /* a channel had zero fundamental in a THD window */

/* ---- feature indices (order = gridmon.features.FEATURE_NAMES = GBM input order) */
enum {
    GM_F_URMS_MIN_1 = 0, GM_F_URMS_MIN_2, GM_F_URMS_MIN_3, GM_F_URMS_MAX_1, GM_F_URMS_MAX_2, GM_F_URMS_MAX_3,
    GM_F_T_BELOW_090, GM_F_T_ABOVE_110, GM_F_V1_REF, GM_F_V1_MIN_REL, GM_F_V1_MAX_REL, GM_F_V1_END_REL,
    GM_F_U2_REF, GM_F_U2_MAX, GM_F_U2_END, GM_F_RHO, GM_F_J_MAX_DEG, GM_F_J_SIGN, GM_F_F_REF, GM_F_DF_END,
    GM_F_DF_MAX, GM_F_ROCOF_MAX, GM_F_T_ROCOF_02, GM_F_ROCOF_END, GM_F_DFPLL_MAX, GM_F_T_FPLL_05,
    GM_F_ENV_MAG_AMP_PCT, GM_F_ENV_MAG_FREQ, GM_F_ENV_MAG_PROM, GM_F_ENV_ANG_AMP_DEG, GM_F_ENV_ANG_FREQ,
    GM_F_ENV_ANG_PROM, GM_F_ENV_MAG_RMS_PCT, GM_F_ENV_ANG_RMS_DEG, GM_F_THD_FIRST, GM_F_THD_LAST, GM_F_THD_MAX
};
/* auxiliary outputs (gridmon.features.AUX_NAMES; never model inputs) */
enum {
    GM_A_T_DIP = 0, GM_A_T_SWELL, GM_A_T_JUMP, GM_A_T_FREQ, GM_A_T_OSC_MAG, GM_A_T_OSC_ANG,
    GM_A_ENV_MAG_GAP, GM_A_ENV_ANG_GAP
};
/* classes (gridmon.CLASSES; rule_label and gbm_label values) */
enum {
    GM_CLASS_NORMAL = 0, GM_CLASS_SAG_BAL, GM_CLASS_SAG_UNBAL, GM_CLASS_SWELL, GM_CLASS_PHASE_JUMP,
    GM_CLASS_FREQ_EVENT, GM_CLASS_OSCILLATION
};
/* rule reason codes (gridmon.baseline_rules.REASONS) */
enum {
    GM_REASON_NONE = 0, GM_REASON_DIP_BALANCED, GM_REASON_DIP_UNBALANCED, GM_REASON_SWELL,
    GM_REASON_OSC_MAGNITUDE, GM_REASON_OSC_ANGLE, GM_REASON_PHASE_JUMP, GM_REASON_FREQ_OFFSET
};

/* ---- per-record result */
typedef struct {
    int16_t status;                       /* gm_status_t */
    uint16_t flags;                       /* GM_FLAG_* */
    float features[GM_N_FEATURES];        /* units: docs/FEATURES.md */
    uint16_t feature_valid[GM_N_FEATURES];/* 1 = finite in the Python reference; 0 = invalid (value 0 or NaN) */
    float aux[GM_N_AUX];                  /* onset times (s) and spectral gaps */
    uint16_t aux_valid[GM_N_AUX];
    int16_t rule_label;                   /* PRIMARY output: GM_CLASS_* */
    int16_t rule_reason;                  /* GM_REASON_* */
    float onset_estimate_s;               /* first detection time of the firing detector (indicative) */
    uint16_t onset_valid;                 /* 0 for normal */
    int16_t gbm_enabled;                  /* GM_ENABLE_GBM at build time */
    int16_t gbm_label;                    /* ADVISORY output: GM_CLASS_*, -1 when GBM is disabled */
    float gbm_proba[GM_N_CLASSES];        /* softmax of the raw scores (0 when disabled) */
    float gbm_raw[GM_N_CLASSES];          /* raw scores (label = first argmax of raw) */
} gm_result_t;

/* ---- working context (caller-provided; no other state exists). Only gm_init/gm_process_record
 * may touch the fields. Phasor-rate buffers are indexed by phasor m (entries m < 19 unused). */
typedef struct {
    uint16_t magic;                       /* GM_CTX_MAGIC after gm_init */
    uint16_t reserved;
    uint32_t n_records;                   /* records processed since gm_init */
    float a1[GM_M];                       /* |V1(m)| (pu) */
    float ang[GM_M];                      /* angle V1(m) (rad) */
    float df[GM_M];                       /* df_raw, then f_dft - 50 (Hz), in place */
    float ref_u2[GM_N_REF];               /* |V2|/max(|V1|,0.05) over REF */
    float ref_v2_re[GM_N_REF];            /* V2 over REF (for the de-rotated reference) */
    float ref_v2_im[GM_N_REF];
    float scratch[GM_N_REF];              /* sorted copies for medians (>= 131 values) */
    float e_mag[GM_P];                    /* envelope (fraction of v1_ref) */
    float e_ang[GM_P];                    /* envelope angle (rad) */
    float r_mag[GM_P];                    /* detrended envelopes */
    float r_ang[GM_P];
    float spec[GM_ENV_NF];                /* one envelope spectrum window */
} gm_ctx_t;

#define GM_CTX_MAGIC 0x6D36u

/* ---- optional intermediate traces (host SIL builds only; golden/README.md arrays). Entries the
 * reference marks NaN are written as NaN. Frequencies are deviations from 50 Hz. */
#ifdef GM_TRACES
typedef struct {
    float urms[GM_N_CH][GM_N_URMS];
    float v1_re[GM_M], v1_im[GM_M], v2_re[GM_M], v2_im[GM_M];
    float df_dft[GM_M];                   /* f_dft - 50 */
    float rocof[GM_M];                    /* unmasked */
    float J[GM_M];
    float f_pll_dev[GM_M];                /* f_pll - 50 at n = 5m + 4 */
    int16_t fmask[GM_M], rmask[GM_M];
    float e_mag[GM_P], e_ang[GM_P], r_mag[GM_P], r_ang[GM_P];
    float S_mag[GM_ENV_NWIN][GM_ENV_NF], S_ang[GM_ENV_NWIN][GM_ENV_NF];
    float thd[GM_THD_NWIN];
} gm_traces_t;
#endif

/* Reset a context (all buffers zeroed, magic set). Must be called once before gm_process_record. */
gm_status_t gm_init(gm_ctx_t *ctx);

/* Size of gm_ctx_t in 16-bit words (sizeof is in 16-bit chars on C28x, 8-bit bytes on hosts). */
uint32_t gm_ctx_words(void);

/* Classify one record. x: planar int16 [ab | bc | ca], n_per_ch = 5000 samples per channel at 5 kHz;
 * v = lsb_pu * x in pu of nominal LL peak (lsb_pu = 2^-10 for the reference 12-bit +/- 2 pu ADC).
 * Fills *out (always, also on error: status set, labels normal / -1). Never reads beyond x[3*5000-1]. */
gm_status_t gm_process_record(gm_ctx_t *ctx, const int16_t *x, int32_t n_per_ch, float lsb_pu, gm_result_t *out);

#ifdef GM_TRACES
/* Same as gm_process_record; additionally writes every intermediate trace to *tr (if not NULL). */
gm_status_t gm_process_record_traces(gm_ctx_t *ctx, const int16_t *x, int32_t n_per_ch, float lsb_pu,
                                     gm_result_t *out, gm_traces_t *tr);
#endif

/* Version string: core version, model id and constant-table id. */
const char *gm_version(void);

#ifdef __cplusplus
}
#endif

#endif /* GRIDMON_H */
