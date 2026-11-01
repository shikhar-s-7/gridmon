/* Host-only SIL helpers for the Python ctypes wrapper (src/gridmon/c_sil.py). Not part of the portable
 * core: uses malloc and is compiled only into the host shared library (GM_TRACES, GM_REPORT_NAN=1). */
#include <stdlib.h>

#include "gm_internal.h"

uint32_t gm_sil_sizeof_ctx(void) { return (uint32_t)sizeof(gm_ctx_t); }
uint32_t gm_sil_sizeof_result(void) { return (uint32_t)sizeof(gm_result_t); }
#ifdef GM_TRACES
uint32_t gm_sil_sizeof_traces(void) { return (uint32_t)sizeof(gm_traces_t); }
#endif

/* Process n_records planar records X[n][3][5000] with one context; outputs are row-major arrays.
 * Returns the number of records with status != GM_OK, or -1 if the context cannot be allocated. */
int32_t gm_sil_batch(const int16_t *X, int32_t n_records, float lsb, float *features, uint16_t *feature_valid,
                     float *aux, uint16_t *aux_valid, int16_t *rule_label, int16_t *rule_reason, float *onset,
                     uint16_t *onset_valid, int16_t *gbm_label, float *gbm_proba, float *gbm_raw, uint16_t *flags,
                     int16_t *status)
{
    gm_ctx_t *ctx = (gm_ctx_t *)malloc(sizeof(gm_ctx_t));
    gm_result_t r;
    int32_t i, k, n_bad = 0;
    if (ctx == NULL) {
        return -1;
    }
    gm_init(ctx);
    for (i = 0; i < n_records; i++) {
        const int16_t *x = X + (size_t)i * (size_t)(GM_N_CH * GM_N_RECORD);
        if (gm_process_record(ctx, x, GM_N_RECORD, lsb, &r) != GM_OK) {
            n_bad++;
        }
        for (k = 0; k < GM_N_FEATURES; k++) {
            features[(size_t)i * GM_N_FEATURES + (size_t)k] = r.features[k];
            feature_valid[(size_t)i * GM_N_FEATURES + (size_t)k] = r.feature_valid[k];
        }
        for (k = 0; k < GM_N_AUX; k++) {
            aux[(size_t)i * GM_N_AUX + (size_t)k] = r.aux[k];
            aux_valid[(size_t)i * GM_N_AUX + (size_t)k] = r.aux_valid[k];
        }
        for (k = 0; k < GM_N_CLASSES; k++) {
            gbm_proba[(size_t)i * GM_N_CLASSES + (size_t)k] = r.gbm_proba[k];
            gbm_raw[(size_t)i * GM_N_CLASSES + (size_t)k] = r.gbm_raw[k];
        }
        rule_label[i] = r.rule_label;
        rule_reason[i] = r.rule_reason;
        onset[i] = r.onset_estimate_s;
        onset_valid[i] = r.onset_valid;
        gbm_label[i] = r.gbm_label;
        flags[i] = r.flags;
        status[i] = r.status;
    }
    free(ctx);
    return n_bad;
}

#ifdef GM_TRACES
/* One record with every intermediate trace (golden-vector comparison). */
int32_t gm_sil_traces(const int16_t *x, int32_t n_per_ch, float lsb, gm_result_t *out, gm_traces_t *tr)
{
    gm_ctx_t *ctx = (gm_ctx_t *)malloc(sizeof(gm_ctx_t));
    int32_t st;
    if (ctx == NULL) {
        return -1;
    }
    gm_init(ctx);
    st = (int32_t)gm_process_record_traces(ctx, x, n_per_ch, lsb, out, tr);
    free(ctx);
    return st;
}
#endif

/* Error-path probe for tests: what = 0 NULL ctx, 1 NULL x, 2 uninitialised ctx, 3 wrong length, 4 bad lsb. */
int32_t gm_sil_error_probe(int32_t what, const int16_t *x)
{
    gm_ctx_t *ctx = (gm_ctx_t *)calloc(1, sizeof(gm_ctx_t));
    gm_result_t r;
    gm_status_t st = GM_OK;
    if (ctx == NULL) {
        return -1;
    }
    if (what != 2) {
        gm_init(ctx);
    }
    switch (what) {
    case 0: st = gm_process_record(NULL, x, GM_N_RECORD, 0x1p-10f, &r); break;
    case 1: st = gm_process_record(ctx, NULL, GM_N_RECORD, 0x1p-10f, &r); break;
    case 2: st = gm_process_record(ctx, x, GM_N_RECORD, 0x1p-10f, &r); break;
    case 3: st = gm_process_record(ctx, x, GM_N_RECORD - 1, 0x1p-10f, &r); break;
    case 4: st = gm_process_record(ctx, x, GM_N_RECORD, 0.0f, &r); break;
    default: break;
    }
    free(ctx);
    return (int32_t)st * 100 + (int32_t)r.status * 10 + (int32_t)(r.gbm_label < 0);
}

/* GBM walker on given feature vectors (model self-test vectors; walker equivalence). */
int32_t gm_sil_gbm(const float *features, const uint16_t *valid, int32_t n, int16_t *label, float *proba, float *raw)
{
#if GM_ENABLE_GBM
    gm_result_t r;
    int32_t i, k;
    for (i = 0; i < n; i++) {
        for (k = 0; k < GM_N_FEATURES; k++) {
            r.features[k] = features[(size_t)i * GM_N_FEATURES + (size_t)k];
            r.feature_valid[k] = valid[(size_t)i * GM_N_FEATURES + (size_t)k];
        }
        gm_gbm(&r);
        label[i] = r.gbm_label;
        for (k = 0; k < GM_N_CLASSES; k++) {
            proba[(size_t)i * GM_N_CLASSES + (size_t)k] = r.gbm_proba[k];
            raw[(size_t)i * GM_N_CLASSES + (size_t)k] = r.gbm_raw[k];
        }
    }
    return 0;
#else
    (void)features; (void)valid; (void)n; (void)label; (void)proba; (void)raw;
    return -1;
#endif
}

const char *gm_sil_version(void) { return gm_version(); }
uint32_t gm_sil_ctx_words(void) { return gm_ctx_words(); }
