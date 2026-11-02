/* WebAssembly glue for the gridmon C core (M8). Not part of the portable core: it owns static buffers
 * (the browser module is a single-instance reactor) and flattens results/traces into float32 arrays so
 * JavaScript needs no struct layout. Built by scripts/build_site.py with
 *   python -m ziglang cc -target wasm32-wasi -mexec-model=reactor -Wl,--no-entry
 *   -DGM_TRACES -DGM_REPORT_NAN=1 -DGM_ENABLE_GBM=1  (same core sources as the SIL library)
 *
 * Output layout (gm_wasm_output, GM_WASM_OUT_LEN floats):
 *   [0] status  [1] flags  [2] rule_label  [3] rule_reason  [4] onset_s (NaN if none)  [5] gbm_label
 *   [6..12] gbm_proba[7]   [13..49] features[37] (NaN = invalid)   [50..57] aux[8] (NaN = invalid)
 * Trace layout (gm_wasm_traces, GM_WASM_TR_LEN floats; NaN where the reference is NaN):
 *   urms[3][99] | v1_re v1_im v2_re v2_im [1000 each] | df_dft rocof J f_pll_dev [1000 each] |
 *   fmask rmask [1000 each, 0/1] | e_mag e_ang r_mag r_ang [100 each] | S_mag S_ang [5][28] | thd[10]
 */
#include <math.h>
#include <stdint.h>

#include "gridmon.h"

#define GM_WASM_OUT_LEN 58
#define GM_WASM_TR_LEN (3 * GM_N_URMS + 10 * GM_M + 4 * GM_P + 2 * GM_ENV_NWIN * GM_ENV_NF + GM_THD_NWIN)

#define GM_EXPORT(name) __attribute__((export_name(#name)))

static gm_ctx_t g_ctx;
static int16_t g_x[GM_N_CH * GM_N_RECORD];
static gm_result_t g_res;
static gm_traces_t g_tr;
static float g_out[GM_WASM_OUT_LEN];
static float g_trf[GM_WASM_TR_LEN];

GM_EXPORT(gm_wasm_input) int16_t *gm_wasm_input(void) { return g_x; }
GM_EXPORT(gm_wasm_output) float *gm_wasm_output(void) { return g_out; }
GM_EXPORT(gm_wasm_traces) float *gm_wasm_traces(void) { return g_trf; }
GM_EXPORT(gm_wasm_out_len) int32_t gm_wasm_out_len(void) { return GM_WASM_OUT_LEN; }
GM_EXPORT(gm_wasm_tr_len) int32_t gm_wasm_tr_len(void) { return GM_WASM_TR_LEN; }
GM_EXPORT(gm_wasm_version) const char *gm_wasm_version(void) { return gm_version(); }
GM_EXPORT(gm_wasm_ctx_bytes) int32_t gm_wasm_ctx_bytes(void) { return (int32_t)sizeof(gm_ctx_t); }

static int32_t put(float *dst, int32_t k, const float *src, int32_t n)
{
    int32_t i;
    for (i = 0; i < n; i++) {
        dst[k + i] = src[i];
    }
    return k + n;
}

/* Classify the record in gm_wasm_input() (planar int16, 3 x 5000) with v = lsb * count (pu). */
GM_EXPORT(gm_wasm_run) int32_t gm_wasm_run(float lsb, int32_t with_traces)
{
    int32_t i, k = 0;
    gm_status_t st;
    gm_init(&g_ctx);
    st = gm_process_record_traces(&g_ctx, g_x, GM_N_RECORD, lsb, &g_res, with_traces ? &g_tr : 0);
    g_out[0] = (float)g_res.status;
    g_out[1] = (float)g_res.flags;
    g_out[2] = (float)g_res.rule_label;
    g_out[3] = (float)g_res.rule_reason;
    g_out[4] = g_res.onset_valid ? g_res.onset_estimate_s : (float)NAN;
    g_out[5] = (float)g_res.gbm_label;
    for (i = 0; i < GM_N_CLASSES; i++) {
        g_out[6 + i] = g_res.gbm_proba[i];
    }
    for (i = 0; i < GM_N_FEATURES; i++) {
        g_out[13 + i] = g_res.features[i]; /* GM_REPORT_NAN=1: NaN where invalid */
    }
    for (i = 0; i < GM_N_AUX; i++) {
        g_out[50 + i] = g_res.aux[i];
    }
    if (with_traces) {
        for (i = 0; i < GM_N_CH; i++) {
            k = put(g_trf, k, g_tr.urms[i], GM_N_URMS);
        }
        k = put(g_trf, k, g_tr.v1_re, GM_M);
        k = put(g_trf, k, g_tr.v1_im, GM_M);
        k = put(g_trf, k, g_tr.v2_re, GM_M);
        k = put(g_trf, k, g_tr.v2_im, GM_M);
        k = put(g_trf, k, g_tr.df_dft, GM_M);
        k = put(g_trf, k, g_tr.rocof, GM_M);
        k = put(g_trf, k, g_tr.J, GM_M);
        k = put(g_trf, k, g_tr.f_pll_dev, GM_M);
        for (i = 0; i < GM_M; i++) {
            g_trf[k + i] = (float)g_tr.fmask[i];
            g_trf[k + GM_M + i] = (float)g_tr.rmask[i];
        }
        k += 2 * GM_M;
        k = put(g_trf, k, g_tr.e_mag, GM_P);
        k = put(g_trf, k, g_tr.e_ang, GM_P);
        k = put(g_trf, k, g_tr.r_mag, GM_P);
        k = put(g_trf, k, g_tr.r_ang, GM_P);
        for (i = 0; i < GM_ENV_NWIN; i++) {
            k = put(g_trf, k, g_tr.S_mag[i], GM_ENV_NF);
        }
        for (i = 0; i < GM_ENV_NWIN; i++) {
            k = put(g_trf, k, g_tr.S_ang[i], GM_ENV_NF);
        }
        k = put(g_trf, k, g_tr.thd, GM_THD_NWIN);
    }
    return (int32_t)st;
}
