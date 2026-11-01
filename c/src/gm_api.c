/* Public API of the gridmon C core: init, record processing (block order), version. */
#include <limits.h>

#include "gm_internal.h"

#if GM_ENABLE_GBM
#include "gm_model_gbm.h"
#endif

static void gm_zero(float *p, int32_t n)
{
    int32_t i;
    for (i = 0; i < n; i++) {
        p[i] = 0.0f;
    }
}

gm_status_t gm_init(gm_ctx_t *ctx)
{
    if (ctx == 0) {
        return GM_ERR_NULL;
    }
    ctx->magic = GM_CTX_MAGIC;
    ctx->reserved = 0u;
    ctx->n_records = 0u;
    gm_zero(ctx->a1, GM_M);
    gm_zero(ctx->ang, GM_M);
    gm_zero(ctx->df, GM_M);
    gm_zero(ctx->ref_u2, GM_N_REF);
    gm_zero(ctx->ref_v2_re, GM_N_REF);
    gm_zero(ctx->ref_v2_im, GM_N_REF);
    gm_zero(ctx->scratch, GM_N_REF);
    gm_zero(ctx->e_mag, GM_P);
    gm_zero(ctx->e_ang, GM_P);
    gm_zero(ctx->r_mag, GM_P);
    gm_zero(ctx->r_ang, GM_P);
    gm_zero(ctx->spec, GM_ENV_NF);
    return GM_OK;
}

uint32_t gm_ctx_words(void)
{
    /* sizeof counts chars: 8-bit bytes on hosts, 16-bit words on C28x */
    return (uint32_t)((sizeof(gm_ctx_t) * (uint32_t)CHAR_BIT) / 16u);
}

const char *gm_version(void)
{
    return "gridmon-core " GM_CORE_VERSION " (model " GM_MODEL_ID "; tables " GM_TABLES_ID "; gbm "
#if GM_ENABLE_GBM
           "on"
#else
           "off"
#endif
           ")";
}

static void gm_result_reset(gm_result_t *out, int16_t status)
{
    int32_t i;
    out->status = status;
    out->flags = 0u;
    for (i = 0; i < GM_N_FEATURES; i++) {
        out->features[i] = 0.0f;
        out->feature_valid[i] = 0u;
    }
    for (i = 0; i < GM_N_AUX; i++) {
        out->aux[i] = 0.0f;
        out->aux_valid[i] = 0u;
    }
    out->rule_label = GM_CLASS_NORMAL;
    out->rule_reason = GM_REASON_NONE;
    out->onset_estimate_s = 0.0f;
    out->onset_valid = 0u;
    out->gbm_enabled = (int16_t)GM_ENABLE_GBM;
    out->gbm_label = -1;
    for (i = 0; i < GM_N_CLASSES; i++) {
        out->gbm_proba[i] = 0.0f;
        out->gbm_raw[i] = 0.0f;
    }
}

static gm_status_t gm_run(gm_ctx_t *ctx, const int16_t *x, int32_t n_per_ch, float lsb_pu, gm_result_t *out,
                          void *traces)
{
    gm_work_t w;
    if (out == 0) {
        return GM_ERR_NULL;
    }
    if (ctx == 0 || x == 0) {
        gm_result_reset(out, (int16_t)GM_ERR_NULL);
        return GM_ERR_NULL;
    }
    if (ctx->magic != GM_CTX_MAGIC) {
        gm_result_reset(out, (int16_t)GM_ERR_NOT_INIT);
        return GM_ERR_NOT_INIT;
    }
    if (n_per_ch != (int32_t)GM_N_RECORD) {
        gm_result_reset(out, (int16_t)GM_ERR_LENGTH);
        return GM_ERR_LENGTH;
    }
    if (!(lsb_pu > 0.0f)) {
        gm_result_reset(out, (int16_t)GM_ERR_LSB);
        return GM_ERR_LSB;
    }
    gm_result_reset(out, (int16_t)GM_OK);
    w.x = x;
    w.lsb = lsb_pu;
    w.out = out;
#ifdef GM_TRACES
    w.tr = (gm_traces_t *)traces;
#else
    (void)traces;
#endif
    gm_block_urms(&w);                /* section 1 */
    gm_block_phasors(ctx, &w);        /* section 2: |V1|, angle, REF buffers, OBS/END statistics */
    gm_block_frequency(ctx, &w);      /* sections 3-5: f_ref, J, masks, frequency features */
    gm_block_unbalance(ctx, &w);      /* section 9: v1_ref, |V2| ratios, rho (needs f_ref) */
    gm_block_pll(&w);                 /* section 6 (needs f_ref) */
    gm_block_envelope(ctx, &w);       /* section 7 (needs v1_ref, f_ref) */
    gm_block_thd(&w);                 /* section 8 */
    gm_rules(out);                    /* section 10: PRIMARY output */
#if GM_ENABLE_GBM
    gm_gbm(out);                      /* ADVISORY output */
#endif
    ctx->n_records++;
    return GM_OK;
}

gm_status_t gm_process_record(gm_ctx_t *ctx, const int16_t *x, int32_t n_per_ch, float lsb_pu, gm_result_t *out)
{
    return gm_run(ctx, x, n_per_ch, lsb_pu, out, 0);
}

#ifdef GM_TRACES
gm_status_t gm_process_record_traces(gm_ctx_t *ctx, const int16_t *x, int32_t n_per_ch, float lsb_pu,
                                     gm_result_t *out, gm_traces_t *tr)
{
    return gm_run(ctx, x, n_per_ch, lsb_pu, out, tr);
}
#endif
