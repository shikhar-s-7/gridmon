/* Feature helpers and DSP_SPEC section 9 magnitude/unbalance reductions (FEATURES.md):
 *   v1_ref = median |V1| over REF; v1_min_rel / v1_max_rel = min / max |V1| over OBS / v1_ref;
 *   v1_end_rel = mean |V1| over END / v1_ref; u2 = |V2|/max(|V1|, 0.05): median REF, max OBS, mean END;
 *   Bollen ratio rho = |V2d(m*) - V2d_ref| / max(v1_ref - |V1(m*)|, 0.01), m* = first argmin_OBS |V1|,
 *   V2d(m) = V2(m) exp(-j 2 pi (f_ref - 50) t_m), V2d_ref = median Re + j median Im over REF. */
#include "gm_internal.h"

float gm_median(float *buf, int32_t n)
{
    int32_t i, j;
    for (i = 1; i < n; i++) { /* insertion sort (n <= 131; no recursion, no allocation) */
        float key = buf[i];
        j = i - 1;
        while (j >= 0 && buf[j] > key) {
            buf[j + 1] = buf[j];
            j--;
        }
        buf[j + 1] = key;
    }
    if ((n % 2) == 1) {
        return buf[n / 2];
    }
    return GM_K_HALF * (buf[n / 2 - 1] + buf[n / 2]);
}

void gm_sort3(float v[3])
{
    float t;
    if (v[1] < v[0]) {
        t = v[0];
        v[0] = v[1];
        v[1] = t;
    }
    if (v[2] < v[1]) {
        t = v[1];
        v[1] = v[2];
        v[2] = t;
    }
    if (v[1] < v[0]) {
        t = v[0];
        v[0] = v[1];
        v[1] = t;
    }
}

void gm_set_feature(gm_result_t *out, int32_t idx, float value, uint16_t valid)
{
#if GM_REPORT_NAN
    out->features[idx] = valid ? value : GM_NANF;
#else
    out->features[idx] = valid ? value : 0.0f;
#endif
    out->feature_valid[idx] = valid ? 1u : 0u;
}

void gm_set_aux(gm_result_t *out, int32_t idx, float value, uint16_t valid)
{
#if GM_REPORT_NAN
    out->aux[idx] = valid ? value : GM_NANF;
#else
    out->aux[idx] = valid ? value : 0.0f;
#endif
    out->aux_valid[idx] = valid ? 1u : 0u;
}

void gm_block_unbalance(gm_ctx_t *ctx, gm_work_t *w)
{
    int32_t i;
    float v1_ref, u2_ref, v2d_ref_re, v2d_ref_im, ph, cs, sn, dre, dim, den, a1s;
    uint16_t ok;

    for (i = 0; i < GM_N_REF; i++) {
        ctx->scratch[i] = ctx->a1[GM_M_FIRST + i];
    }
    v1_ref = gm_median(ctx->scratch, GM_N_REF);
    ok = (uint16_t)(v1_ref > 0.0f);
    w->v1_ref = v1_ref;
    w->v1_ref_ok = ok;
    if (!ok) {
        w->out->flags |= GM_FLAG_NO_SIGNAL;
    }
    gm_set_feature(w->out, GM_F_V1_REF, v1_ref, 1u);
    gm_set_feature(w->out, GM_F_V1_MIN_REL, ok ? GM_DIVF(w->a1_obs_min, v1_ref) : 0.0f, ok);
    gm_set_feature(w->out, GM_F_V1_MAX_REL, ok ? GM_DIVF(w->a1_obs_max, v1_ref) : 0.0f, ok);
    gm_set_feature(w->out, GM_F_V1_END_REL, ok ? GM_DIVF(w->a1_end_sum * GM_K_END_INV, v1_ref) : 0.0f, ok);

    for (i = 0; i < GM_N_REF; i++) {
        ctx->scratch[i] = ctx->ref_u2[i];
    }
    u2_ref = gm_median(ctx->scratch, GM_N_REF);
    gm_set_feature(w->out, GM_F_U2_REF, u2_ref, 1u);
    gm_set_feature(w->out, GM_F_U2_MAX, w->u2_obs_max, 1u);
    gm_set_feature(w->out, GM_F_U2_END, w->u2_end_sum * GM_K_END_INV, 1u);

    /* V2 de-rotated by the reference frequency: real parts, then imaginary parts (one scratch buffer) */
    for (i = 0; i < GM_N_REF; i++) {
        ph = -(GM_K_TWO_PI * w->df_ref) * gm_time_of_phasor(GM_M_FIRST + i);
        ctx->scratch[i] = ctx->ref_v2_re[i] * GM_COSF(ph) - ctx->ref_v2_im[i] * GM_SINF(ph);
    }
    v2d_ref_re = gm_median(ctx->scratch, GM_N_REF);
    for (i = 0; i < GM_N_REF; i++) {
        ph = -(GM_K_TWO_PI * w->df_ref) * gm_time_of_phasor(GM_M_FIRST + i);
        ctx->scratch[i] = ctx->ref_v2_re[i] * GM_SINF(ph) + ctx->ref_v2_im[i] * GM_COSF(ph);
    }
    v2d_ref_im = gm_median(ctx->scratch, GM_N_REF);

    ph = -(GM_K_TWO_PI * w->df_ref) * gm_time_of_phasor(w->m_star);
    cs = GM_COSF(ph);
    sn = GM_SINF(ph);
    dre = (w->v2_star_re * cs - w->v2_star_im * sn) - v2d_ref_re;
    dim = (w->v2_star_re * sn + w->v2_star_im * cs) - v2d_ref_im;
    a1s = ctx->a1[w->m_star];
    den = v1_ref - a1s;
    if (!(den > GM_K_RHO_DEN_FLOOR)) {
        den = GM_K_RHO_DEN_FLOOR;
    }
    gm_set_feature(w->out, GM_F_RHO, GM_DIVF(GM_SQRTF(dre * dre + dim * dim), den), 1u);
}
