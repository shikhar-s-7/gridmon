/* DSP_SPEC section 2: one-cycle DFT phasors at 1 kHz and symmetrical components.
 *   X_c(m) = (2/100) sum_{i=0}^{99} v[c][n-i] W[(n-i) mod 100],  n = 5m + 4,  m = 19..999,
 *   W[k] = exp(-j 2 pi k/100) (gm_tw100_*), non-recursive (no sliding DFT).
 *   V1 = (X_ab + a X_bc + a^2 X_ca)/3,  V2 = (X_ab + a^2 X_bc + a X_ca)/3,  a = exp(j 2 pi/3).
 * Stores |V1(m)| and angle V1(m) = atan2f(Im, Re) for the later blocks, the REF values of
 * |V2|/max(|V1|, 0.05) and V2, and running OBS/END statistics (FEATURES.md, single pass). */
#include "gm_internal.h"

void gm_block_phasors(gm_ctx_t *ctx, gm_work_t *w)
{
    int32_t m;
    w->a1_obs_min = 0.0f;
    w->a1_obs_max = 0.0f;
    w->a1_end_sum = 0.0f;
    w->m_star = GM_M_OBS_FIRST;
    w->v2_star_re = 0.0f;
    w->v2_star_im = 0.0f;
    w->u2_obs_max = 0.0f;
    w->u2_end_sum = 0.0f;
    for (m = 0; m < GM_M_FIRST; m++) {
        ctx->a1[m] = 0.0f;
        ctx->ang[m] = 0.0f;
#ifdef GM_TRACES
        if (w->tr) {
            w->tr->v1_re[m] = GM_NANF;
            w->tr->v1_im[m] = GM_NANF;
            w->tr->v2_re[m] = GM_NANF;
            w->tr->v2_im[m] = GM_NANF;
        }
#endif
    }
    for (m = GM_M_FIRST; m < GM_M; m++) {
        const int32_t n = GM_STEP * m + (GM_STEP - 1);
        const int32_t j0 = n - (GM_N_CYCLE - 1);
        float xr[GM_N_CH], xi[GM_N_CH];
        float v1r, v1i, v2r, v2i, a1, a2, u2;
        int32_t c, i;
        for (c = 0; c < GM_N_CH; c++) {
            const int16_t *p = w->x + c * GM_N_RECORD + j0;
            int32_t idx = j0 % GM_N_CYCLE;
            float sr = 0.0f, si = 0.0f;
            for (i = 0; i < GM_N_CYCLE; i++) {
                float v = w->lsb * (float)p[i];
                sr += v * gm_tw100_re[idx];
                si += v * gm_tw100_im[idx];
                idx++;
                if (idx == GM_N_CYCLE) {
                    idx = 0;
                }
            }
            xr[c] = GM_K_DFT_SCALE * sr;
            xi[c] = GM_K_DFT_SCALE * si;
        }
        /* a X = (a_re x_re - a_im x_im) + j (a_re x_im + a_im x_re) */
        v1r = (xr[0] + (GM_K_A_RE * xr[1] - GM_K_A_IM * xi[1]) + (GM_K_A2_RE * xr[2] - GM_K_A2_IM * xi[2])) * GM_K_THIRD;
        v1i = (xi[0] + (GM_K_A_RE * xi[1] + GM_K_A_IM * xr[1]) + (GM_K_A2_RE * xi[2] + GM_K_A2_IM * xr[2])) * GM_K_THIRD;
        v2r = (xr[0] + (GM_K_A2_RE * xr[1] - GM_K_A2_IM * xi[1]) + (GM_K_A_RE * xr[2] - GM_K_A_IM * xi[2])) * GM_K_THIRD;
        v2i = (xi[0] + (GM_K_A2_RE * xi[1] + GM_K_A2_IM * xr[1]) + (GM_K_A_RE * xi[2] + GM_K_A_IM * xr[2])) * GM_K_THIRD;
        a1 = GM_SQRTF(v1r * v1r + v1i * v1i);
        a2 = GM_SQRTF(v2r * v2r + v2i * v2i);
        u2 = GM_DIVF(a2, a1 > GM_K_V1_FLOOR ? a1 : GM_K_V1_FLOOR);
        ctx->a1[m] = a1;
        ctx->ang[m] = GM_ATAN2F(v1i, v1r);
#ifdef GM_TRACES
        if (w->tr) {
            w->tr->v1_re[m] = v1r;
            w->tr->v1_im[m] = v1i;
            w->tr->v2_re[m] = v2r;
            w->tr->v2_im[m] = v2i;
        }
#endif
        if (m <= GM_M_REF_LAST) {
            ctx->ref_u2[m - GM_M_FIRST] = u2;
            ctx->ref_v2_re[m - GM_M_FIRST] = v2r;
            ctx->ref_v2_im[m - GM_M_FIRST] = v2i;
        }
        if (m >= GM_M_OBS_FIRST) {
            if (m == GM_M_OBS_FIRST || a1 < w->a1_obs_min) { /* first argmin */
                w->a1_obs_min = a1;
                w->m_star = m;
                w->v2_star_re = v2r;
                w->v2_star_im = v2i;
            }
            if (m == GM_M_OBS_FIRST || a1 > w->a1_obs_max) {
                w->a1_obs_max = a1;
            }
            if (m == GM_M_OBS_FIRST || u2 > w->u2_obs_max) {
                w->u2_obs_max = u2;
            }
        }
        if (m >= GM_M_END_FIRST) {
            w->a1_end_sum += a1;
            w->u2_end_sum += u2;
        }
    }
}
