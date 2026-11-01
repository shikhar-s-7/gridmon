/* DSP_SPEC section 1: Urms(1/2) on a fixed grid and the Urms features (FEATURES.md).
 *   U[c][k] = sqrt(2 * sum_{i=0}^{99} v[c][50k+i]^2 / 100)  (pu RMS), k = 0..98, v = lsb * x.
 * Features: per-channel min/max over k, sorted over channels (urms_min_1..3, urms_max_1..3);
 * t_below_090 / t_above_110 = 0.01 s x number of k with any channel < 0.90 / > 1.10;
 * aux_t_dip / aux_t_swell = window centre (50k + 50)/5000 s of the first such k. */
#include "gm_internal.h"

void gm_block_urms(gm_work_t *w)
{
    float cmin[GM_N_CH], cmax[GM_N_CH];
    int32_t n_below = 0, n_above = 0, k_dip = -1, k_swell = -1;
    int32_t k, c, i;

    for (c = 0; c < GM_N_CH; c++) {
        cmin[c] = 0.0f;
        cmax[c] = 0.0f;
    }
    for (k = 0; k < GM_N_URMS; k++) {
        float lo = 0.0f, hi = 0.0f;
        for (c = 0; c < GM_N_CH; c++) {
            const int16_t *p = w->x + (int32_t)c * GM_N_RECORD + k * GM_URMS_HOP;
            float acc = 0.0f, u;
            for (i = 0; i < GM_N_CYCLE; i++) {
                float v = w->lsb * (float)p[i];
                acc += v * v;
            }
            u = GM_SQRTF(GM_K_URMS_SCALE * acc);
#ifdef GM_TRACES
            if (w->tr) {
                w->tr->urms[c][k] = u;
            }
#endif
            if (k == 0 || u < cmin[c]) {
                cmin[c] = u;
            }
            if (k == 0 || u > cmax[c]) {
                cmax[c] = u;
            }
            if (c == 0 || u < lo) {
                lo = u;
            }
            if (c == 0 || u > hi) {
                hi = u;
            }
        }
        if (lo < GM_TH_URMS_LOW_LT) {
            n_below++;
            if (k_dip < 0) {
                k_dip = k;
            }
        }
        if (hi > GM_TH_URMS_HIGH_GT) {
            n_above++;
            if (k_swell < 0) {
                k_swell = k;
            }
        }
    }
    gm_sort3(cmin);
    gm_sort3(cmax);
    for (c = 0; c < GM_N_CH; c++) {
        gm_set_feature(w->out, GM_F_URMS_MIN_1 + c, cmin[c], 1u);
        gm_set_feature(w->out, GM_F_URMS_MAX_1 + c, cmax[c], 1u);
    }
    gm_set_feature(w->out, GM_F_T_BELOW_090, (float)n_below * GM_K_URMS_T_STEP, 1u);
    gm_set_feature(w->out, GM_F_T_ABOVE_110, (float)n_above * GM_K_URMS_T_STEP, 1u);
    gm_set_aux(w->out, GM_A_T_DIP, k_dip >= 0 ? gm_time_of_sample(k_dip * GM_URMS_HOP + GM_URMS_HOP) : 0.0f,
               (uint16_t)(k_dip >= 0));
    gm_set_aux(w->out, GM_A_T_SWELL, k_swell >= 0 ? gm_time_of_sample(k_swell * GM_URMS_HOP + GM_URMS_HOP) : 0.0f,
               (uint16_t)(k_swell >= 0));
}
