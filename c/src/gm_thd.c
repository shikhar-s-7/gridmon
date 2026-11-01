/* DSP_SPEC section 8: windowed THD by direct DFT at the harmonic bins.
 *   X_h = sum_{j=0}^{499} v[c][500w + j] T[(5 h j) mod 500],  T[i] = exp(-j 2 pi i/500),  h = 1..15
 *   THD_c = sqrt(sum_{h=2}^{15} |X_h|^2) / |X_1|;  THD(w) = mean over the 3 channels;  w = 0..9.
 * Features: thd_first = THD(0), thd_last = THD(9), thd_max = max_w THD(w). A channel with zero
 * fundamental makes the reference non-finite: the C core flags those features invalid instead.
 * C note: each 500-term sum is accumulated as 10 partial sums of 50 terms (same sum, smaller float32
 * rounding error; the reference is float64). */
#include "gm_internal.h"

#define GM_THD_BLK 50

void gm_block_thd(gm_work_t *w)
{
    float thd_first = 0.0f, thd_last = 0.0f, thd_max = 0.0f;
    uint16_t ok_first = 1u, ok_last = 1u, ok_all = 1u;
    int32_t win, c, h, b, j;

    for (win = 0; win < GM_THD_NWIN; win++) {
        float sum_c = 0.0f;
        uint16_t ok = 1u;
        for (c = 0; c < GM_N_CH; c++) {
            const int16_t *p = w->x + c * GM_N_RECORD + win * GM_THD_WIN;
            float p1 = 0.0f, ph = 0.0f;
            for (h = 1; h <= GM_THD_HMAX; h++) {
                const int32_t step = GM_STEP * h; /* bin 5h = 50h Hz */
                int32_t idx = 0;
                float xr = 0.0f, xi = 0.0f;
                for (b = 0; b < GM_THD_WIN; b += GM_THD_BLK) {
                    float br = 0.0f, bi = 0.0f;
                    for (j = b; j < b + GM_THD_BLK; j++) {
                        const float v = w->lsb * (float)p[j];
                        br += v * gm_tw500_re[idx];
                        bi += v * gm_tw500_im[idx];
                        idx += step;
                        if (idx >= GM_THD_WIN) {
                            idx -= GM_THD_WIN;
                        }
                    }
                    xr += br;
                    xi += bi;
                }
                if (h == 1) {
                    p1 = xr * xr + xi * xi;
                } else {
                    ph += xr * xr + xi * xi;
                }
            }
            if (p1 > 0.0f) {
                sum_c += GM_DIVF(GM_SQRTF(ph), GM_SQRTF(p1));
            } else {
                ok = 0u;
            }
        }
        {
            const float t = GM_DIVF(sum_c, (float)GM_N_CH);
#ifdef GM_TRACES
            if (w->tr) {
                w->tr->thd[win] = ok ? t : GM_NANF;
            }
#endif
            if (win == 0) {
                thd_first = t;
                ok_first = ok;
            }
            if (win == GM_THD_NWIN - 1) {
                thd_last = t;
                ok_last = ok;
            }
            if (win == 0 || t > thd_max) {
                thd_max = t;
            }
            if (!ok) {
                ok_all = 0u;
            }
        }
    }
    if (!ok_all) {
        w->out->flags |= GM_FLAG_THD_INVALID;
    }
    gm_set_feature(w->out, GM_F_THD_FIRST, thd_first, ok_first);
    gm_set_feature(w->out, GM_F_THD_LAST, thd_last, ok_last);
    gm_set_feature(w->out, GM_F_THD_MAX, thd_max, ok_all);
}
