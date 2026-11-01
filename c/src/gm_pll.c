/* DSP_SPEC section 6: DSOGI-PLL at 5 kHz (fixed w0, no FLL) and the PLL features.
 *   alpha = (v_ab - v_ca)/sqrt3, beta = v_bc; SOGI D/Q biquads (Tustin, prewarped; DF-II transposed:
 *   y = b0 x + s1; s1 = b1 x - a1 y + s2; s2 = b2 x - a2 y); positive sequence
 *   vap = (Da - Qb)/2, vbp = (Qa + Db)/2; e = (-vap sin th + vbp cos th)/max(|v+|, 0.05);
 *   PI: u += b0 e + b1 e_prev; w = 2 pi 50 + u; f_pll = w/(2 pi); th = wrap(th + w Ts).
 *   Order per sample: filters -> positive sequence -> e (with th[n]) -> PI -> f_pll[n] -> th[n+1].
 *   Initial: states 0, u = 0, e_prev = 0, th[0] = atan2f(beta[0], alpha[0]).
 * Features over OBS phasor instants n = 5m + 4 (m >= 150): dfpll_max = max |f_pll - f_ref|,
 * t_fpll_05 = 0.001 s x count(|f_pll - f_ref| > 0.5 Hz). The deviation f_pll - 50 = u/(2 pi) is
 * used (float32 resolution near 50 Hz). */
#include "gm_internal.h"

void gm_block_pll(gm_work_t *w)
{
    const int16_t *xab = w->x;
    const int16_t *xbc = w->x + GM_N_RECORD;
    const int16_t *xca = w->x + 2 * GM_N_RECORD;
    float sda1 = 0.0f, sda2 = 0.0f, sqa1 = 0.0f, sqa2 = 0.0f;
    float sdb1 = 0.0f, sdb2 = 0.0f, sqb1 = 0.0f, sqb2 = 0.0f;
    float u = 0.0f, e_prev = 0.0f, th, dfp_max = -1.0f;
    int32_t n, n_flag = 0, phase = 0;

    th = GM_ATAN2F(w->lsb * (float)xbc[0], (w->lsb * (float)xab[0] - w->lsb * (float)xca[0]) * GM_K_INV_SQRT3);
    for (n = 0; n < GM_N_RECORD; n++) {
        const float xa = (w->lsb * (float)xab[n] - w->lsb * (float)xca[n]) * GM_K_INV_SQRT3;
        const float xb = w->lsb * (float)xbc[n];
        float da, qa, db, qb, vap, vbp, mag, e, wh;
        da = GM_K_SOGI_BD0 * xa + sda1;
        sda1 = -GM_K_SOGI_A1 * da + sda2;
        sda2 = GM_K_SOGI_BD2 * xa - GM_K_SOGI_A2 * da;
        qa = GM_K_SOGI_BQ0 * xa + sqa1;
        sqa1 = (GM_K_SOGI_BQ1 * xa - GM_K_SOGI_A1 * qa) + sqa2;
        sqa2 = GM_K_SOGI_BQ2 * xa - GM_K_SOGI_A2 * qa;
        db = GM_K_SOGI_BD0 * xb + sdb1;
        sdb1 = -GM_K_SOGI_A1 * db + sdb2;
        sdb2 = GM_K_SOGI_BD2 * xb - GM_K_SOGI_A2 * db;
        qb = GM_K_SOGI_BQ0 * xb + sqb1;
        sqb1 = (GM_K_SOGI_BQ1 * xb - GM_K_SOGI_A1 * qb) + sqb2;
        sqb2 = GM_K_SOGI_BQ2 * xb - GM_K_SOGI_A2 * qb;
        vap = (da - qb) * GM_K_HALF;
        vbp = (qa + db) * GM_K_HALF;
        mag = GM_SQRTF(vap * vap + vbp * vbp);
        e = GM_DIVF(-vap * GM_SINF(th) + vbp * GM_COSF(th), mag > GM_K_PLL_VMIN ? mag : GM_K_PLL_VMIN);
        u = (u + GM_K_PI_B0 * e) + GM_K_PI_B1 * e_prev;
        e_prev = e;
        wh = GM_K_W0 + u;
        if (phase == GM_STEP - 1) { /* phasor instant n = 5m + 4 */
            const int32_t m = n / GM_STEP;
            const float fdev = u * GM_K_INV_2PI;
#ifdef GM_TRACES
            if (w->tr) {
                w->tr->f_pll_dev[m] = fdev;
            }
#endif
            if (m >= GM_M_OBS_FIRST) {
                const float d = GM_FABSF(fdev - w->df_ref);
                if (d > dfp_max) {
                    dfp_max = d;
                }
                if (d > GM_TH_FPLL_FLAG_GT) {
                    n_flag++;
                }
            }
            phase = 0;
        } else {
            phase++;
        }
        th = gm_wrap(th + wh * GM_K_TS);
    }
    gm_set_feature(w->out, GM_F_DFPLL_MAX, dfp_max, 1u);
    gm_set_feature(w->out, GM_F_T_FPLL_05, (float)n_flag * GM_K_PHASOR_T_STEP, 1u);
}
