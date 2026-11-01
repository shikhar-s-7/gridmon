/* DSP_SPEC section 7: fundamental envelope at 100 Hz, detrending, spectra and the env_* features.
 *   e_mag(p) = (1/10) sum_{m=10p}^{10p+9} |V1(m)| / v1_ref,  e_ang(p) = (1/10) sum angD(m),  p = 2..99
 *   angD(m) = angU(m) - 2 pi (f_ref - 50) t_m,  angU(19) = ang(19), angU(m) = angU(m-1) + wrap(ang(m) - ang(m-1)).
 *   C note: angU(m) = ang(m) - 2 pi N(m) with the integer N(m) = sum of the wrap counts
 *   floor((ang(m) - ang(m-1) + pi)/(2 pi)) -- identical in exact arithmetic, no float32 accumulation.
 *   r(p) = e(p) - median{e(q): max(2, p-10) <= q <= min(99, p+10)}
 *   S_q(f) = (2/sum w) |sum_{i=0}^{49} w[i] r(2+10q+i) E[(f i) mod 100]|, f = 3..30 Hz, q = 0..4,
 *   on 100 r_mag (%) and degrees(r_ang). Peak statistics at the window with the largest peak:
 *   peak, f_peak (first argmax), prominence = peak / max(median_f S, 1e-12); gap = (S1 - S2)/max(S1, 1e-12)
 *   over all 5 x 28 cells; rms of the detrended envelopes; aux_t_osc_* = first p with |r| > peak/2. */
#include "gm_internal.h"

/* centred running-median detrending of e[2..99] into r[2..99] */
static void gm_detrend(gm_ctx_t *ctx, const float *e, float *r)
{
    int32_t p, q;
    for (p = GM_P_FIRST; p < GM_P; p++) {
        const int32_t lo = (p - GM_MED_HALF) > GM_P_FIRST ? (p - GM_MED_HALF) : GM_P_FIRST;
        const int32_t hi = (p + GM_MED_HALF) < (GM_P - 1) ? (p + GM_MED_HALF) : (GM_P - 1);
        for (q = lo; q <= hi; q++) {
            ctx->scratch[q - lo] = e[q];
        }
        r[p] = e[p] - gm_median(ctx->scratch, hi - lo + 1);
    }
}

/* spectra of scale_r * r and their peak statistics; writes features [i_amp, i_freq, i_prom] and aux gap */
static void gm_spectra(gm_ctx_t *ctx, gm_work_t *w, const float *r, float scale_r, uint16_t valid,
                       int32_t i_amp, int32_t i_freq, int32_t i_prom, int32_t i_gap, float *trace)
{
    float best_pk = -1.0f, best_prom = 0.0f, s1 = -1.0f, s2 = -1.0f, best_f = 0.0f;
    int32_t q, f, i;
    for (q = 0; q < GM_ENV_NWIN; q++) {
        const int32_t s0 = GM_P_FIRST + GM_ENV_HOP * q;
        float pk = -1.0f, fpk = 0.0f;
        for (f = 0; f < GM_ENV_NF; f++) {
            const int32_t fhz = GM_ENV_F_FIRST + f;
            float sr = 0.0f, si = 0.0f, v;
            int32_t idx = 0;
            for (i = 0; i < GM_ENV_WIN; i++) {
                const float a = (r[s0 + i] * scale_r) * gm_hann50[i];
                sr += a * gm_tw100_re[idx];
                si += a * gm_tw100_im[idx];
                idx += fhz;
                if (idx >= GM_ENV_FS) {
                    idx -= GM_ENV_FS;
                }
            }
            v = GM_SQRTF(sr * sr + si * si) * GM_K_SPEC_SCALE;
            ctx->spec[f] = v;
            if (trace) {
                trace[q * GM_ENV_NF + f] = v;
            }
            if (v > pk) {
                pk = v;
                fpk = (float)fhz;
            }
            if (v > s1) {
                s2 = s1;
                s1 = v;
            } else if (v > s2) {
                s2 = v;
            }
        }
        if (pk > best_pk) {
            float med = gm_median(ctx->spec, GM_ENV_NF);
            best_pk = pk;
            best_f = fpk;
            best_prom = GM_DIVF(pk, med > GM_K_TINY ? med : GM_K_TINY);
        }
    }
    gm_set_feature(w->out, i_amp, best_pk, valid);
    gm_set_feature(w->out, i_freq, best_f, valid);
    gm_set_feature(w->out, i_prom, best_prom, valid);
    gm_set_aux(w->out, i_gap, GM_DIVF(s1 - s2, s1 > GM_K_TINY ? s1 : GM_K_TINY), valid);
}

void gm_block_envelope(gm_ctx_t *ctx, gm_work_t *w)
{
    const float *a1 = ctx->a1;
    const float *ang = ctx->ang;
    const float v1_ref = w->v1_ref;
    const uint16_t mag_ok = w->v1_ref_ok;
    const float k_ramp = GM_K_TWO_PI * w->df_ref;
    float *trace_mag = 0, *trace_ang = 0;
    int32_t m, p, turns = 0, p_mag = -1, p_ang = -1;
    float ssq_mag = 0.0f, ssq_ang = 0.0f, half_mag, half_ang;

    /* envelopes: blocks p = 0, 1 contain invalid phasors (m < 19); the turn count starts at m = 19 */
    for (p = 0; p < GM_P; p++) {
        float sm = 0.0f, sa = 0.0f;
        for (m = GM_ENV_BLOCK * p; m < GM_ENV_BLOCK * (p + 1); m++) {
            if (m > GM_M_FIRST) {
                turns += (int32_t)GM_FLOORF(GM_DIVF((ang[m] - ang[m - 1]) + GM_K_PI, GM_K_TWO_PI));
            }
            if (p >= GM_P_FIRST) {
                const float ang_u = ang[m] - GM_K_TWO_PI * (float)turns;
                sm += mag_ok ? GM_DIVF(a1[m], v1_ref) : 0.0f;
                sa += ang_u - k_ramp * gm_time_of_phasor(m);
            }
        }
        ctx->e_mag[p] = sm * GM_K_BLOCK_INV;
        ctx->e_ang[p] = sa * GM_K_BLOCK_INV;
        ctx->r_mag[p] = 0.0f;
        ctx->r_ang[p] = 0.0f;
    }
    gm_detrend(ctx, ctx->e_mag, ctx->r_mag);
    gm_detrend(ctx, ctx->e_ang, ctx->r_ang);

#ifdef GM_TRACES
    if (w->tr) {
        for (p = 0; p < GM_P; p++) {
            const uint16_t ok = (uint16_t)(p >= GM_P_FIRST);
            w->tr->e_mag[p] = ok && mag_ok ? ctx->e_mag[p] : GM_NANF;
            w->tr->r_mag[p] = ok && mag_ok ? ctx->r_mag[p] : GM_NANF;
            w->tr->e_ang[p] = ok ? ctx->e_ang[p] : GM_NANF;
            w->tr->r_ang[p] = ok ? ctx->r_ang[p] : GM_NANF;
        }
        trace_mag = &w->tr->S_mag[0][0];
        trace_ang = &w->tr->S_ang[0][0];
    }
#endif
    gm_spectra(ctx, w, ctx->r_mag, GM_K_PCT, mag_ok, GM_F_ENV_MAG_AMP_PCT, GM_F_ENV_MAG_FREQ, GM_F_ENV_MAG_PROM,
               GM_A_ENV_MAG_GAP, trace_mag);
    gm_spectra(ctx, w, ctx->r_ang, GM_K_RAD2DEG, 1u, GM_F_ENV_ANG_AMP_DEG, GM_F_ENV_ANG_FREQ, GM_F_ENV_ANG_PROM,
               GM_A_ENV_ANG_GAP, trace_ang);

    for (p = GM_P_FIRST; p < GM_P; p++) {
        ssq_mag += ctx->r_mag[p] * ctx->r_mag[p];
        ssq_ang += ctx->r_ang[p] * ctx->r_ang[p];
    }
    gm_set_feature(w->out, GM_F_ENV_MAG_RMS_PCT, GM_SQRTF(GM_DIVF(ssq_mag, (float)(GM_P - GM_P_FIRST))) * GM_K_PCT, mag_ok);
    gm_set_feature(w->out, GM_F_ENV_ANG_RMS_DEG, GM_SQRTF(GM_DIVF(ssq_ang, (float)(GM_P - GM_P_FIRST))) * GM_K_RAD2DEG, 1u);

    half_mag = GM_K_HALF * w->out->features[GM_F_ENV_MAG_AMP_PCT];
    half_ang = GM_K_HALF * w->out->features[GM_F_ENV_ANG_AMP_DEG];
    for (p = GM_P_FIRST; p < GM_P; p++) {
        if (p_mag < 0 && mag_ok && GM_FABSF(ctx->r_mag[p]) * GM_K_PCT > half_mag) {
            p_mag = p;
        }
        if (p_ang < 0 && GM_FABSF(ctx->r_ang[p]) * GM_K_RAD2DEG > half_ang) {
            p_ang = p;
        }
    }
    /* envelope sample time: centre phasor 10p + 5 -> (5 (10p + 5) + 4)/5000 */
    gm_set_aux(w->out, GM_A_T_OSC_MAG, p_mag >= 0 ? gm_time_of_phasor(GM_ENV_BLOCK * p_mag + GM_ENV_BLOCK / 2) : 0.0f,
               (uint16_t)(p_mag >= 0));
    gm_set_aux(w->out, GM_A_T_OSC_ANG, p_ang >= 0 ? gm_time_of_phasor(GM_ENV_BLOCK * p_ang + GM_ENV_BLOCK / 2) : 0.0f,
               (uint16_t)(p_ang >= 0));
}
