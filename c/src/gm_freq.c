/* DSP_SPEC sections 3-5 and the frequency features (FEATURES.md). All frequencies are deviations
 * from 50 Hz (float32 resolution).
 *   df_raw(m) = wrap(ang(m) - ang(m-20)) / (2 pi 0.02)            m >= 39
 *   df(m)     = (1/20) sum_{j=0}^{19} df_raw(m-j)  (direct sum)   m >= 58   (f_dft = 50 + df)
 *   ROCOF(m)  = sum_{j=0}^{99} c_j df(m-99+j)  (direct 100 taps)  m >= 157
 *   J(m)      = wrap(ang(m) - 2 ang(m-20) + ang(m-40))            m >= 59
 * f_ref = 50 + median df over REF (m 58..149). Jump: mJ = first argmax_OBS |J|, detected if
 * |J(mJ)| > 3 deg; f-mask m in [mJ-40, mJ+40], ROCOF-mask m in [mJ-40, mJ+139]; masked samples are
 * excluded from df_end, df_max, rocof_max, t_rocof_02, rocof_end (D-051: rocof_end invalid when the
 * whole END window is masked, exactly where the reference yields NaN). */
#include "gm_internal.h"

static inline float gm_jump_metric(const float *ang, int32_t m)
{
    return gm_wrap((ang[m] - 2.0f * ang[m - GM_LAG]) + ang[m - 2 * GM_LAG]);
}

static inline float gm_rocof_at(const float *df, int32_t m)
{
    const float *p = df + (m - (GM_ROCOF_WIN - 1));
    float s = 0.0f;
    int32_t j;
    for (j = 0; j < GM_ROCOF_WIN; j++) {
        s += gm_rocof_c[j] * p[j];
    }
    return s;
}

void gm_block_frequency(gm_ctx_t *ctx, gm_work_t *w)
{
    float *df = ctx->df;
    const float *ang = ctx->ang;
    gm_result_t *out = w->out;
    int32_t m, j, mj = GM_M_OBS_FIRST, first_lobe = -1, m_onset = -1, n_end = 0;
    int32_t lo_f, hi_f, hi_r, n_rocof_flag = 0, n_rocof_end = 0;
    float jmax = -1.0f, jmax_deg, df_ref, s, end_sum = 0.0f, df_max = -1.0f, rocof_max = -1.0f, rocof_end_sum = 0.0f;
    uint16_t detected;

    /* df_raw into df[39..999], then the 20-term moving average in place (descending m: reads only
     * indices <= m that still hold df_raw) */
    for (m = 0; m < GM_M_DFRAW_FIRST; m++) {
        df[m] = 0.0f;
    }
    for (m = GM_M_DFRAW_FIRST; m < GM_M; m++) {
        df[m] = gm_wrap(ang[m] - ang[m - GM_LAG]) * GM_K_DF_SCALE;
    }
    for (m = GM_M - 1; m >= GM_M_FDFT_FIRST; m--) {
        s = 0.0f;
        for (j = GM_MA - 1; j >= 0; j--) {
            s += df[m - j];
        }
        df[m] = s * GM_K_MA_INV;
    }
    for (m = GM_M_DFRAW_FIRST; m < GM_M_FDFT_FIRST; m++) {
        df[m] = 0.0f; /* not valid as f_dft */
    }

    /* f_ref */
    for (m = GM_M_FDFT_FIRST; m <= GM_M_REF_LAST; m++) {
        ctx->scratch[m - GM_M_FDFT_FIRST] = df[m];
    }
    df_ref = gm_median(ctx->scratch, GM_N_FREF);
    w->df_ref = df_ref;
    gm_set_feature(out, GM_F_F_REF, GM_K_F_NOM + df_ref, 1u);

    /* jump metric over OBS: max, first argmax */
    for (m = GM_M_OBS_FIRST; m < GM_M; m++) {
        float a = GM_FABSF(gm_jump_metric(ang, m));
        if (a > jmax) {
            jmax = a;
            mj = m;
        }
    }
    jmax_deg = jmax * GM_K_RAD2DEG;
    w->jmax = jmax;
    w->m_jump = mj;
    gm_set_feature(out, GM_F_J_MAX_DEG, jmax_deg, 1u);
    gm_set_feature(out, GM_F_J_SIGN, 0.0f, 1u);
    for (m = GM_M_OBS_FIRST; m < GM_M && first_lobe < 0; m++) {
        float J = gm_jump_metric(ang, m);
        if (GM_FABSF(J) >= GM_K_HALF * jmax) {
            first_lobe = m;
            gm_set_feature(out, GM_F_J_SIGN, J > 0.0f ? 1.0f : (J < 0.0f ? -1.0f : 0.0f), 1u);
        }
    }
    detected = (uint16_t)(jmax_deg > GM_TH_JUMP_DETECT_DEG_GT);
    w->jump_detected = detected;
    if (detected) {
        out->flags |= GM_FLAG_JUMP_DETECTED;
        for (m = GM_M_OBS_FIRST; m < GM_M; m++) {
            if (GM_FABSF(gm_jump_metric(ang, m)) > GM_TH_JUMP_ONSET_RAD_GT) {
                m_onset = m;
                break;
            }
        }
    }
    gm_set_aux(out, GM_A_T_JUMP, m_onset >= 0 ? gm_time_of_phasor(m_onset) : 0.0f, (uint16_t)(m_onset >= 0));
    /* masks (empty when no jump is detected) */
    lo_f = detected ? mj - GM_JUMP_MASK_HALF : GM_M;
    hi_f = detected ? mj + GM_JUMP_MASK_HALF : -1;
    hi_r = detected ? mj + GM_JUMP_MASK_HALF + GM_ROCOF_WIN - 1 : -1;

    /* frequency features over OBS (f-masked) */
    m_onset = -1;
    for (m = GM_M_OBS_FIRST; m < GM_M; m++) {
        float d, a;
        if (m >= lo_f && m <= hi_f) {
            continue;
        }
        d = df[m] - df_ref;
        a = GM_FABSF(d);
        if (a > df_max) {
            df_max = a;
        }
        if (m_onset < 0 && a > GM_TH_FREQ_ONSET_GT) {
            m_onset = m;
        }
        if (m >= GM_M_END_FIRST) {
            end_sum += d;
            n_end++;
        }
    }
    if (n_end == 0) { /* whole END masked: unmasked mean (reference fallback) */
        out->flags |= GM_FLAG_DF_END_FALLBACK;
        for (m = GM_M_END_FIRST; m < GM_M; m++) {
            end_sum += df[m] - df_ref;
        }
        n_end = GM_N_END;
    }
    gm_set_feature(out, GM_F_DF_END, GM_DIVF(end_sum, (float)n_end), 1u);
    gm_set_feature(out, GM_F_DF_MAX, df_max, 1u);
    gm_set_aux(out, GM_A_T_FREQ, m_onset >= 0 ? gm_time_of_phasor(m_onset) : 0.0f, (uint16_t)(m_onset >= 0));

    /* ROCOF over OBS (valid m >= 157; windows touching the f-mask excluded) */
    for (m = GM_M_ROCOF_FIRST; m < GM_M; m++) {
        float r, a;
        if (m < GM_M_OBS_FIRST || (m >= lo_f && m <= hi_r)) {
            continue;
        }
        r = gm_rocof_at(df, m);
        a = GM_FABSF(r);
        if (a > rocof_max) {
            rocof_max = a;
        }
        if (a > GM_TH_ROCOF_FLAG_GT) {
            n_rocof_flag++;
        }
        if (m >= GM_M_END_FIRST) {
            rocof_end_sum += r;
            n_rocof_end++;
        }
    }
    gm_set_feature(out, GM_F_ROCOF_MAX, rocof_max, 1u);
    gm_set_feature(out, GM_F_T_ROCOF_02, (float)n_rocof_flag * GM_K_PHASOR_T_STEP, 1u);
    if (n_rocof_end == 0) {
        out->flags |= GM_FLAG_ROCOF_END_NONE;
    }
    gm_set_feature(out, GM_F_ROCOF_END, n_rocof_end > 0 ? GM_DIVF(rocof_end_sum, (float)n_rocof_end) : 0.0f,
                   (uint16_t)(n_rocof_end > 0));

#ifdef GM_TRACES
    if (w->tr) {
        for (m = 0; m < GM_M; m++) {
            w->tr->df_dft[m] = m >= GM_M_FDFT_FIRST ? df[m] : GM_NANF;
            w->tr->J[m] = m >= GM_M_J_FIRST ? gm_jump_metric(ang, m) : GM_NANF;
            w->tr->rocof[m] = m >= GM_M_ROCOF_FIRST ? gm_rocof_at(df, m) : GM_NANF;
            w->tr->fmask[m] = (int16_t)(m >= lo_f && m <= hi_f);
            w->tr->rmask[m] = (int16_t)(m >= lo_f && m <= hi_r);
        }
    }
#endif
}
