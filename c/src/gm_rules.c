/* DSP_SPEC section 10: priority-ordered rule classifier (baseline_rules.py), frozen thresholds
 * (configs/frozen_methods.yaml; float32 constants with directed rounding, gm_tables.h).
 *   1 urms_min_1 < 0.90 and rho > 0.25 -> sag_unbal (2); urms_min_1 < 0.90 -> sag_bal (1)
 *   2 urms_max_3 > 1.10 -> swell (3)
 *   3 env_mag_amp_pct > 1.0 and env_mag_prom >= 4 -> oscillation (4);
 *     env_ang_amp_deg > 1.0 and env_ang_prom >= 4 -> oscillation (5)
 *   4 j_max_deg > 5 and max(|v1_min_rel - 1|, |v1_max_rel - 1|) < 0.10 -> phase_jump (6)
 *   5 |df_end| > 0.2 Hz -> freq_event (7)
 *   6 otherwise normal (0)
 * A comparison involving an invalid feature is false (the reference compares NaN as false).
 * Onset estimate = aux time of the firing detector (dip, swell, osc_mag, osc_ang, jump, freq). */
#include "gm_internal.h"

#define GM_V(i) (out->feature_valid[(i)] != 0u)
#define GM_F(i) (out->features[(i)])

void gm_rules(gm_result_t *out)
{
    int16_t reason = GM_REASON_NONE, label = GM_CLASS_NORMAL;
    int32_t onset = -1;
    const uint16_t dip = (uint16_t)(GM_V(GM_F_URMS_MIN_1) && GM_F(GM_F_URMS_MIN_1) < GM_TH_DIP_LT);
    uint16_t dv1_ok = 0u;
    float dv1 = 0.0f;

    if (GM_V(GM_F_V1_MIN_REL) && GM_V(GM_F_V1_MAX_REL)) {
        const float d0 = GM_FABSF(GM_F(GM_F_V1_MIN_REL) - 1.0f);
        const float d1 = GM_FABSF(GM_F(GM_F_V1_MAX_REL) - 1.0f);
        dv1 = d0 > d1 ? d0 : d1;
        dv1_ok = 1u;
    }
    if (dip && GM_V(GM_F_RHO) && GM_F(GM_F_RHO) > GM_TH_RHO_GT) {
        reason = GM_REASON_DIP_UNBALANCED;
        label = GM_CLASS_SAG_UNBAL;
        onset = GM_A_T_DIP;
    } else if (dip) {
        reason = GM_REASON_DIP_BALANCED;
        label = GM_CLASS_SAG_BAL;
        onset = GM_A_T_DIP;
    } else if (GM_V(GM_F_URMS_MAX_3) && GM_F(GM_F_URMS_MAX_3) > GM_TH_SWELL_GT) {
        reason = GM_REASON_SWELL;
        label = GM_CLASS_SWELL;
        onset = GM_A_T_SWELL;
    } else if (GM_V(GM_F_ENV_MAG_AMP_PCT) && GM_V(GM_F_ENV_MAG_PROM) && GM_F(GM_F_ENV_MAG_AMP_PCT) > GM_TH_OSC_MAG_GT &&
               GM_F(GM_F_ENV_MAG_PROM) >= GM_TH_OSC_PROM_GE) {
        reason = GM_REASON_OSC_MAGNITUDE;
        label = GM_CLASS_OSCILLATION;
        onset = GM_A_T_OSC_MAG;
    } else if (GM_V(GM_F_ENV_ANG_AMP_DEG) && GM_V(GM_F_ENV_ANG_PROM) && GM_F(GM_F_ENV_ANG_AMP_DEG) > GM_TH_OSC_ANG_GT &&
               GM_F(GM_F_ENV_ANG_PROM) >= GM_TH_OSC_PROM_GE) {
        reason = GM_REASON_OSC_ANGLE;
        label = GM_CLASS_OSCILLATION;
        onset = GM_A_T_OSC_ANG;
    } else if (GM_V(GM_F_J_MAX_DEG) && GM_F(GM_F_J_MAX_DEG) > GM_TH_JUMP_GT && dv1_ok && dv1 < GM_TH_JUMP_DV1_LT) {
        reason = GM_REASON_PHASE_JUMP;
        label = GM_CLASS_PHASE_JUMP;
        onset = GM_A_T_JUMP;
    } else if (GM_V(GM_F_DF_END) && GM_FABSF(GM_F(GM_F_DF_END)) > GM_TH_DF_GT) {
        reason = GM_REASON_FREQ_OFFSET;
        label = GM_CLASS_FREQ_EVENT;
        onset = GM_A_T_FREQ;
    }
    out->rule_reason = reason;
    out->rule_label = label;
    if (onset >= 0 && out->aux_valid[onset]) {
        out->onset_estimate_s = out->aux[onset];
        out->onset_valid = 1u;
    } else {
        out->onset_estimate_s = 0.0f;
        out->onset_valid = 0u;
    }
}
