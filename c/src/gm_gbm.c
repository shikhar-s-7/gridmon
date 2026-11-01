/* GBM-embedded walker (docs/MODEL_FORMAT.md section 4) over the frozen tables c/generated/gm_model_gbm.c.
 *   x[i] = valid_i ? f_i : 0.0f   (D-040/D-051: explicit validity flag, no NaN arithmetic)
 *   raw[k] = init[k]; for t = 0..T-1 (this order): walk from tree_root[t], left iff x[feature] <= threshold;
 *   raw[t % K] += value[leaf].  label = first argmax raw;  proba = softmax(raw - max raw) (float32).
 * ADVISORY output only (D-050). */
#include "gm_internal.h"

#if GM_ENABLE_GBM
#include "gm_model_gbm.h"

#if GM_GBM_N_FEATURES != GM_N_FEATURES || GM_GBM_N_CLASSES != GM_N_CLASSES
#error "model tables do not match the feature/class layout"
#endif

void gm_gbm(gm_result_t *out)
{
    float x[GM_N_FEATURES], raw[GM_N_CLASSES], e[GM_N_CLASSES], mx, sum;
    int32_t i, t, k = 0;
    int16_t label = 0;

    for (i = 0; i < GM_N_FEATURES; i++) {
        x[i] = out->feature_valid[i] ? out->features[i] : 0.0f;
    }
    for (i = 0; i < GM_N_CLASSES; i++) {
        raw[i] = gm_gbm_init[i];
    }
    for (t = 0; t < GM_GBM_N_TREES; t++) {
        int32_t n = (int32_t)gm_gbm_tree_root[t];
        while (gm_gbm_left[n] >= 0) {
            n = (x[gm_gbm_feature[n]] <= gm_gbm_threshold[n]) ? (int32_t)gm_gbm_left[n] : (int32_t)gm_gbm_right[n];
        }
        raw[k] += gm_gbm_value[n];
        k++;
        if (k == GM_N_CLASSES) {
            k = 0;
        }
    }
    mx = raw[0];
    for (i = 1; i < GM_N_CLASSES; i++) {
        if (raw[i] > mx) {
            mx = raw[i];
            label = (int16_t)i;
        }
    }
    sum = 0.0f;
    for (i = 0; i < GM_N_CLASSES; i++) {
        e[i] = GM_EXPF(raw[i] - mx);
        sum += e[i];
    }
    for (i = 0; i < GM_N_CLASSES; i++) {
        out->gbm_raw[i] = raw[i];
        out->gbm_proba[i] = GM_DIVF(e[i], sum);
    }
    out->gbm_label = label;
}
#endif
