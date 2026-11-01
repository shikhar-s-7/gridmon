# GBM-embedded model format (gridmon-gbm-v1)

Exported by `src/gridmon/export_gbm.py` from a fitted sklearn `GradientBoostingClassifier`
(multiclass, K = 7, class k = `gridmon.CLASSES[k]`). Files: `models/gbm_embedded.json` (the model
that goes into the C core; seed 0), `models/gbm_embedded_seed<k>.json` (the other seeds, used for
mean +/- sd in M3/M4) and the generated C tables `c/generated/gm_model_gbm.{h,c}` (const arrays only).
Simulation-only; the model is a monitor classifier, never a control or protection input.

## 1. Input
37 features in the order of `feature_names` (= `gridmon.features.FEATURE_NAMES`, docs/FEATURES.md),
computed by the M2 feature code. Mapping before the walk (D-040): `x[i] = isfinite(f[i]) ? (float)f[i] : 0.0f`.
Only `rocof_end` can be non-finite in a full record (jump mask covering the whole END window).
The Python reference casts its float64 features to float32 (round to nearest), exactly as sklearn does.

## 2. Arrays (struct of arrays; all trees concatenated)
| array | C type | length | meaning |
|---|---|---|---|
| `feature` | int16_t | n_nodes | feature index of the split; -1 for a leaf |
| `threshold` | float | n_nodes | go left iff `x[feature] <= threshold`; 0 for a leaf |
| `left`, `right` | int16_t | n_nodes | GLOBAL child node indices; -1 for a leaf |
| `value` | float | n_nodes | leaf value with the learning rate folded in, `float32(lr * v_sklearn)`; 0 for internal nodes |
| `tree_root` | uint16_t | n_trees | global index of each tree's root |
| `init` | float | K | per-class initial raw score |

Tree order is stage-major: tree t belongs to boosting stage t // K and class t % K.
Limits: n_nodes <= 32767 (int16 child indices; the 64 KB budget allows at most ~4,680 nodes).

## 3. Size accounting
bytes = 14 * n_nodes + 2 * n_trees + 4 * K
(14 B/node = 2 + 4 + 2 + 2 + 4: feature, threshold, left, right, value; 2 B/tree: root index;
4 B/class: init score). A leaf stores the same 14 bytes as an internal node (fixed-stride arrays).
Budget: <= 65,536 bytes (brief B8/D-011). On C28x, int16_t is one 16-bit word and float two words,
so the table occupies bytes/2 words of flash. Grid sizes for every configuration: reports/m3_gbm_val.json.

## 4. Evaluation (what the C core must do; float32 only)
```
raw[k] = init[k]                                   for k = 0..K-1
for t = 0 .. n_trees-1:                            /* this order: float32 adds are not associative */
    n = tree_root[t]
    while (left[n] >= 0)
        n = (x[feature[n]] <= threshold[n]) ? left[n] : right[n]
    raw[t % K] += value[n]
label = first k with raw[k] == max(raw)            /* argmax of raw, not of proba */
m = max(raw); e[k] = expf(raw[k] - m); proba[k] = e[k] / sum(e)
```
Cost per record: one comparison per visited internal node (sum over trees of the path length; mean
and worst case in reports/m3_gbm_val.json), one float add per tree, K expf and K divisions. Working
RAM: x[37] + raw[7] floats = 176 bytes. No recursion, no allocation.

## 5. Threshold rounding (exactness)
sklearn trees compare `float32(x) <= t64` with a float64 threshold t64. The exporter stores
t32 = the largest float32 <= t64 (cast to float32, then one `nextafter` step towards -inf if the cast
rounded up). For every float32 x: x <= t64 implies x <= t32 (t32 is the largest float32 not above t64),
and x <= t32 implies x <= t64. So the float32 walker takes exactly sklearn's branches; the only
differences are float32 accumulation of the leaf values and float32 softmax (max |proba diff| on
VAL ~1e-7, tested). `threshold_check()` verifies the branching on every VAL record and node.

## 6. Initial scores
`init` = `decision_function(X) - lr * sum of tree predictions` (public API, constant over rows), i.e.
sklearn's prior-based init for the multinomial loss, stored as float32.

## 7. C literals
Floats are written with 9 significant digits plus an `f` suffix (round-trips to the exact float32;
checked at export). The header carries GM_GBM_MODEL_SHA256 = SHA-256 of the JSON file it came from.

## 8. JSON keys
`format`, `sklearn_version`, `estimator`, `method`, `seed`, `hyperparameters`, `trained_on`,
`git_commit`, `n_features`, `feature_names`, `n_classes`, `classes`, `n_stages`, `n_trees`, `n_nodes`,
`tree_order`, `learning_rate_folded`, `threshold_rounding`, `input_nan_fill`, `init`, `tree_root`,
`nodes{feature, threshold, left, right, value}`, `size_bytes`, `max_depth`, `sum_max_depth`,
`selftest{x, sklearn_proba, sklearn_label}` (first VAL rows, for the C SIL in Prompt 03).
