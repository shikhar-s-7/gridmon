"""C-ready export of a sklearn GradientBoostingClassifier and the float32 reference tree walker.

Format (docs/MODEL_FORMAT.md, "gridmon-gbm-v1"): all trees concatenated in stage-major order
(tree t = stage t // K, class t % K). Per node (struct of arrays, 14 bytes/node):
  int16 feature (-1 = leaf), float32 threshold, int16 left, int16 right (global node indices, -1 = leaf),
  float32 value (leaf value with the learning rate folded in: float32(lr * sklearn leaf value); 0 for
  internal nodes). Plus uint16 tree_root[T] (2 bytes/tree) and float32 init[K] (per-class init scores).
Bytes = 14 * n_nodes + 2 * n_trees + 4 * K.

Walker (mimics the C arithmetic in float32): x32 = float32(features); raw[k] = init[k]; for t = 0..T-1:
walk from tree_root[t], go left iff x32[feature] <= threshold, raw[t % K] += value[leaf] (float32 adds in
tree order); label = first argmax of raw; proba = softmax(raw - max raw) in float32. Non-finite inputs
are replaced by input_nan_fill (0.0, D-040) before the walk.

Threshold export: sklearn compares float32-cast features with float64 thresholds t64. Exporting
t32 = the LARGEST float32 <= t64 (cast, then one step towards -inf if the cast rounded up) gives
(x <= t32) == (x <= t64) for every float32 x, so the float32 walker branches exactly like sklearn.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "gridmon-gbm-v1"
BYTES_PER_NODE = 14  # int16 feature + float32 threshold + int16 left + int16 right + float32 value
BYTES_PER_TREE = 2   # uint16 root index
BYTES_PER_CLASS = 4  # float32 init score


def model_bytes(n_nodes: int, n_trees: int, n_classes: int) -> int:
    """Exported size in bytes: 14 n_nodes + 2 n_trees + 4 K."""
    return BYTES_PER_NODE * int(n_nodes) + BYTES_PER_TREE * int(n_trees) + BYTES_PER_CLASS * int(n_classes)


def threshold_floor_f32(t64: np.ndarray) -> np.ndarray:
    """Largest float32 <= t64 (elementwise): cast (round to nearest), step down where the cast rounded up."""
    t64 = np.asarray(t64, dtype=np.float64)
    t32 = t64.astype(np.float32)
    up = t32.astype(np.float64) > t64
    t32[up] = np.nextafter(t32[up], np.float32(-np.inf))
    return t32


def gbc_init_scores(clf, X: np.ndarray) -> np.ndarray:
    """Per-class init raw scores (float64) of a fitted multiclass GradientBoostingClassifier, from the
    public API: decision_function(X) - lr * sum of tree predictions (constant over rows)."""
    raw = clf.decision_function(X)
    trees = np.zeros_like(raw)
    for stage in clf.estimators_:
        for k, tree in enumerate(stage):
            trees[:, k] += tree.predict(X)
    init = raw - clf.learning_rate * trees
    if float(np.ptp(init, axis=0).max()) > 1e-9:
        raise ValueError("init scores are not constant over rows")
    return init[0]


def export_gbc(clf, feature_names: list[str], classes: list[str], X_init: np.ndarray, n_stages: int | None = None,
               extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Model dict (JSON-serialisable) of a fitted multiclass GradientBoostingClassifier.

    n_stages truncates to the first n stages (identical to a model trained with n_estimators = n).
    """
    import sklearn

    K = len(clf.classes_)
    if K < 3 or list(clf.classes_) != list(range(K)):
        raise ValueError("expects >= 3 classes encoded 0..K-1 (gridmon.CLASSES order)")
    S = clf.estimators_.shape[0] if n_stages is None else int(n_stages)
    lr = float(clf.learning_rate)
    init = gbc_init_scores(clf, X_init)  # computed from the full model; init does not depend on stages
    feat, thr, left, right, value, roots = [], [], [], [], [], []
    for s in range(S):
        for k in range(K):
            tr = clf.estimators_[s, k].tree_
            off = len(feat)
            roots.append(off)
            is_leaf = tr.children_left < 0
            feat.extend(np.where(is_leaf, -1, tr.feature).tolist())
            thr.extend(np.where(is_leaf, 0.0, threshold_floor_f32(tr.threshold).astype(np.float64)).tolist())
            left.extend(np.where(is_leaf, -1, tr.children_left + off).tolist())
            right.extend(np.where(is_leaf, -1, tr.children_right + off).tolist())
            leafv = (np.float64(lr) * tr.value[:, 0, 0]).astype(np.float32).astype(np.float64)
            value.extend(np.where(is_leaf, leafv, 0.0).tolist())
    n_nodes, n_trees = len(feat), len(roots)
    if n_nodes > 32767:
        raise ValueError(f"{n_nodes} nodes exceed int16 child indices")
    depth = _tree_depths(left, right, roots)
    model = {
        "format": FORMAT,
        "sklearn_version": sklearn.__version__,
        "estimator": "sklearn.ensemble.GradientBoostingClassifier",
        "n_features": len(feature_names), "feature_names": list(feature_names),
        "n_classes": K, "classes": list(classes),
        "n_stages": S, "n_trees": n_trees, "n_nodes": n_nodes,
        "tree_order": "stage-major: tree t belongs to stage t // n_classes and class t % n_classes",
        "learning_rate_folded": lr,
        "threshold_rounding": "largest float32 <= sklearn float64 threshold; go left iff float32(x) <= threshold",
        "input_nan_fill": 0.0,
        "init": [float(np.float32(v)) for v in init],
        "tree_root": roots,
        "nodes": {"feature": feat, "threshold": thr, "left": left, "right": right, "value": value},
        "size_bytes": model_bytes(n_nodes, n_trees, K),
        "max_depth": int(max(depth)), "sum_max_depth": int(sum(depth)),
    }
    if extra:
        model.update(extra)
    return model


def _tree_depths(left: list[int], right: list[int], roots: list[int]) -> list[int]:
    out = []
    for r in roots:
        stack, best = [(r, 0)], 0
        while stack:
            n, d = stack.pop()
            if left[n] < 0:
                best = max(best, d)
            else:
                stack.extend([(left[n], d + 1), (right[n], d + 1)])
        out.append(best)
    return out


def gbc_size(clf, n_stages: int | None = None) -> tuple[int, int, int]:
    """(n_nodes, n_trees, bytes) of the first n_stages of a fitted GradientBoostingClassifier."""
    S = clf.estimators_.shape[0] if n_stages is None else int(n_stages)
    K = clf.estimators_.shape[1]
    n_nodes = int(sum(clf.estimators_[s, k].tree_.node_count for s in range(S) for k in range(K)))
    return n_nodes, S * K, model_bytes(n_nodes, S * K, len(clf.classes_))


def hgb_size(clf) -> tuple[int, int, int]:
    """(n_nodes, n_trees, bytes) of a fitted HistGradientBoostingClassifier in the same format (accounting only;
    uses the private _predictors list, sklearn 1.9)."""
    n_nodes = int(sum(p.nodes.shape[0] for stage in clf._predictors for p in stage))
    n_trees = int(sum(len(stage) for stage in clf._predictors))
    return n_nodes, n_trees, model_bytes(n_nodes, n_trees, len(clf.classes_))


# ----------------------------------------------------------------------------------------------
# JSON I/O and the float32 reference walker
# ----------------------------------------------------------------------------------------------

def write_model(model: dict[str, Any], path: Path) -> str:
    """Write the model JSON (deterministic formatting); return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(model, separators=(",", ":"), sort_keys=False).encode("utf-8")
    path.write_bytes(blob + b"\n")
    return hashlib.sha256(blob + b"\n").hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(path: Path) -> dict[str, Any]:
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} model")
    return model


class Float32Walker:
    """Reference tree walker reading the exported model; float32 arithmetic as in the C core."""

    def __init__(self, model: dict[str, Any]):
        nd = model["nodes"]
        self.K = int(model["n_classes"])
        self.n_features = int(model["n_features"])
        self.feature = np.asarray(nd["feature"], dtype=np.int16).astype(np.int64)
        self.threshold = np.asarray(nd["threshold"], dtype=np.float64).astype(np.float32)
        self.left = np.asarray(nd["left"], dtype=np.int16).astype(np.int64)
        self.right = np.asarray(nd["right"], dtype=np.int16).astype(np.int64)
        self.value = np.asarray(nd["value"], dtype=np.float64).astype(np.float32)
        self.roots = np.asarray(model["tree_root"], dtype=np.int64)
        self.init = np.asarray(model["init"], dtype=np.float64).astype(np.float32)
        self.nan_fill = np.float32(model.get("input_nan_fill", 0.0))
        for name, arr in (("threshold", nd["threshold"]), ("value", nd["value"]), ("init", model["init"])):
            a = np.asarray(arr, dtype=np.float64)
            if not np.array_equal(a.astype(np.float32).astype(np.float64), a):
                raise ValueError(f"{name}: values are not exact float32")

    def raw(self, X: np.ndarray) -> np.ndarray:
        """float32 raw scores [N, K]."""
        x = self._input(X)
        n = x.shape[0]
        rows = np.arange(n)
        raw = np.tile(self.init, (n, 1))
        f_safe = np.maximum(self.feature, 0)
        for t, root in enumerate(self.roots):
            node = np.full(n, root, dtype=np.int64)
            while True:
                internal = self.left[node] >= 0
                if not internal.any():
                    break
                go_left = x[rows, f_safe[node]] <= self.threshold[node]
                node = np.where(internal, np.where(go_left, self.left[node], self.right[node]), node)
            k = t % self.K
            raw[:, k] = raw[:, k] + self.value[node]  # float32 + float32 in tree order
        return raw

    def _input(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X).astype(np.float32)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError("feature count mismatch")
        return np.where(np.isfinite(x), x, self.nan_fill)

    @staticmethod
    def softmax32(raw: np.ndarray) -> np.ndarray:
        r = raw.astype(np.float32)
        e = np.exp(r - r.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.softmax32(self.raw(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Class index = first argmax of the float32 raw scores."""
        return np.argmax(self.raw(X), axis=1)

    def path_lengths(self, X: np.ndarray) -> np.ndarray:
        """Number of node comparisons per record (sum over trees of the path depth) [N]."""
        x = self._input(X)
        n = x.shape[0]
        rows = np.arange(n)
        f_safe = np.maximum(self.feature, 0)
        total = np.zeros(n, dtype=np.int64)
        for root in self.roots:
            node = np.full(n, root, dtype=np.int64)
            while True:
                internal = self.left[node] >= 0
                if not internal.any():
                    break
                total += internal
                go_left = x[rows, f_safe[node]] <= self.threshold[node]
                node = np.where(internal, np.where(go_left, self.left[node], self.right[node]), node)
        return total


# ----------------------------------------------------------------------------------------------
# C tables (const arrays only)
# ----------------------------------------------------------------------------------------------

def _cfloat(v: float) -> str:
    """Decimal float literal that parses to exactly float32(v) (9 significant digits)."""
    s = format(float(np.float32(v)), ".9g")
    if not any(ch in s for ch in ".en"):
        s += ".0"
    return s + "f"


def _rows(vals: list[str], per_line: int) -> str:
    return ",\n".join("    " + ", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line))


def write_c_tables(model: dict[str, Any], model_sha256: str, out_dir: Path, source: str) -> tuple[Path, Path]:
    """Write gm_model_gbm.h / gm_model_gbm.c (const arrays only) for the C core (Prompt 03)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nd = model["nodes"]
    N, T, K, F = model["n_nodes"], model["n_trees"], model["n_classes"], model["n_features"]
    for v in nd["threshold"] + nd["value"] + model["init"]:  # literals must round-trip exactly
        if np.float32(float(_cfloat(v)[:-1])) != np.float32(v):
            raise ValueError(f"float literal does not round-trip: {v!r}")
    banner = (f"/* Generated by src/gridmon/export_gbm.py from {source} (sha256 {model_sha256[:16]}...).\n"
              f" * Format {model['format']} (docs/MODEL_FORMAT.md). Do not edit. Simulation-only.\n"
              f" * {model['n_stages']} stages x {K} classes = {T} trees, {N} nodes, {model['size_bytes']} bytes. */\n")
    h = out_dir / "gm_model_gbm.h"
    c = out_dir / "gm_model_gbm.c"
    names = "\n".join(f" *  {i:2d} {n}" for i, n in enumerate(model["feature_names"]))
    cls = "\n".join(f" *  {i} {n}" for i, n in enumerate(model["classes"]))
    h.write_text(
        banner + "#ifndef GM_MODEL_GBM_H\n#define GM_MODEL_GBM_H\n\n#include <stdint.h>\n\n"
        f"#define GM_GBM_N_FEATURES {F}\n#define GM_GBM_N_CLASSES {K}\n#define GM_GBM_N_STAGES {model['n_stages']}\n"
        f"#define GM_GBM_N_TREES {T}\n#define GM_GBM_N_NODES {N}\n#define GM_GBM_MAX_DEPTH {model['max_depth']}\n"
        f"#define GM_GBM_SIZE_BYTES {model['size_bytes']}\n#define GM_GBM_MODEL_SHA256 \"{model_sha256}\"\n\n"
        "/* Feature order (input vector index):\n" + names + " */\n"
        "/* Class order (raw score index):\n" + cls + " */\n\n"
        "/* Tree t belongs to class t % GM_GBM_N_CLASSES; go left iff x[feature] <= threshold;\n"
        " * left < 0 marks a leaf; raw[k] = init[k] + sum of leaf values (learning rate folded in). */\n"
        "extern const int16_t gm_gbm_feature[GM_GBM_N_NODES];\n"
        "extern const float gm_gbm_threshold[GM_GBM_N_NODES];\n"
        "extern const int16_t gm_gbm_left[GM_GBM_N_NODES];\n"
        "extern const int16_t gm_gbm_right[GM_GBM_N_NODES];\n"
        "extern const float gm_gbm_value[GM_GBM_N_NODES];\n"
        "extern const uint16_t gm_gbm_tree_root[GM_GBM_N_TREES];\n"
        "extern const float gm_gbm_init[GM_GBM_N_CLASSES];\n\n#endif /* GM_MODEL_GBM_H */\n",
        encoding="utf-8")
    body = [banner, '#include "gm_model_gbm.h"\n']
    body.append("const int16_t gm_gbm_feature[GM_GBM_N_NODES] = {\n" + _rows([str(v) for v in nd["feature"]], 16) + "\n};\n")
    body.append("const float gm_gbm_threshold[GM_GBM_N_NODES] = {\n" + _rows([_cfloat(v) for v in nd["threshold"]], 6) + "\n};\n")
    body.append("const int16_t gm_gbm_left[GM_GBM_N_NODES] = {\n" + _rows([str(v) for v in nd["left"]], 16) + "\n};\n")
    body.append("const int16_t gm_gbm_right[GM_GBM_N_NODES] = {\n" + _rows([str(v) for v in nd["right"]], 16) + "\n};\n")
    body.append("const float gm_gbm_value[GM_GBM_N_NODES] = {\n" + _rows([_cfloat(v) for v in nd["value"]], 6) + "\n};\n")
    body.append("const uint16_t gm_gbm_tree_root[GM_GBM_N_TREES] = {\n" + _rows([f"{v}u" for v in model["tree_root"]], 12) + "\n};\n")
    body.append("const float gm_gbm_init[GM_GBM_N_CLASSES] = {\n" + _rows([_cfloat(v) for v in model["init"]], 7) + "\n};\n")
    c.write_text("\n".join(body), encoding="utf-8")
    return h, c


def threshold_check(clf, X: np.ndarray, n_stages: int | None = None) -> dict[str, int]:
    """Compare float32-walker branching (x32 <= t32) with sklearn's (x32 <= t64) at every internal node of
    the first n_stages, for every row of X. Returns counts (mismatches must be 0)."""
    x32 = np.asarray(X).astype(np.float32).astype(np.float64)
    S = clf.estimators_.shape[0] if n_stages is None else int(n_stages)
    n_cmp = n_bad = n_down = n_nodes = 0
    for s in range(S):
        for tree in clf.estimators_[s]:
            tr = tree.tree_
            internal = tr.children_left >= 0
            t64 = tr.threshold[internal]
            t32 = threshold_floor_f32(t64).astype(np.float64)
            n_down += int((t64.astype(np.float32).astype(np.float64) > t64).sum())
            n_nodes += int(internal.sum())
            xs = x32[:, tr.feature[internal]]
            n_bad += int(((xs <= t32) != (xs <= t64)).sum())
            n_cmp += xs.size
    return {"internal_nodes": n_nodes, "thresholds_stepped_down": n_down, "comparisons": n_cmp, "mismatches": n_bad}
