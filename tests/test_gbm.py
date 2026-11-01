"""M3 tests: float32 threshold export, staged models, export/walker equivalence, size accounting,
and (if present) the committed GBM-embedded model against its sklearn pickle on VAL."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from gridmon import CLASSES
from gridmon.export_gbm import (BYTES_PER_NODE, Float32Walker, export_gbc, gbc_size, load_model, model_bytes,
                                sha256_file, threshold_check, threshold_floor_f32, write_c_tables, write_model)
from gridmon.features import FEATURE_NAMES
from gridmon.io import ROOT, data_root, load_ml_config, models_root

RNG = np.random.default_rng(np.random.SeedSequence(20261101))


def _synthetic(n: int = 840, d: int = len(FEATURE_NAMES)) -> tuple[np.ndarray, np.ndarray]:
    """7-class synthetic problem with float64 features that are NOT exact float32 values."""
    X = RNG.normal(size=(n, d)) * RNG.uniform(1e-4, 10, size=d) + RNG.uniform(-5, 5, size=d)
    W = RNG.normal(size=(d, len(CLASSES)))
    y = np.argmax(X / X.std(0) @ W + 0.5 * RNG.normal(size=(n, len(CLASSES))), axis=1)
    y[: len(CLASSES)] = np.arange(len(CLASSES))  # every class present
    return X, y


def test_threshold_floor_f32_property():
    t64 = np.concatenate([RNG.normal(size=2000) * 10 ** RNG.uniform(-6, 3, size=2000),
                          np.array([0.0, 1.0, -1.0, 0.1, 1e-30, -1e-30, 2.5, 1.0 + 1e-12])])
    t32 = threshold_floor_f32(t64)
    assert t32.dtype == np.float32
    assert np.all(t32.astype(np.float64) <= t64)
    assert np.all(np.nextafter(t32, np.float32(np.inf)).astype(np.float64) > t64)  # largest such float32
    # every float32 near the threshold branches identically
    for k in range(-3, 4):
        x = t32.copy()
        step = np.float32(np.inf) if k > 0 else np.float32(-np.inf)
        for _ in range(abs(k)):
            x = np.nextafter(x, step)
        assert np.array_equal(x.astype(np.float64) <= t64, x <= t32)
    exact = np.array([0.5, -2.25, 3.0], dtype=np.float32).astype(np.float64)
    assert np.array_equal(threshold_floor_f32(exact), exact.astype(np.float32))


def test_staged_model_equals_refit():
    """The first n stages of a fit equal a model trained with n_estimators = n (also with subsampling)."""
    X, y = _synthetic(420)
    kw = dict(max_leaf_nodes=8, learning_rate=0.2, subsample=0.8, max_depth=None, random_state=3)
    full = GradientBoostingClassifier(n_estimators=12, **kw).fit(X, y)
    short = GradientBoostingClassifier(n_estimators=5, **kw).fit(X, y)
    staged = list(full.staged_decision_function(X))[4]
    assert np.array_equal(staged, short.decision_function(X))
    assert gbc_size(full, 5)[0] == gbc_size(short)[0]


def test_export_walker_equivalence(tmp_path):
    X, y = _synthetic()
    clf = GradientBoostingClassifier(n_estimators=15, max_leaf_nodes=8, learning_rate=0.2, subsample=0.8,
                                     max_depth=None, random_state=0).fit(X[:600], y[:600])
    model = export_gbc(clf, FEATURE_NAMES, list(CLASSES), X[:50])
    sha = write_model(model, tmp_path / "m.json")
    assert sha == sha256_file(tmp_path / "m.json")
    w = Float32Walker(load_model(tmp_path / "m.json"))
    Xt = X[600:]
    p_sk, p_w = clf.predict_proba(Xt), w.predict_proba(Xt)
    assert p_w.dtype == np.float32
    assert np.abs(p_w - p_sk).max() <= 1e-5
    assert np.array_equal(w.predict(Xt), clf.predict(Xt))
    assert threshold_check(clf, Xt)["mismatches"] == 0
    # truncated export equals the staged model
    m5 = export_gbc(clf, FEATURE_NAMES, list(CLASSES), X[:50], n_stages=5)
    p5 = Float32Walker(m5).predict_proba(Xt)
    d5 = list(clf.staged_predict_proba(Xt))[4]
    assert np.abs(p5 - d5).max() <= 1e-5
    # size accounting
    assert model["size_bytes"] == model_bytes(model["n_nodes"], model["n_trees"], len(CLASSES))
    assert model["size_bytes"] == BYTES_PER_NODE * model["n_nodes"] + 2 * model["n_trees"] + 4 * len(CLASSES)
    # C tables: counts and exact float literals
    h, c = write_c_tables(model, sha, tmp_path / "c", source="test")
    src = c.read_text()
    assert f"#define GM_GBM_N_NODES {model['n_nodes']}" in h.read_text()
    body = src.split("gm_gbm_threshold[GM_GBM_N_NODES] = {")[1].split("};")[0]
    lits = [s.strip().rstrip("f") for s in body.replace("\n", "").split(",") if s.strip()]
    assert len(lits) == model["n_nodes"]
    assert np.array_equal(np.array(lits, dtype=np.float64).astype(np.float32),
                          np.array(model["nodes"]["threshold"], dtype=np.float64).astype(np.float32))


def test_walker_nan_input_maps_to_zero():
    X, y = _synthetic(300)
    clf = GradientBoostingClassifier(n_estimators=5, max_leaf_nodes=4, max_depth=None, random_state=0).fit(X, y)
    w = Float32Walker(export_gbc(clf, FEATURE_NAMES, list(CLASSES), X[:20]))
    Xn = X[:20].copy()
    Xn[:, 3] = np.nan
    Xz = X[:20].copy()
    Xz[:, 3] = 0.0
    assert np.array_equal(w.raw(Xn), w.raw(Xz))


def _committed():
    path = models_root(False) / "gbm_embedded.json"
    feats = data_root(False) / "val" / "features.csv"
    if not path.exists():
        pytest.skip("no committed models/gbm_embedded.json")
    return path, feats


def test_committed_model_format_and_budget():
    path, _ = _committed()
    m = load_model(path)
    mcfg = load_ml_config()
    assert m["feature_names"] == FEATURE_NAMES and m["classes"] == list(CLASSES)
    assert m["size_bytes"] == model_bytes(m["n_nodes"], m["n_trees"], m["n_classes"]) <= mcfg["budget_bytes"]
    assert m["n_trees"] == m["n_stages"] * m["n_classes"] == len(m["tree_root"])
    hdr = (ROOT / "c" / "generated" / "gm_model_gbm.h").read_text()
    assert f'#define GM_GBM_MODEL_SHA256 "{sha256_file(path)}"' in hdr
    # selftest rows stored in the JSON reproduce with the float32 walker
    st = m["selftest"]
    w = Float32Walker(m)
    assert np.abs(w.predict_proba(np.array(st["x"])) - np.array(st["sklearn_proba"])).max() <= 1e-3
    assert w.predict(np.array(st["x"])).tolist() == st["sklearn_label"]


def test_committed_model_equivalence_val():
    """Exported seed models vs their sklearn pickles on VAL: max |proba diff| <= 1e-3, labels 100 %."""
    import joblib
    import pandas as pd

    from gridmon.ml_gbm import model_input

    path, feats = _committed()
    if not feats.exists():
        pytest.skip("no data/val/features.csv (run scripts/run_baseline.py)")
    X = model_input(pd.read_csv(feats))
    rep = json.loads((ROOT / "reports" / "m3_gbm_val.json").read_text())
    for mdl in rep["embedded"]["models"]:
        jpath = ROOT / mdl["path"]
        assert sha256_file(jpath) == mdl["sha256"]
        clf = joblib.load(models_root(False) / mdl["joblib"])
        w = Float32Walker(load_model(jpath))
        assert np.abs(w.predict_proba(X) - clf.predict_proba(X)).max() <= 1e-3
        assert np.array_equal(w.predict(X), clf.predict(X))
        assert threshold_check(clf, X)["mismatches"] == 0
