"""S2E1 tests: pre-registered configuration, interharmonic generator option (Stage 1 streams unchanged),
sealed guard for test2_*, physics subset, model remapping, hybrid, statistics, decision rule, data checks,
and (once they exist) the S2E1 freeze, refusal and results."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from gridmon import CLASSES
from gridmon import s2e1
from gridmon.features import FEATURE_NAMES
from gridmon.generator import dataset as ds
from gridmon.generator.nuisance import clean_pre_event, draw_interharmonic_freq, draw_pre_event, interharmonic_intervals
from gridmon.generator.synth import synth_record
from gridmon.generator.events import quiet_normal
from gridmon.io import ROOT, SealedSplitError, data_root, load_dataset_config, load_split, load_waveforms
from gridmon.stats import macro_f1_rows, stratified_indices
from gridmon.units import LSB_PU


@pytest.fixture(scope="module")
def s2():
    return s2e1.load_config()


@pytest.fixture(scope="module")
def dcfg():
    return s2e1.dataset_config()


# ---------------------------------------------------------------------------------------------- pre-registration
def test_splits_and_new_seeds(s2, dcfg):
    st1 = load_dataset_config()
    seeds = [v["seed"] for v in s2["splits"].values()]
    assert len(set(seeds)) == len(seeds)
    used = {v["seed"] for v in st1["splits"].values()} | set(s2["used_seeds_stage1"])
    assert not set(seeds) & used
    sp = s2["splits"]
    assert (sp["train_wide"]["seed"], sp["val_wide"]["seed"], sp["test2_shift"]["seed"], sp["test2_novel"]["seed"]) == \
        (1101, 2202, 5005, 6006)
    assert sp["train_wide"]["per_class"] == 800 and all(sp[k]["per_class"] == 250 for k in ("val_wide", "test2_shift", "test2_novel"))
    assert [k for k, v in sp.items() if v.get("sealed")] == list(s2e1.SEALED)
    # every Stage 1 key except nuisance/splits is reused unchanged
    for k in st1:
        if k not in ("nuisance", "splits"):
            assert dcfg[k] == st1[k], k


def test_nuisance_configs(dcfg):
    st1 = load_dataset_config()["nuisance"]
    assert dcfg["nuisance"]["shift"] == st1["shift"]                    # test2_shift = Stage 1 test_shift config
    w, n = dcfg["nuisance"]["wide"], dcfg["nuisance"]["novel"]
    assert w["snr_db"] == [25.0, 60.0] and w["f_pre_hz"] == [49.5, 50.5] and w["unbalance"] == [0.0, 0.025]
    assert w["harmonics_pct"] == {5: [0, 5], 7: [0, 4], 11: [0, 2], 13: [0, 2]}
    assert w["gain"] == [-0.02, 0.02] and w["offset_pu"] == [-0.02, 0.02] and w["transient"] == {"p": 0.0}
    assert {k: v for k, v in n.items() if k != "interharmonic"} == w
    ih = n["interharmonic"]
    assert ih == {"p": 0.5, "freq_hz": [70.0, 950.0], "exclude_harmonic_hz": 10.0, "amp_pct": [0.5, 2.0]}
    assert "interharmonic" not in w and "interharmonic" not in st1["shift"]


def test_physics_subset_is_a_partition(s2):
    phys, excl = s2["feature_subsets"]["physics"], s2["feature_subsets"]["physics_excluded"]
    assert len(phys) == 23 and len(excl) == 14 and not set(phys) & set(excl)
    assert sorted(phys + excl) == sorted(FEATURE_NAMES)
    for f in ("rocof_max", "t_rocof_02", "rocof_end", "env_mag_prom", "env_ang_prom", "thd_first", "thd_last", "thd_max"):
        assert f in excl
    assert s2e1.feature_columns("physics", s2) == [FEATURE_NAMES.index(f) for f in phys]
    assert s2e1.feature_columns("all37", s2) == list(range(37))
    assert s2["hybrid"]["threshold"] == 0.5 and s2["evaluation"]["bootstrap"]["resamples"] == 2000


# ---------------------------------------------------------------------------------------------- generator option
def test_interharmonic_intervals_and_draws():
    iv = interharmonic_intervals(70.0, 950.0, 50.0, 10.0)
    assert iv[0] == (70.0, 90.0) and iv[-1] == (910.0, 940.0) and sum(b - a for a, b in iv) == 530.0
    icfg = {"freq_hz": [70.0, 950.0], "exclude_harmonic_hz": 10.0}
    f = np.array([draw_interharmonic_freq(np.random.default_rng(k), icfg) for k in range(3000)])
    assert f.min() >= 70.0 and f.max() <= 940.0
    assert (np.abs(f - 50 * np.round(f / 50)) >= 10.0).all()
    # uniform over the allowed set: share in [70, 90) close to 20/530
    assert abs((f < 90).mean() - 20 / 530) < 0.015


def test_stage1_draws_unchanged(dcfg):
    """Configurations without 'interharmonic' draw exactly as before (no extra draw, no manifest columns)."""
    nuis = {**load_dataset_config()["nuisance"], **dcfg["nuisance"]}
    for nz in ("iid", "shift", "wide"):
        p = draw_pre_event(np.random.default_rng(11), nuis[nz])
        assert p.ih_on is None and not any(k.startswith("ih_") for k in p.to_record())
    a = draw_pre_event(np.random.default_rng(5), dcfg["nuisance"]["wide"])
    b = draw_pre_event(np.random.default_rng(5), dcfg["nuisance"]["novel"])
    # novel = wide + draws appended after every Stage 1 draw: identical common fields
    assert all(getattr(a, k) == getattr(b, k) for k in ("u_pre", "unb", "f_pre", "snr_db", "gain", "offset", "theta0"))
    assert b.ih_on in (True, False)


def test_interharmonic_waveform_amplitude_and_sequence():
    pre = clean_pre_event(ih_on=True, ih_f=123.0, ih_amp=0.015, ih_phase=0.3, u_pre=1.02)
    x = synth_record(pre, quiet_normal()).astype(float) * LSB_PU
    X = np.fft.rfft(x, axis=1) / x.shape[1] * 2
    f = np.fft.rfftfreq(x.shape[1], 1 / 5000)
    i, j = np.argmin(np.abs(f - 123.0)), np.argmin(np.abs(f - 50.0))
    assert np.allclose(np.abs(X[:, i]) / np.abs(X[:, j]), 0.015, atol=2e-4)  # % of the fundamental, every LL channel
    assert abs(np.degrees(np.angle(X[1, i] / X[0, i])) + 120.0) < 0.5        # positive sequence
    off = synth_record(clean_pre_event(u_pre=1.02), quiet_normal()).astype(float) * LSB_PU
    assert np.abs(np.fft.rfft(off, axis=1)[:, i]).max() / x.shape[1] * 2 < 1e-3


def test_novel_specs_respect_margins(dcfg):
    """A few novel-config records per class: margins hold, manifest carries the interharmonic columns."""
    c = copy.deepcopy(dcfg)
    rows = []
    seeds, _ = ds.split_seeds(424242, 28)
    for i, (lab, sd) in enumerate(zip([cl for cl in CLASSES for _ in range(4)], seeds)):
        spec = ds.make_record_spec(i, lab, sd, c, "novel")
        r = {"label": lab, "n_reject": spec.n_reject, **spec.pre.to_record(), **spec.metrics}
        r.update({k: v for k, v in ds.event_to_record(spec.ev).items()})
        rows.append(r)
    man = pd.DataFrame(rows)
    assert s2e1.verify_margins(man, c["margins"], int(c["max_redraws"]))["all_hold_except_d039_fallback"]
    ih = s2e1.verify_interharmonic(man, c["nuisance"]["novel"]["interharmonic"])
    assert ih["configured"] and ih["f_in_range"] and ih["harmonic_exclusion_holds"]


# ---------------------------------------------------------------------------------------------- sealing
def test_test2_splits_are_sealed():
    for split in s2e1.SEALED:
        with pytest.raises(SealedSplitError):
            load_split(split)
        with pytest.raises(SealedSplitError):
            load_waveforms(split)


# ---------------------------------------------------------------------------------------------- arms
def test_hybrid_rule():
    ml = np.array(["sag_bal", "normal", "swell"])
    p = np.array([[0.1, 0.6, 0.3], [0.5, 0.25, 0.25], [0.4, 0.3, 0.3]])
    rules = np.array(["normal", "normal", "phase_jump"])
    assert s2e1.hybrid(ml, p, rules, 0.5).tolist() == ["sag_bal", "normal", "phase_jump"]


def test_remapped_subset_model_equals_sklearn():
    from sklearn.ensemble import GradientBoostingClassifier
    from gridmon.export_gbm import Float32Walker, export_gbc
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 37))
    y = (X[:, 3] > 0).astype(int) + (X[:, 20] > 0.5).astype(int) + 0 * X[:, 0].astype(int)
    cols = [3, 7, 20, 30]
    clf = GradientBoostingClassifier(n_estimators=5, max_leaf_nodes=4, random_state=0).fit(X[:, cols], y)
    m = export_gbc(clf, [FEATURE_NAMES[c] for c in cols], ["a", "b", "c"], X[:50, cols])
    full = s2e1.remap_model_features(m, cols)
    w = Float32Walker(full)
    assert full["n_features"] == 37 and full["size_bytes"] == m["size_bytes"]
    assert set(f for f in full["nodes"]["feature"] if f >= 0) <= set(cols)
    assert np.array_equal(w.predict(X), clf.predict(X[:, cols]))
    assert np.abs(w.predict_proba(X) - clf.predict_proba(X[:, cols])).max() < 1e-4


# ---------------------------------------------------------------------------------------------- statistics
def test_vectorised_bootstrap_f1_matches_stage1_helper():
    rng = np.random.default_rng(3)
    y = np.repeat(np.arange(7), 30)
    p = np.where(rng.random(len(y)) < 0.8, y, rng.integers(0, 7, len(y)))
    idx = stratified_indices(y, 50, np.random.default_rng(1))
    assert np.allclose(s2e1.macro_f1_boot(y, p, idx), macro_f1_rows(y, p, idx, 7))


def test_decision_rule_and_survival():
    assert s2e1.decision({"a": [0.01, 0.02], "b": [0.001, 0.03]}) == "ML better"
    assert s2e1.decision({"a": [-0.01, 0.02], "b": [0.001, 0.03]}) == "ML matches"
    assert s2e1.decision({"a": [-0.03, -0.01], "b": [0.01, 0.03]}) == "rules better"
    y = np.repeat(np.arange(7), 100)
    mask = np.tile([True, False], 350)
    perfect = [y.copy()]
    assert s2e1.survival(y, perfect, mask, 200, np.random.default_rng(0), 0.03)["verdict"] == "survives"
    bad = y.copy()
    bad[mask & (np.arange(len(y)) % 4 == 0)] = 0  # half of the masked records wrong
    assert s2e1.survival(y, [bad], mask, 200, np.random.default_rng(0), 0.03)["verdict"] == "fails"


# ---------------------------------------------------------------------------------------------- data (if generated)
def _summary():
    for fast in (False, True):
        p = ROOT / "reports" / ("fast" if fast else "") / "s2e1_data_summary.json"
        if p.exists() and (data_root(fast) / "train_wide" / "manifest.csv").exists():
            return json.loads(p.read_text())
    pytest.skip("S2E1 data not generated (scripts/s2e1_make_data.py [--fast])")


def test_generated_data_margins_and_nuisances(s2):
    summ = _summary()
    for name, v in summ["splits"].items():
        assert v["margins"]["all_hold_except_d039_fallback"], name
        cfg = s2["splits"][name]
        assert v["split_seed"] == cfg["seed"] and v["nuisance_config"] == cfg["nuisance"]
        if cfg["nuisance"] in ("wide", "novel"):
            assert v["n_transient"] == 0 and v["snr_db_range"][0] >= 25.0
        if cfg["nuisance"] == "novel":
            ih = v["interharmonic"]
            assert ih["f_in_range"] and ih["harmonic_exclusion_holds"] and 0.35 < ih["share_on"] < 0.65
            assert ih["amp_pct_min"] >= 0.5 and ih["amp_pct_max"] <= 2.0


# ---------------------------------------------------------------------------------------------- freeze, refusal, results
def test_s2e1_evaluate_refuses_without_valid_freeze(monkeypatch):
    """s2e1_evaluate.main returns 3 and never loads a sealed split if the S2E1 freeze (or its absence) fails."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s2e1_evaluate", ROOT / "scripts" / "s2e1_evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    def no_loading(*a, **k):
        raise AssertionError("a sealed split was loaded despite a failed freeze check")

    if (ROOT / s2e1.S2E1_FROZEN).exists():
        real = s2e1.load_frozen

        def tampered(root=ROOT):
            f = copy.deepcopy(real(root))
            f["arms"]["GW"]["models"][0]["sha256"] = "0" * 64
            return f
        monkeypatch.setattr(s2e1, "load_frozen", tampered)
    monkeypatch.setattr(mod, "load_waveforms", no_loading)
    monkeypatch.setattr(mod, "load_labels", no_loading)
    assert mod.main([]) == 3


def test_s2e1_freeze_verifies():
    if not (ROOT / s2e1.S2E1_FROZEN).exists():
        pytest.skip("S2E1 not frozen yet")
    frozen = s2e1.load_frozen()
    assert s2e1.verify(ROOT, frozen, check_models=True, check_git=False) == []
    tr = json.loads((ROOT / "reports" / "s2e1_train.json").read_text())
    for arm in s2e1.TRAINED_ARMS:
        assert frozen["arms"][arm]["config"] == tr["arms"][arm]["selected"]["config"]
        assert [m["sha256"] for m in frozen["arms"][arm]["models"]] == [m["sha256"] for m in tr["arms"][arm]["models"]]
        assert all(m["size_bytes"] <= 65536 for m in frozen["arms"][arm]["models"])
        assert tr["arms"][arm]["pre_event_only_control"]["pass"]
        assert all(e["pass"] for e in tr["arms"][arm]["equivalence_val_wide"])
    assert frozen["hybrid"]["threshold"] == 0.5
    assert frozen["arms"]["GP"]["features"] == s2e1.load_config()["feature_subsets"]["physics"]


def test_s2e1_results_consistent():
    p = ROOT / "reports" / "s2e1_results.json"
    if not p.exists():
        pytest.skip("S2E1 not evaluated yet")
    res = json.loads(p.read_text())
    assert not res["dry_run"] and not res["fast"]
    for arm, v in res["decision"].items():
        assert v["verdict"] == s2e1.decision(v["ci95_by_set"])
        for s, ci in v["ci95_by_set"].items():
            assert ci == res["per_set"][s]["comparisons"][f"{arm}-R"]["ci95_seed_mean"]
    for name in s2e1.SEALED:
        d = res["per_set"][name]
        assert d["n"] == 1750 and d["data_meta"]["split"] == name
        assert d["arms"]["R"]["n_seeds"] == 1 and d["arms"]["GW"]["n_seeds"] == 3
    md = (ROOT / "reports" / "S2E1_RESULTS.md").read_text()
    assert len(md.splitlines()) <= 150
    for f in res["figures"]:
        assert (ROOT / "reports" / "figures" / f).exists()
