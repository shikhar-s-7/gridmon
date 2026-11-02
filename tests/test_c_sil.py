"""M6 software-in-the-loop tests: the portable C core (ctypes, host build) vs the frozen Python reference.

Uses the golden vectors (committed) and VAL (data/ or data/fast). Sealed test splits are compared only by
scripts/run_sil.py (equivalence, reports/m6_sil.json), not here. Skipped when no C compiler is available.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gridmon import CLASSES
from gridmon.baseline_rules import classify
from gridmon.c_build import BUILD_DIR, BuildError, build_cli, find_compiler
from gridmon.export_gbm import Float32Walker, load_model
from gridmon.features import FEATURE_NAMES, extract_features
from gridmon.io import ROOT, load_baseline_config, load_waveforms, load_yaml
from gridmon.units import LSB_PU

GOLDEN = sorted((ROOT / "golden").glob("*.npz"))


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def core():
    from gridmon.c_sil import CCore
    try:
        find_compiler(install_zig=False)
        return CCore()
    except BuildError as e:
        pytest.skip(f"no C compiler: {e}")


@pytest.fixture(scope="module")
def thresholds():
    return load_yaml(ROOT / "configs" / "frozen_methods.yaml")["rules"]["thresholds"]


@pytest.fixture(scope="module")
def val_subset(fast_flag):
    """VAL records (all of data/fast/val, or every 5th record of data/val) with Python reference features."""
    X, meta = load_waveforms("val", fast=fast_flag)
    idx = np.arange(X.shape[0]) if fast_flag else np.arange(0, X.shape[0], 5)
    Xs = np.asarray(X[idx])
    Fp = extract_features(Xs, load_baseline_config()["features"], lsb=meta["lsb_pu"])
    return Xs, Fp, float(meta["lsb_pu"])


# ---------------------------------------------------------------------------------------------- tables and build
def test_tables_are_generated_and_current():
    """c/tables/gm_tables.{h,c} and c/host/gm_names.h equal a fresh generation from the frozen reference (D-054)."""
    assert _script("gen_c_tables").main(["--check"]) == 0


def test_tables_are_hex_float32_from_reference():
    from gridmon.dsp.freq import rocof_kernel
    from gridmon.dsp.pll import sogi_coeffs
    txt = (ROOT / "c" / "tables" / "gm_tables.c").read_text()
    body = txt.split("gm_rocof_c[100] = {")[1].split("};")[0]
    vals = [float.fromhex(s.strip().rstrip("f")) for s in body.split(",") if s.strip()]
    assert np.array_equal(np.array(vals, dtype=np.float32), rocof_kernel(np.float64).astype(np.float32))
    h = (ROOT / "c" / "tables" / "gm_tables.h").read_text()
    a1 = h.split("#define GM_K_SOGI_A1 (")[1].split(")")[0]
    assert np.float32(float.fromhex(a1.rstrip("f"))) == np.float32(sogi_coeffs()["a"][1])


def test_version_and_memory(core):
    v = core.version
    model_sha = (ROOT / "c" / "generated" / "gm_model_gbm.h").read_text().split('GM_GBM_MODEL_SHA256 "')[1][:16]
    tables_id = (ROOT / "c" / "tables" / "gm_tables.h").read_text().split('GM_TABLES_ID "')[1][:16]
    assert model_sha in v and tables_id in v and "gbm on" in v
    assert core.sizeof_ctx <= 32 * 1024                     # budget: <= 32 KB working state
    assert core.ctx_words * 2 == core.sizeof_ctx            # 8-bit chars on the host


def test_error_codes(core):
    # what: 0 NULL ctx, 1 NULL x, 2 uninitialised ctx, 3 wrong length, 4 bad lsb -> status*100 + out.status*10 + no-gbm
    assert [core.error_probe(w) for w in range(5)] == [111, 111, 221, 331, 441]


# ---------------------------------------------------------------------------------------------- golden vectors
@pytest.mark.parametrize("path", GOLDEN, ids=[p.stem for p in GOLDEN])
def test_golden_vector(core, path):
    """Every trace and feature within golden/README.md and FEATURES.md tolerances; rule label and reason identical."""
    from gridmon.c_sil import compare_golden
    g = dict(np.load(path, allow_pickle=False))
    c = core.traces(g["x"])
    r = compare_golden(g, c)
    bad = {k: v for k, v in r["traces"].items() if not v["ok"]}
    assert not bad, bad
    assert not r["features_over_tol"], r["features_over_tol"]
    assert r["features_nan_pattern_equal"]
    assert r["rule_label_equal"] and r["rule_reason_equal"], (c["rule_label"], str(g["rule_label"]))
    if np.isfinite(g["rule_onset"]):
        assert abs(c["rule_onset"] - float(g["rule_onset"])) <= 0.0011


def test_no_state_between_records(core):
    """A record's result does not depend on the records processed before it with the same context."""
    xs = np.stack([np.load(p)["x"] for p in GOLDEN[:6]])
    a = core.process(xs)
    b = core.process(xs[::-1].copy())
    fa = a["features_raw"]
    fb = b["features_raw"][::-1]
    assert np.array_equal(fa, fb) and np.array_equal(a["gbm_raw"], b["gbm_raw"][::-1])
    assert np.array_equal(a["rule_reason"], b["rule_reason"][::-1])


def test_no_signal_record(core, thresholds):
    """All-zero input (outside the reference's domain): no crash, status OK, NO_SIGNAL flag, |V1|-relative
    features invalid, the same rule label as the reference (balanced dip)."""
    x = np.zeros((1, 3, 5000), dtype=np.int16)
    c = core.process(x)
    assert c["status"][0] == 0 and c["flags"][0] & 0x0008
    assert not c["feature_valid"][0][FEATURE_NAMES.index("v1_min_rel")]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Fp = extract_features(x, load_baseline_config()["features"])
    assert classify(Fp, thresholds)["rule_label"][0] == c["rule_label_name"][0] == "sag_bal"


# ---------------------------------------------------------------------------------------------- VAL equivalence
def test_val_features_rules_gbm(core, val_subset, thresholds):
    """VAL: C features within tolerance (SIL gates incl. D-069 for rho; ungated rho ties bounded), NaN only where
    the reference is NaN (D-051), rule labels/reasons 100 %, GBM bit-exact on the same C features and 100 %
    end to end."""
    Xs, Fp, lsb = val_subset
    c = core.process(Xs, lsb)
    Fc = c["features"]
    from gridmon.c_sil import sil_tolerance_check
    tc = sil_tolerance_check(Fp, Fc)  # FEATURES.md gates + D-069/D-079 rho gate (urms_min_1 < 0.90)
    over = {k: v["n_over_tol"] for k, v in tc.items() if v["n_over_tol"]}
    assert over == {}, over
    assert tc["rho"].get("n_over_tol_ungated", 0) <= max(1, len(Xs) // 200)  # float32 argmin ties (D-060)
    for k in FEATURE_NAMES:
        assert np.array_equal(np.isfinite(Fp[k].to_numpy()), np.isfinite(Fc[k].to_numpy())), k
    rp = classify(Fp, thresholds)
    assert np.array_equal(rp["rule_reason"].to_numpy().astype(int), c["rule_reason"].astype(int))
    assert np.array_equal(rp["rule_label"].to_numpy(), c["rule_label_name"])
    walker = Float32Walker(load_model(ROOT / "models" / "gbm_embedded.json"))
    Xc = Fc[FEATURE_NAMES].to_numpy(dtype=np.float64)
    raw = walker.raw(Xc)
    assert np.array_equal(raw, c["gbm_raw"])                                   # bit-exact raw scores
    assert np.array_equal(np.argmax(raw, axis=1), c["gbm_label"])
    assert np.abs(walker.softmax32(raw) - c["gbm_proba"]).max() <= 1e-6
    assert (walker.predict(Fp[FEATURE_NAMES].to_numpy()) == c["gbm_label"]).mean() >= 0.995


def test_rocof_end_validity_matches_reference(core):
    """D-051: rocof_end is invalid in C exactly where the reference yields NaN (whole END window jump-masked),
    using golden-style synthetic jumps placed so that the jump mask covers m 860..999."""
    from gridmon.generator.events import quiet_normal  # noqa: F401  (generator import check)
    from gridmon.generator.nuisance import clean_pre_event
    from gridmon.generator.synth import synth_record
    pre = clean_pre_event(theta0=0.4)
    xs = np.stack([synth_record(pre, {"label": "phase_jump", "t0": t0, "dphi": np.radians(20.0), "dv1": 0.0})
                   for t0 in (0.30, 0.86, 0.90, 0.95)])
    Fp = extract_features(xs, load_baseline_config()["features"])
    c = core.process(xs)
    py_nan = ~np.isfinite(Fp["rocof_end"].to_numpy())
    c_nan = ~c["feature_valid"][:, FEATURE_NAMES.index("rocof_end")].astype(bool)
    assert py_nan.any() and not py_nan.all()
    assert np.array_equal(py_nan, c_nan)
    assert np.array_equal(c_nan, (c["flags"] & 0x0004) > 0)


def test_model_selftest_vectors(core):
    """The C walker reproduces sklearn on the self-test rows stored in the model JSON (MODEL_FORMAT section 8)."""
    m = load_model(ROOT / "models" / "gbm_embedded.json")
    X = np.array(m["selftest"]["x"], dtype=float)
    lab, proba, raw = core.gbm(X)
    assert np.array_equal(lab, np.array(m["selftest"]["sklearn_label"]))
    assert np.abs(proba - np.array(m["selftest"]["sklearn_proba"])).max() <= 1e-6
    assert np.array_equal(raw, Float32Walker(m).raw(X))


# ---------------------------------------------------------------------------------------------- CLI and sanitizer
def _golden_blob(path: Path) -> list[str]:
    labels = []
    with open(path, "wb") as fh:
        for p in GOLDEN:
            g = np.load(p, allow_pickle=False)
            fh.write(np.ascontiguousarray(g["x"], dtype="<i2").tobytes())
            labels.append(str(g["rule_label"]))
    return labels


def test_cli_json_lines(core, tmp_path):
    exe = build_cli()
    expect = _golden_blob(tmp_path / "g.bin")
    r = subprocess.run([str(exe), "--features", str(tmp_path / "g.bin")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(x) for x in r.stdout.splitlines()]
    assert [x["rule_label"] for x in rows] == expect
    assert all(x["gbm_label"] in CLASSES and abs(sum(x["gbm_proba"]) - 1) < 1e-5 for x in rows)
    assert set(rows[0]["features"]) == set(FEATURE_NAMES)
    v = subprocess.run([str(exe), "--version"], capture_output=True, text=True).stdout
    assert core.version in v


def test_sanitizer_build_runs_golden_cleanly(core, tmp_path):
    try:
        exe = build_cli(sanitize=True)
    except BuildError as e:
        pytest.skip(f"sanitizers unsupported by this toolchain: {str(e).splitlines()[-1][:120]}")
    expect = _golden_blob(tmp_path / "g.bin")
    r = subprocess.run([str(exe), str(tmp_path / "g.bin")], capture_output=True, text=True)
    assert r.returncode == 0 and not r.stderr.strip(), r.stderr[-2000:]
    assert [json.loads(x)["rule_label"] for x in r.stdout.splitlines()] == expect


def test_build_dir_is_ignored():
    assert "build/" in (ROOT / ".gitignore").read_text().split()
    assert BUILD_DIR == ROOT / "build"


def test_lsb_default_matches_reference():
    assert np.float32(LSB_PU) == np.float32(2.0 ** -10)


def test_rho_sil_gate_is_the_rules_dip_condition():
    """D-079: the rho comparison gate is exactly where the rules read rho (urms_min_1 < dip_pu = 0.90)."""
    from gridmon.c_sil import SIL_GATES_BELOW
    dip_pu = load_yaml(ROOT / "configs" / "frozen_methods.yaml")["rules"]["thresholds"]["dip_pu"]
    assert SIL_GATES_BELOW == {"rho": [("urms_min_1", dip_pu)]}
    assert dip_pu == 0.90
