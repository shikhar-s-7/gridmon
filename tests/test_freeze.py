"""M4 freeze tests: frozen-config hash check, tamper detection (hashes and git), evaluate_test refusal,
sealed-split access only from scripts/evaluate_test.py, and the M4 statistics helpers."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess

import numpy as np
import pytest
import yaml

from gridmon import frozen as fz
from gridmon.io import ROOT
from gridmon.stats import mcnemar_exact, paired_bootstrap, stratified_indices


def _frozen_or_skip() -> dict:
    if not (ROOT / fz.FROZEN_PATH).exists():
        pytest.skip("methods not frozen yet (configs/frozen_methods.yaml missing)")
    return fz.load_frozen(ROOT)


def test_frozen_config_hashes():
    """Every method file and model file matches the SHA-256 recorded at the M4 freeze."""
    frozen = _frozen_or_skip()
    assert fz.verify(ROOT, frozen, check_models=True, check_git=False) == []
    m2 = json.loads((ROOT / "reports" / "m2_baseline_val.json").read_text())
    m3 = json.loads((ROOT / "reports" / "m3_gbm_val.json").read_text())
    assert frozen["rules"]["thresholds"] == {k: float(v) for k, v in m2["thresholds"]["tuned"].items()}
    assert frozen["gbm_embedded"]["config"] == m3["embedded"]["selected"]["config"]
    assert [m["sha256"] for m in frozen["gbm_embedded"]["models"]] == [m["sha256"] for m in m3["embedded"]["models"]]
    assert frozen["gbm_embedded"]["models"][0]["path"] == "models/gbm_embedded.json"


def _copy_frozen_tree(dst) -> dict:
    frozen = _frozen_or_skip()
    files = list(frozen["method_files"]) + [fz.FROZEN_PATH] + [m["path"] for k in ("gbm_embedded", "gbm_unconstrained")
                                                             for m in frozen[k]["models"]]
    for f in files:
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dst / f)
    return frozen


def test_verify_detects_tampering(tmp_path):
    frozen = _copy_frozen_tree(tmp_path)
    assert fz.verify(tmp_path, frozen, check_git=False) == []
    feat = tmp_path / "src/gridmon/features.py"
    feat.write_text(feat.read_text() + "\n# tweak\n")
    p = fz.verify(tmp_path, frozen, check_git=False)
    assert any("features.py" in x and "changed" in x for x in p)
    (tmp_path / "src/gridmon/dsp/extra.py").write_text("x = 1\n")
    assert any("unlisted" in x for x in fz.verify(tmp_path, frozen, check_git=False))
    mdl = tmp_path / frozen["gbm_embedded"]["models"][0]["path"]
    mdl.write_bytes(mdl.read_bytes() + b" ")
    assert any("model file changed" in x for x in fz.verify(tmp_path, frozen, check_git=False))


def test_verify_git_diff_against_freeze_commit(tmp_path):
    """In a git checkout, a working tree that differs from the M4-freeze commit is refused."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    frozen = _copy_frozen_tree(tmp_path)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(git + ["add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(git + ["commit", "-q", "-m", "M4-freeze: test"], cwd=tmp_path, check=True)
    assert fz.verify(tmp_path, frozen) == []
    cfg = tmp_path / "configs/baseline.yaml"
    cfg.write_text(cfg.read_text().replace("rho_th: 0.25", "rho_th: 0.30"))
    p = fz.verify(tmp_path, frozen)
    assert any("differs from the M4-freeze commit" in x and "baseline.yaml" in x for x in p)


def test_evaluate_test_refuses_when_method_files_differ(monkeypatch):
    """evaluate_test.main returns 3 and never loads a test split if a method file differs from the freeze."""
    _frozen_or_skip()
    spec = importlib.util.spec_from_file_location("evaluate_test", ROOT / "scripts" / "evaluate_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    real = fz.file_hashes

    def tampered(root=ROOT):
        h = real(root)
        h["src/gridmon/baseline_rules.py"] = "0" * 64
        return h

    def no_loading(*a, **k):
        raise AssertionError("a sealed split was loaded despite a failed freeze check")

    monkeypatch.setattr(fz, "file_hashes", tampered)
    monkeypatch.setattr(mod, "load_waveforms", no_loading)
    monkeypatch.setattr(mod, "load_labels", no_loading)
    assert mod.main([]) == 3


def test_only_evaluate_test_unseals():
    """No module or script except scripts/evaluate_test.py passes allow_test (other than False)."""
    pat = re.compile(r"allow_test\s*=\s*([A-Za-z_]+)")
    offenders = []
    for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if rel in ("src/gridmon/io.py", "scripts/evaluate_test.py"):
            continue
        for v in pat.findall(p.read_text(encoding="utf-8")):
            if v not in ("False", "allow_test"):
                offenders.append((rel, v))
    assert offenders == []


def test_frozen_yaml_is_plain():
    frozen = _frozen_or_skip()
    txt = (ROOT / fz.FROZEN_PATH).read_text()
    assert yaml.safe_load(txt) == frozen
    plan = frozen["evaluation_plan"]
    assert plan["bootstrap_resamples"] == 2000 and plan["bootstrap_stratified_by_class"]


# ---------------------------------------------------------------------------------------------- statistics
def test_mcnemar_exact_values():
    a = np.array([True] * 6 + [True] * 10)
    b = np.array([False] * 6 + [True] * 10)
    r = mcnemar_exact(a, b)
    assert (r["a_correct_b_wrong"], r["a_wrong_b_correct"]) == (6, 0)
    assert abs(r["p_value"] - 2 * 0.5 ** 6) < 1e-12
    assert mcnemar_exact(a, a)["p_value"] == 1.0


def test_bootstrap_stratified_and_paired():
    rng = np.random.default_rng(np.random.SeedSequence(7))
    y = np.repeat(np.arange(7), 30)
    idx = stratified_indices(y, 50, rng)
    assert all(np.array_equal(np.bincount(y[r], minlength=7), np.bincount(y, minlength=7)) for r in idx)
    same = paired_bootstrap(y, [y.copy()], y.copy(), 7, 200, rng)
    assert same["diff_seed_mean"] == 0.0 and same["ci95_seed_mean"] == [0.0, 0.0] and not same["excludes_zero"]
    worse = y.copy()
    worse[:20] = (worse[:20] + 1) % 7
    better = paired_bootstrap(y, [y.copy()], worse, 7, 500, rng)
    assert better["diff_seed_mean"] > 0 and better["excludes_zero"]
