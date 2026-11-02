"""M8 website: WebAssembly C core equivalence (Node), data provenance, page checks and the explorer set.

The WASM test needs `python -m ziglang` and `node`; it is skipped with the reason otherwise. Page checks run on the
committed web/dist (built by scripts/build_site.py)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gridmon import CLASSES
from gridmon.io import ROOT

DIST = ROOT / "web" / "dist"
SRC = ROOT / "web" / "src"
BANNER = "Simulation-only research prototype. Not validated on measured data. Not for protection or control."


def _site_data() -> dict:
    txt = (SRC / "data" / "site_data.js").read_text(encoding="utf-8")
    return json.loads(txt[txt.index("=") + 1:].strip().rstrip(";"))


def _build_site_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def wasm(tmp_path_factory):
    if shutil.which("node") is None:
        pytest.skip("node not available (WASM equivalence needs a WebAssembly runtime)")
    if subprocess.run([sys.executable, "-m", "ziglang", "version"], capture_output=True).returncode != 0:
        pytest.skip("ziglang not installed (python -m pip install ziglang)")
    out = tmp_path_factory.mktemp("wasm") / "gridmon.wasm"
    info = _build_site_module().build_wasm(out)
    return out, info


def test_wasm_has_no_imports(wasm):
    out, info = wasm
    r = subprocess.run(["node", "-e", f"const m=new WebAssembly.Module(require('fs').readFileSync('{out.as_posix()}'));"
                        "console.log(JSON.stringify(WebAssembly.Module.imports(m)))"], capture_output=True, text=True)
    assert json.loads(r.stdout) == []
    assert info["bytes"] < 200_000


def test_wasm_matches_native_c_on_golden(wasm, tmp_path):
    """Rule labels/reasons and GBM labels identical to the native C core; features within the SIL tolerances."""
    import pandas as pd

    from gridmon.c_sil import CCore, sil_tolerance_check
    from gridmon.features import AUX_NAMES, FEATURE_NAMES
    out, _ = wasm
    X = np.stack([np.load(p)["x"] for p in sorted((ROOT / "golden").glob("*.npz"))]).astype("<i2")
    (tmp_path / "g.bin").write_bytes(X.tobytes())
    r = subprocess.run(["node", str(ROOT / "web" / "tools" / "wasm_run.mjs"), str(out), str(tmp_path / "g.bin")],
                       capture_output=True, text=True, check=True)
    W = np.array([[np.nan if v is None else v for v in json.loads(ln)["out"]] for ln in r.stdout.splitlines()])
    c = CCore().process(X)
    assert np.array_equal(W[:, 2], c["rule_label"]) and np.array_equal(W[:, 3], c["rule_reason"])
    assert np.array_equal(W[:, 5], c["gbm_label"])
    Fw = pd.DataFrame(np.concatenate([W[:, 13:50], W[:, 50:58]], axis=1), columns=FEATURE_NAMES + AUX_NAMES)
    tc = sil_tolerance_check(c["features"], Fw)
    assert {k: v["n_over_tol"] for k, v in tc.items() if v["n_over_tol"]} == {}
    assert np.abs(W[:, 6:13] - c["gbm_proba"]).max() <= 1e-6


def test_site_numbers_come_from_reports():
    sd = _site_data()
    r = json.loads((ROOT / "reports" / "results.json").read_text())
    for split in ("test_iid", "test_shift"):
        for m in ("rules", "gbm_embedded", "gbm_unconstrained"):
            a = sd["results"]["splits"][split]["methods"][m]["macro_f1"]["mean"]
            assert a == pytest.approx(r["per_split"][split]["methods"][m]["macro_f1"]["mean"], rel=1e-5)
            assert sd["results"]["splits"][split]["methods"][m]["confusion"]["matrix"] == \
                r["per_split"][split]["methods"][m]["primary"]["confusion"]["matrix"]
    m6 = json.loads((ROOT / "reports" / "m6_sil.json").read_text())
    assert sd["sil"]["memory"]["sizeof_gm_ctx_t_bytes"] == m6["memory"]["sizeof_gm_ctx_t_bytes"]
    assert [x["level"] for x in sd["ladder"]] == [f"L{i}" for i in range(7)]


def test_built_pages():
    if not (DIST / "index.html").exists():
        pytest.skip("web/dist not built (python scripts/build_site.py)")
    pages = sorted(DIST.glob("*.html"))
    assert {p.stem for p in pages} == {"index", "results", "explorer", "tryit", "firmware", "about"}
    for p in pages:
        html = p.read_text(encoding="utf-8")
        assert BANNER in html and "{{" not in html, p.name
        # no external scripts/styles/images at runtime (links to the repo are plain anchors)
        assert not re.search(r'<(script|link|img)[^>]+(src|href)="https?://', html), p.name
    total = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
    assert total <= 15 * 1024 * 1024
    assert (DIST / "wasm" / "gridmon.wasm").exists() and (DIST / ".nojekyll").exists()


def test_explorer_set():
    idx_txt = (SRC / "data" / "explorer_index.js").read_text(encoding="utf-8")
    idx = json.loads(idx_txt[idx_txt.index("=") + 1:].strip().rstrip(";"))
    assert len(idx) == 70
    assert {c: sum(r["true"] == c for r in idx) for c in CLASSES} == {c: 10 for c in CLASSES}
    assert sum(r["rule_label"] != r["gbm_label"] for r in idx) <= 20
    figure = {e["record"] for e in json.loads((ROOT / "reports" / "results.json").read_text())["disagreement_examples_test_shift"]}
    assert figure <= {r["record"] for r in idx if r["split"] == "test_shift"}
    size = sum(p.stat().st_size for p in (SRC / "data" / "explorer").glob("*.json"))
    assert size <= 8 * 1024 * 1024
