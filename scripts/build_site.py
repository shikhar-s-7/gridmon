"""M8: build the static website web/dist/ from web/src/ (+ exported data and the WebAssembly C core).

Steps
  1. data: scripts/export_web.py -> web/src/data/, web/src/examples/ (skipped with --skip-export, or
     automatically when the full data/ splits are missing; the committed export is then reused)
  2. WebAssembly: `python -m ziglang cc -target wasm32-wasi -mexec-model=reactor -Wl,--no-entry` over the same
     core sources as the PC SIL library + web/wasm/gm_wasm.c (GM_TRACES, GM_REPORT_NAN=1, GM_ENABLE_GBM=1),
     same warning flags (-Werror) -> web/dist/wasm/gridmon.wasm (D-075)
  3. pages: web/src/pages/*.html with build-time includes (header/footer/banner) and number tokens
     {{dotted.path|format}} resolved against window.GM_SITE (reports/*.json via export_web) -- no number is typed
     by hand; formats: f1..f4, pct0/pct1, int, sci1, signed3, ci3, raw
  4. assets: css/, js/, data/, examples/ copied; build_info.json; total size must stay <= 15 MB
  5. --check: run the WASM module in Node on the 27 golden vectors and every explorer record and compare with the
     native C core (rule label/reason, GBM label identical; features within the SIL tolerances)
     -> reports/m8_wasm_equivalence.json
Local preview: python -m http.server -d web/dist 8000

Usage: python scripts/build_site.py [--fast] [--skip-export] [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "web" / "src"
DIST = ROOT / "web" / "dist"
MAX_BYTES = 15 * 1024 * 1024
BANNER = "Simulation-only research prototype. Not validated on measured data. Not for protection or control."
REPO = "https://github.com/shikhar-s-7/gridmon"
PAGES = [("index", "Overview"), ("results", "Results"), ("explorer", "Explorer"), ("tryit", "Try it"),
         ("firmware", "Firmware"), ("about", "About")]


# ------------------------------------------------------------------------------------------- WebAssembly
def build_wasm(out: Path) -> dict[str, Any]:
    from gridmon.c_build import CORE_SOURCES, INCLUDES, WARN_FLAGS
    zig = [sys.executable, "-m", "ziglang", "cc"]
    if subprocess.run(zig + ["--version"], capture_output=True).returncode != 0:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ziglang"], check=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    srcs = [str(s) for s in CORE_SOURCES] + [str(ROOT / "web" / "wasm" / "gm_wasm.c")]
    cmd = zig + ["-target", "wasm32-wasi", "-mexec-model=reactor"] + WARN_FLAGS + ["-O2", "-DGM_TRACES", "-DGM_REPORT_NAN=1",
                                                                                  "-DGM_ENABLE_GBM=1"] + INCLUDES + \
        ["-Wl,--no-entry", "-s"] + srcs + ["-o", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("WASM build failed:\n" + (r.stderr or r.stdout)[-3000:])
    zv = subprocess.run(zig[:-1] + ["version"], capture_output=True, text=True).stdout.strip()
    return {"toolchain": f"zig {zv} cc -target wasm32-wasi -mexec-model=reactor", "bytes": out.stat().st_size,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


# ------------------------------------------------------------------------------------------- templating
def _get(data: Any, path: str) -> Any:
    cur = data
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def _fmt(v: Any, f: str) -> str:
    if v is None:
        return "—"
    if f == "raw":
        return str(v)
    if f.startswith("f"):
        return f"{float(v):.{int(f[1:])}f}"
    if f.startswith("pct"):
        return f"{100 * float(v):.{int(f[3:])}f} %"
    if f == "int":
        return f"{int(round(float(v))):,}"
    if f.startswith("sci"):
        m, e = f"{float(v):.{int(f[3:])}e}".split("e")
        return f"{m} &times; 10<sup>{int(e)}</sup>"
    if f.startswith("signed"):
        x = float(v)
        return ("+" if x >= 0 else "&minus;") + f"{abs(x):.{int(f[6:])}f}"
    if f.startswith("ci"):
        d = int(f[2:])
        return "[" + ", ".join(("+" if x >= 0 else "&minus;") + f"{abs(x):.{d}f}" for x in v) + "]"
    raise ValueError(f"unknown format {f}")


def render(page: str, title: str, site: dict[str, Any], body: str) -> str:
    def tok(m: re.Match) -> str:
        path, _, f = m.group(1).strip().partition("|")
        return _fmt(_get(site, path.strip()), (f or "raw").strip())
    body = re.sub(r"\{\{(.*?)\}\}", tok, body)
    cur = ' aria-current="page"'
    nav = "".join(f'<li><a href="{p}.html"{cur if p == page else ""}>{t}</a></li>' for p, t in PAGES)
    m = re.search(r"<!--\s*scripts:(.*?)-->", body)
    scripts = "".join(f'<script src="{s.strip() if "/" in s else "js/" + s.strip()}"></script>'
                      for s in (m.group(1).split(",") if m else []) if s.strip())
    desc = re.search(r"<!--\s*description:(.*?)-->", body)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — gridmon</title>
<meta name="description" content="{(desc.group(1).strip() if desc else 'gridmon: grid-event monitor for BESS PCS voltage records (simulation-only research prototype).')}">
<link rel="stylesheet" href="css/site.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<div class="banner" role="note"><strong>{BANNER}</strong></div>
<header class="site"><div class="wrap">
<a class="brand" href="index.html">gridmon <small>grid-event monitor</small></a>
<nav class="main" aria-label="Main"><ul>{nav}</ul></nav>
<button class="theme-toggle" type="button">Dark theme</button>
</div></header>
<main id="main"><div class="wrap">
{body}
</div></main>
<footer class="site"><div class="wrap">
<p>{BANNER}</p>
<p>gridmon Stage 1 &middot; numbers on this site are generated from the committed reports at build time
(commit {site['build']['commit']}) &middot; <a href="{REPO}">source repository</a> &middot; no tracking, no cookies, no external requests.</p>
</div></footer>
<script src="data/site_data.js"></script>
<script src="js/common.js"></script>
<script src="js/charts.js"></script>
{scripts}
</body>
</html>
"""


def load_site_data() -> dict[str, Any]:
    txt = (SRC / "data" / "site_data.js").read_text(encoding="utf-8")
    return json.loads(txt[txt.index("=") + 1:].strip().rstrip(";"))


def git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


# ------------------------------------------------------------------------------------------- equivalence check
def wasm_check(wasm: Path) -> dict[str, Any]:
    import base64

    import numpy as np
    import pandas as pd

    from gridmon.c_sil import CCore, sil_tolerance_check
    from gridmon.features import AUX_NAMES, FEATURE_NAMES
    if shutil.which("node") is None:
        return {"status": "skipped", "reason": "node not found"}
    golden = sorted((ROOT / "golden").glob("*.npz"))
    xs, lsbs, names = [], [], []
    for g in golden:
        xs.append(np.load(g)["x"].astype("<i2"))
        lsbs.append(2.0 ** -10)
        names.append(g.stem)
    for p in sorted((SRC / "data" / "explorer").glob("*.json")):
        rec = json.loads(p.read_text())
        xs.append(np.frombuffer(base64.b64decode(rec["x"]), dtype="<i2").reshape(3, 5000))
        lsbs.append(rec["lsb"])
        names.append(p.stem)
    X = np.stack(xs)
    tmp = ROOT / "build" / "wasm"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "check.bin").write_bytes(X.tobytes())
    r = subprocess.run(["node", str(ROOT / "web" / "tools" / "wasm_run.mjs"), str(wasm), str(tmp / "check.bin"), "0.0009765625"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {"status": "fail", "reason": r.stderr[-1000:]}
    W = np.array([[np.nan if v is None else v for v in json.loads(ln)["out"]] for ln in r.stdout.splitlines()])
    c = CCore().process(X)
    Fw = pd.DataFrame(np.concatenate([W[:, 13:50], W[:, 50:58]], axis=1), columns=FEATURE_NAMES + AUX_NAMES)
    tc = sil_tolerance_check(c["features"], Fw)
    over = {k: v["n_over_tol"] for k, v in tc.items() if v["n_over_tol"]}
    d = np.abs(W[:, 13:50] - c["features"][FEATURE_NAMES].to_numpy())
    res = {"n_records": int(len(X)), "n_golden": len(golden), "n_explorer": int(len(X) - len(golden)),
           "rule_label_equal": int((W[:, 2] == c["rule_label"]).sum()), "rule_reason_equal": int((W[:, 3] == c["rule_reason"]).sum()),
           "gbm_label_equal": int((W[:, 5] == c["gbm_label"]).sum()),
           "features_outside_sil_tolerance": over, "features_bit_identical_fraction": float(np.mean((d == 0) | np.isnan(d))),
           "max_abs_proba_diff": float(np.nanmax(np.abs(W[:, 6:13] - c["gbm_proba"]))),
           "nan_pattern_equal": bool(np.array_equal(np.isnan(W[:, 13:50]), ~c["feature_valid"].astype(bool)))}
    ok = (res["rule_label_equal"] == res["rule_reason_equal"] == res["gbm_label_equal"] == len(X)) and not over and res["nan_pattern_equal"]
    res["status"] = "pass" if ok else "fail"
    return res


# ------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="export from data/fast (development only)")
    ap.add_argument("--skip-export", action="store_true", help="reuse the committed web/src/data")
    ap.add_argument("--check", action="store_true", help="WASM vs native C equivalence in Node")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    full_data = all((ROOT / "data" / s / "X.npy").exists() for s in ("val", "test_iid", "test_shift"))
    if not args.skip_export and (full_data or args.fast):
        sys.path.insert(0, str(ROOT / "scripts"))
        import export_web  # noqa: E402
        export_web.main(["--fast"] if args.fast else [])
    elif not args.skip_export:
        print("full data/ splits not found: reusing the committed web/src/data (export skipped)")
    site = load_site_data()
    site["build"] = {"commit": git_commit(), "n_explorer": len(list((SRC / "data" / "explorer").glob("*.json")))}
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    wasm_info = build_wasm(DIST / "wasm" / "gridmon.wasm")
    site["build"]["wasm"] = wasm_info
    for page, title in PAGES:
        body = (SRC / "pages" / f"{page}.html").read_text(encoding="utf-8")
        (DIST / f"{page}.html").write_text(render(page, title, site, body), encoding="utf-8")
    for d in ("css", "js", "data", "examples"):
        shutil.copytree(SRC / d, DIST / d)
    (DIST / ".nojekyll").write_text("")
    data_bytes = sum(p.stat().st_size for p in (DIST / "data").rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file())
    info = {"commit": site["build"]["commit"], "wasm": wasm_info, "pages": [p for p, _ in PAGES],
            "bytes_total": total, "bytes_data": data_bytes, "n_explorer_records": len(list((DIST / "data" / "explorer").glob("*.json")))}
    (DIST / "build_info.json").write_text(json.dumps(info, indent=1))
    print(f"web/dist: {len(PAGES)} pages, {total / 1e6:.2f} MB (data {data_bytes / 1e6:.2f} MB), wasm {wasm_info['bytes'] / 1e3:.0f} KB "
          f"[{wasm_info['toolchain']}], {time.perf_counter() - t0:.1f} s")
    if total > MAX_BYTES:
        print(f"site too large: {total} > {MAX_BYTES} bytes")
        return 1
    if args.check:
        res = wasm_check(DIST / "wasm" / "gridmon.wasm")
        res["wasm"] = wasm_info
        (ROOT / "reports" / "m8_wasm_equivalence.json").write_text(json.dumps(res, indent=2))
        print(f"WASM vs native C: {res['status']} " + json.dumps({k: res.get(k) for k in ("n_records", "rule_label_equal",
                                                                                          "gbm_label_equal", "features_outside_sil_tolerance")}))
        if res["status"] == "fail":
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
