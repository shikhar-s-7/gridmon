"""M6 software-in-the-loop: the portable C core vs the frozen Python reference (EQUIVALENCE only).

1. golden vectors: every trace, feature, rule label/reason (golden/README.md, FEATURES.md tolerances)
2. splits val, test_iid, test_shift (full) and train (every 10th record; --fast: data/fast splits):
   C vs Python features (documented tolerances; worst feature, count outside), rule-label agreement
   with every disagreement and its cause, GBM (a) C walker vs the Python float32 walker on the SAME
   C features (labels bit-exact, |proba diff| <= 1e-6), (b) end-to-end C vs Python GBM labels
3. sanitizer build (address, undefined) over all golden vectors
4. portability: static checks, -m32 and 16-bit-int compiles
5. memory/time: sizeof(gm_ctx_t), constant-table bytes, object sizes, host time per record
Writes reports[/fast]/m6_sil.json. The sealed test splits are read for equivalence only (CLAUDE.md
rule 4, PROMPT_03 A1.3): no labels are loaded and no accuracy is computed.

Usage: python scripts/run_sil.py [--fast] [--splits val test_iid ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gridmon import CLASSES
from gridmon.baseline_rules import classify
from gridmon.c_build import BUILD_DIR, build_all, compiler_version, find_compiler, object_sizes
from gridmon.c_portability import cross_compile_checks, static_checks
from gridmon.c_sil import CCore, compare_golden
from gridmon.export_gbm import Float32Walker, load_model
from gridmon.features import AUX_NAMES, FEATURE_SPEC, extract_features, tolerance_check
from gridmon.io import ROOT, load_baseline_config, load_waveforms, load_yaml, reports_root, write_json

SPLITS = ("val", "test_iid", "test_shift", "train")
SEALED_FOR_EQUIVALENCE = ("test_iid", "test_shift")
TRAIN_STRIDE = 10
TOL = {f[0]: f[2] for f in FEATURE_SPEC}


def rule_conditions(F: pd.DataFrame, th: dict) -> dict[str, np.ndarray]:
    """Boolean value of every rule comparison (NaN -> False), for disagreement causes."""
    f = {c: F[c].to_numpy(dtype=float) for c in F.columns}
    with np.errstate(invalid="ignore"):
        dv1 = np.maximum(np.abs(f["v1_min_rel"] - 1.0), np.abs(f["v1_max_rel"] - 1.0))
        return {
            "urms_min_1 < dip_pu": f["urms_min_1"] < th["dip_pu"], "rho > rho_th": f["rho"] > th["rho_th"],
            "urms_max_3 > swell_pu": f["urms_max_3"] > th["swell_pu"],
            "env_mag_amp_pct > osc_mag_pct": f["env_mag_amp_pct"] > th["osc_mag_pct"],
            "env_mag_prom >= osc_prom": f["env_mag_prom"] >= th["osc_prom"],
            "env_ang_amp_deg > osc_ang_deg": f["env_ang_amp_deg"] > th["osc_ang_deg"],
            "env_ang_prom >= osc_prom": f["env_ang_prom"] >= th["osc_prom"],
            "j_max_deg > jump_deg": f["j_max_deg"] > th["jump_deg"], "dv1 < jump_max_dv1": dv1 < th["jump_max_dv1"],
            "|df_end| > df_hz": np.abs(f["df_end"]) > th["df_hz"],
        }


COND_FEATURE = {"urms_min_1 < dip_pu": ("urms_min_1", "dip_pu"), "rho > rho_th": ("rho", "rho_th"),
                "urms_max_3 > swell_pu": ("urms_max_3", "swell_pu"),
                "env_mag_amp_pct > osc_mag_pct": ("env_mag_amp_pct", "osc_mag_pct"),
                "env_mag_prom >= osc_prom": ("env_mag_prom", "osc_prom"),
                "env_ang_amp_deg > osc_ang_deg": ("env_ang_amp_deg", "osc_ang_deg"),
                "env_ang_prom >= osc_prom": ("env_ang_prom", "osc_prom"),
                "j_max_deg > jump_deg": ("j_max_deg", "jump_deg"), "dv1 < jump_max_dv1": (None, "jump_max_dv1"),
                "|df_end| > df_hz": ("df_end", "df_hz")}


def tolerance_ratios(Fp: pd.DataFrame, Fc: pd.DataFrame) -> dict[str, float]:
    """max over compared (gated) records of |C - Python| / (tol + rel_tol |Python|); tol 0 -> 0 or inf."""
    from gridmon.features import REL_TOL, TOL_GATES
    out = {}
    for name, tol in TOL.items():
        a = Fp[name].to_numpy(dtype=float)
        d = np.abs(a - Fc[name].to_numpy(dtype=float))
        gate = np.isfinite(a)
        for col, lo in TOL_GATES.get(name, []):
            gate &= Fp[col].to_numpy(dtype=float) >= lo
        lim = tol + REL_TOL.get(name, 0.0) * np.abs(a)
        if not gate.any():
            out[name] = 0.0
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(lim[gate] > 0, d[gate] / lim[gate], np.where(d[gate] > 0, np.inf, 0.0))
        out[name] = float(np.nanmax(np.where(np.isnan(r), np.inf, r)))
    return out


def explain_outside(core: CCore, x: np.ndarray, lsb: float, feature: str, fcfg: dict) -> str:
    """Cause of one feature value outside tolerance (diagnosed from the C and Python traces)."""
    from gridmon.features import compute_traces, features_from_traces
    tr = compute_traces(x[None], lsb)
    _, ex = features_from_traces(tr, fcfg)
    ct = core.traces(x, lsb)
    if feature == "rho":
        re, im = ct["v1"].real.astype(np.float32), ct["v1"].imag.astype(np.float32)
        a32 = np.sqrt(re * re + im * im)
        mc = 150 + int(np.argmin(a32[150:]))
        mp = int(ex["m_star"][0])
        if mc != mp:
            a64 = np.abs(tr["v1"][0])
            return (f"argmin |V1| tie in float32: m* python {mp}, C {mc}; float32 |V1| equal ({a32[mc]:.8g}), "
                    f"float64 differs by {abs(a64[mp] - a64[mc]):.1e}; |V2(m*_py) - V2(m*_C)| = "
                    f"{abs(tr['v2'][0][mp] - tr['v2'][0][mc]):.1e} pu")
    return "unexplained"


def disagreement_causes(i: int, Fp: pd.DataFrame, Fc: pd.DataFrame, cp: dict, cc: dict, th: dict) -> list[dict]:
    out = []
    for name, (feat, thname) in COND_FEATURE.items():
        if bool(cp[name][i]) != bool(cc[name][i]):
            d = {"condition": name, "threshold": th[thname], "python_true": bool(cp[name][i]), "c_true": bool(cc[name][i])}
            if feat:
                a, b = float(Fp.at[i, feat]), float(Fc.at[i, feat])
                tol = TOL[feat] + (0.01 * abs(a) if feat.endswith("_prom") else 0.0)
                d.update({"feature": feat, "python": a, "c": b, "abs_diff": abs(a - b), "tolerance": tol,
                          "within_tolerance": bool(abs(a - b) <= tol),
                          "python_distance_to_threshold": abs(abs(a) - th[thname]) if feat == "df_end" else abs(a - th[thname])})
            out.append(d)
    return out


def run_golden(core: CCore) -> dict:
    files = sorted((ROOT / "golden").glob("*.npz"))
    worst: dict[str, float] = {}
    per_file, n_ok = [], 0
    feat_worst: dict[str, float] = {}
    for f in files:
        g = dict(np.load(f, allow_pickle=False))
        c = core.traces(g["x"])
        r = compare_golden(g, c)
        n_ok += bool(r["ok"])
        for k, v in r["traces"].items():
            if "max_abs_diff" in v:
                worst[k] = max(worst.get(k, 0.0), v["max_abs_diff"])
        cols = [f[0] for f in FEATURE_SPEC] + list(AUX_NAMES)
        tc = tolerance_check(pd.DataFrame([np.concatenate([g["features"], g["aux"]])], columns=cols),
                             pd.DataFrame([np.concatenate([c["features"], c["aux"]])], columns=cols))
        for name, v in tc.items():  # gated as in FEATURES.md
            feat_worst[name] = max(feat_worst.get(name, 0.0), v["max_abs_diff"])
        per_file.append({"file": f.name, "ok": bool(r["ok"]), "rule_label": c["rule_label"],
                         "rule_reason": c["rule_reason"], "failed": [k for k, v in r["traces"].items() if not v["ok"]]
                         + r["features_over_tol"]})
    return {"n_files": len(files), "n_pass": n_ok, "trace_max_abs_diff": worst,
            "trace_tolerance": {k: v for k, v in __import__("gridmon.c_sil", fromlist=["GOLDEN_TOL"]).GOLDEN_TOL.items()},
            "feature_max_abs_diff_gated": feat_worst, "files": per_file}


def run_sanitizer(info: dict) -> dict:
    exe = info.get("asan", "")
    if not exe or exe.startswith("unsupported"):
        return {"status": "skipped", "note": exe or "not built"}
    files = sorted((ROOT / "golden").glob("*.npz"))
    blob = BUILD_DIR / "golden_records.bin"
    expect = []
    with open(blob, "wb") as fh:
        for f in files:
            g = np.load(f, allow_pickle=False)
            fh.write(np.ascontiguousarray(g["x"], dtype="<i2").tobytes())
            expect.append(str(g["rule_label"]))
    r = subprocess.run([str(ROOT / exe), "--features", str(blob)], capture_output=True, text=True)
    lines = [json.loads(x) for x in r.stdout.splitlines() if x.strip()]
    labels = [x["rule_label"] for x in lines]
    ok = r.returncode == 0 and not r.stderr.strip() and labels == expect
    return {"status": "pass" if ok else "fail", "tool": "gm_cli_asan (-fsanitize=address,undefined, no recover)",
            "records": len(lines), "exit_code": r.returncode, "stderr": r.stderr.strip()[:500],
            "labels_equal_golden": labels == expect}


def run_split(core: CCore, walker: Float32Walker, split: str, fast: bool, fcfg: dict, th: dict) -> dict:
    allow = split in SEALED_FOR_EQUIVALENCE  # M6 SIL: equivalence only (CLAUDE.md rule 4)
    X, meta = load_waveforms(split, fast=fast, allow_test=allow)
    idx = np.arange(X.shape[0])
    if split == "train" and not fast:
        idx = idx[::TRAIN_STRIDE]
    X = np.asarray(X[idx])
    lsb = float(meta["lsb_pu"])
    t0 = time.perf_counter()
    Fp = extract_features(X, fcfg, lsb=lsb)
    t_py = time.perf_counter() - t0
    rp = classify(Fp, th)
    c = core.process(X, lsb)
    Fc = c["features"]
    tc = tolerance_check(Fp, Fc)
    nan_mismatch = {k: int((np.isfinite(Fp[k].to_numpy()) != np.isfinite(Fc[k].to_numpy())).sum()) for k in Fp.columns
                    if k in TOL}
    over = {k: v["n_over_tol"] for k, v in tc.items() if v["n_over_tol"]}
    outside = []
    for k in over:
        a, b = Fp[k].to_numpy(dtype=float), Fc[k].to_numpy(dtype=float)
        for i in np.where(np.abs(a - b) > TOL[k])[0]:
            outside.append({"record": int(idx[i]), "feature": k, "python": float(a[i]), "c": float(b[i]),
                            "abs_diff": float(abs(a[i] - b[i])), "tol": TOL[k], "python_rule_label": str(rp["rule_label"][i]),
                            "cause": explain_outside(core, X[i], lsb, k, fcfg)})
    rel = tolerance_ratios(Fp, Fc)
    worst = max(rel, key=rel.get)
    n = len(idx)
    py_lab = rp["rule_label"].to_numpy()
    c_lab = c["rule_label_name"]
    py_rea = rp["rule_reason"].to_numpy().astype(int)
    agree = py_lab == c_lab
    agree_reason = py_rea == c["rule_reason"].astype(int)
    cp, cc = rule_conditions(Fp, th), rule_conditions(Fc, th)
    dis = [{"record": int(idx[i]), "python": f"{py_lab[i]} ({py_rea[i]})", "c": f"{c_lab[i]} ({int(c['rule_reason'][i])})",
            "causes": disagreement_causes(i, Fp, Fc, cp, cc, th)} for i in np.where(~agree_reason)[0]]
    # GBM (a): same C features through the Python float32 walker
    Xc = Fc[[f[0] for f in FEATURE_SPEC]].to_numpy(dtype=np.float64)
    raw_py_on_c = walker.raw(Xc)
    lab_py_on_c = np.argmax(raw_py_on_c, axis=1)
    proba_py_on_c = walker.softmax32(raw_py_on_c)
    gbm_a_lab = lab_py_on_c == c["gbm_label"]
    gbm_a_dp = float(np.max(np.abs(proba_py_on_c - c["gbm_proba"])))
    gbm_a_raw = float(np.max(np.abs(raw_py_on_c - c["gbm_raw"])))
    # GBM (b): Python features end to end
    Xp = Fp[[f[0] for f in FEATURE_SPEC]].to_numpy(dtype=np.float64)
    lab_py = walker.predict(Xp)
    proba_py = walker.predict_proba(Xp)
    gbm_b = lab_py == c["gbm_label"]
    gbm_b_dis = [{"record": int(idx[i]), "python": CLASSES[int(lab_py[i])], "c": CLASSES[int(c["gbm_label"][i])],
                  "python_max_proba": float(proba_py[i].max()), "c_max_proba": float(c["gbm_proba"][i].max())}
                 for i in np.where(~gbm_b)[0]]
    return {
        "n_records": n, "subset": f"every {TRAIN_STRIDE}th record" if (split == "train" and not fast) else "all",
        "status_not_ok": int((c["status"] != 0).sum()),
        "features": {"n_features_within_tolerance_on_all_records": sum(1 for v in tc.values() if v["n_over_tol"] == 0),
                     "n_features": len(tc), "n_values_outside_tolerance": int(sum(over.values())),
                     "outside_by_feature": over, "outside_values": outside, "worst_feature_rel_to_tol": {"feature": worst, "max_abs_diff":
                                                                             tc[worst]["max_abs_diff"], "tol": tc[worst]["tol"],
                                                                             "ratio": rel[worst]},
                     "nan_pattern_mismatch": {k: v for k, v in nan_mismatch.items() if v},
                     "n_python_nonfinite": {k: int((~np.isfinite(Fp[k].to_numpy())).sum()) for k in TOL
                                            if (~np.isfinite(Fp[k].to_numpy())).any()},
                     "per_feature": tc},
        "rules": {"label_agree": int(agree.sum()), "label_agreement": float(agree.mean()),
                  "reason_agree": int(agree_reason.sum()), "disagreements": dis},
        "gbm_a_same_c_features": {"label_equal": int(gbm_a_lab.sum()), "bit_exact_labels": bool(gbm_a_lab.all()),
                                  "max_abs_proba_diff": gbm_a_dp, "max_abs_raw_diff": gbm_a_raw},
        "gbm_b_end_to_end": {"label_agree": int(gbm_b.sum()), "label_agreement": float(gbm_b.mean()),
                             "disagreements": gbm_b_dis},
        "time": {"c_us_per_record": 1e6 * c["seconds"] / n, "python_ms_per_record": 1e3 * t_py / n},
    }


def memory_report(core: CCore) -> dict:
    from gridmon.c_build import CORE_SOURCES
    import re
    th = (ROOT / "c" / "tables" / "gm_tables.h").read_text()
    tables = {m.group(1): 4 * int(m.group(2)) for m in re.finditer(r"extern const float (\w+)\[(\d+)\]", th)}
    mh = (ROOT / "c" / "generated" / "gm_model_gbm.h").read_text()
    model_bytes = int(re.search(r"GM_GBM_SIZE_BYTES (\d+)", mh).group(1))
    cc = find_compiler(install_zig=False)
    objs = object_sizes(cc)
    return {"sizeof_gm_ctx_t_bytes": core.sizeof_ctx, "gm_ctx_words16": core.ctx_words, "ctx_budget_bytes": 32768,
            "ctx_within_budget": core.sizeof_ctx <= 32768,
            "ctx_note": "record buffer (3 x 5000 int16 = 30,000 B) is the caller's; gm_result_t is "
                        f"{__import__('ctypes').sizeof(__import__('gridmon.c_sil', fromlist=['GmResult']).GmResult)} B",
            "dsp_tables_bytes": sum(tables.values()), "dsp_tables": tables, "gbm_model_bytes": model_bytes,
            "const_total_bytes": sum(tables.values()) + model_bytes,
            "host_objects_O2": objs, "host_text_total": sum(v["text"] for v in objs.values()),
            "host_object_compiler": compiler_version(cc), "n_core_sources": len(CORE_SOURCES)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="data/fast splits (all records), reports/fast")
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    args = ap.parse_args(argv)
    t_all = time.perf_counter()
    info = build_all()
    core = CCore(ROOT / info["sil"])
    print(core.version)
    rep: dict = {"version": core.version, "fast": args.fast, "build": {k: v for k, v in info.items() if k != "checks"},
                 "build_checks": json.loads(info["checks"])}
    rep["golden"] = run_golden(core)
    print(f"golden: {rep['golden']['n_pass']}/{rep['golden']['n_files']} pass")
    rep["sanitizer"] = run_sanitizer(info)
    print(f"sanitizer: {rep['sanitizer']['status']}")
    sc = static_checks()
    rep["portability"] = {"static_checks": {k: (len(v) == 0) for k, v in sc.items()},
                          "static_violations": {k: v for k, v in sc.items() if v}, "cross_compile": cross_compile_checks()}
    print("portability: static " + ("pass" if not any(sc.values()) else "FAIL") + ", " +
          ", ".join(f"{k} {v['status']}" for k, v in rep["portability"]["cross_compile"].items()))
    bcfg = load_baseline_config()
    th = load_yaml(ROOT / "configs" / "frozen_methods.yaml")["rules"]["thresholds"]
    walker = Float32Walker(load_model(ROOT / "models" / "gbm_embedded.json"))
    rep["splits"] = {}
    for split in args.splits:
        r = run_split(core, walker, split, args.fast, bcfg["features"], th)
        rep["splits"][split] = r
        f = r["features"]
        print(f"{split:>10s} n={r['n_records']:5d}: features {f['n_features_within_tolerance_on_all_records']}/37 "
              f"(outside: {f['n_values_outside_tolerance']} values, worst {f['worst_feature_rel_to_tol']['feature']} "
              f"{f['worst_feature_rel_to_tol']['ratio']:.3f} x tol); rules {r['rules']['label_agree']}/{r['n_records']}; "
              f"GBM(a) {'bit-exact' if r['gbm_a_same_c_features']['bit_exact_labels'] else 'MISMATCH'} "
              f"dp {r['gbm_a_same_c_features']['max_abs_proba_diff']:.1e}; GBM(b) {r['gbm_b_end_to_end']['label_agree']}"
              f"/{r['n_records']}; C {r['time']['c_us_per_record']:.0f} us/record")
    rep["memory"] = memory_report(core)
    rep["c_us_per_record_mean"] = float(np.mean([r["time"]["c_us_per_record"] for r in rep["splits"].values()])) \
        if rep["splits"] else None
    rep["runtime_s"] = round(time.perf_counter() - t_all, 1)
    out = reports_root(args.fast) / "m6_sil.json"
    write_json(out, rep)
    print(f"sizeof(gm_ctx_t) {rep['memory']['sizeof_gm_ctx_t_bytes']} B; const tables {rep['memory']['const_total_bytes']} B; "
          f"-> {out.relative_to(ROOT)} ({rep['runtime_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
