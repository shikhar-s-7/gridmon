"""M8: export the website data (web/src/data/) from committed reports and the frozen reference.

Writes
  web/src/data/site_data.js      window.GM_SITE = {...}: results subset (reports/results.json), size-vs-F1 grid
                                 (reports/m3_gbm_val.json), M2 budget, C core / SIL (reports/m6_sil.json), port
                                 stream (reports/m7_port_stream.json), C28x sizes (reports/m7_c28x_size.json, if
                                 present), validation ladder (docs/VALIDATION.md), thresholds, names
  web/src/data/explorer_index.js window.GM_EXPLORER = [...] (record list for the explorer)
  web/src/data/explorer/<id>.json one file per explorer record: int16 waveform (base64), lsb, plot traces (float32
                                 base64), 37 features + aux, rule label/reason/onset, GBM-embedded (seed 0) proba,
                                 true label, split, SNR, key event parameters -- all from the float64 Python reference
  web/src/examples/*.csv         two example uploads (LN at 10 kHz, LL at 5 kHz, volts, U_n = 400 V)
Explorer set (~70 records): 10 per class from val + test_iid + test_shift, including the 6 records of
reports/figures/m4_disagreements.png and up to 20 rules-vs-GBM disagreements. The sealed test splits are read
for DISPLAY only, after the one-shot M4 evaluation (no method selection; D-077, CLAUDE.md rule 4).

Usage: python scripts/export_web.py [--fast]   (--fast: data/fast splits, fewer records; for development)
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gridmon import CLASSES, GENERATOR_VERSION
from gridmon.baseline_rules import REASONS, classify
from gridmon.export_gbm import Float32Walker, load_model
from gridmon.features import AUX_NAMES, FEATURE_NAMES, FEATURE_SPEC, compute_traces, features_from_traces
from gridmon.io import ROOT, load_baseline_config, load_manifest, load_waveforms, load_yaml
from gridmon.units import LSB_PU

WEB = ROOT / "web" / "src"
DATA = WEB / "data"
SPLITS = ("val", "test_iid", "test_shift")
SEALED = ("test_iid", "test_shift")
PER_CLASS = 10
MAX_DISAGREE = 20
PARAM_COLS = ("ev_t0", "ev_V", "ev_dur", "ev_sag_type", "ev_special", "ev_S", "ev_dphi_deg", "ev_dv1", "ev_rocof",
              "ev_df_target", "ev_depth", "ev_ang_amp_deg", "ev_f_m", "ev_sigma", "hn_step_on", "hn_ang_on", "hn_harm_on",
              "hn_ramp_on", "u_pre", "unb", "f_pre", "tr_on")


def _b64(a: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def _clean(o: Any) -> Any:
    """JSON-safe: numpy scalars -> python, NaN/inf -> None, floats rounded to 6 significant digits."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if not np.isfinite(f) else float(f"{f:.6g}")
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ------------------------------------------------------------------------------------------- site data
def validation_ladder() -> list[dict[str, str]]:
    rows = []
    for line in (ROOT / "docs" / "VALIDATION.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\| (L\d) \| (.*?) \| (.*?) \| (.*) \|$", line)
        if m:
            status = re.sub(r"\*\*", "", m.group(4))
            rows.append({"level": m.group(1), "what": m.group(2), "evidence": m.group(3), "status": status})
    return rows


def results_subset(r: dict) -> dict:
    out: dict[str, Any] = {"git_commit": r["git_commit"][:7], "freeze_commit": r["freeze_commit"][:7],
                           "methods": ["rules", "gbm_embedded", "gbm_unconstrained"], "splits": {}}
    for split in ("test_iid", "test_shift"):
        s = r["per_split"][split]
        d: dict[str, Any] = {"n": s["n"], "methods": {}, "comparisons": {}, "breakdowns": s["breakdowns"],
                             "naive_pll": s["naive_pll"]}
        for m, v in s["methods"].items():
            d["methods"][m] = {"macro_f1": v["macro_f1"], "accuracy": v["accuracy"], "false_alarm_rate": v["false_alarm_rate"],
                               "miss_rate": v["miss_rate"], "per_class_mean": v["per_class_mean"],
                               "critical_confusions_mean": v["critical_confusions_mean"], "n_seeds": v["n_seeds"],
                               "confusion": v["primary"]["confusion"]}
        for c, v in s["comparisons"].items():
            b = v["bootstrap"]
            d["comparisons"][c] = {"diff": b["diff_seed_mean"], "ci95": b["ci95_seed_mean"], "excludes_zero": b["excludes_zero"],
                                   "mcnemar_p": [x["p_value"] for x in v["mcnemar_per_seed"]],
                                   "mcnemar_counts": [[x["a_correct_b_wrong"], x["a_wrong_b_correct"]] for x in v["mcnemar_per_seed"]],
                                   "note": v.get("note", "")}
        d["rules_vs_gbm_disagreements"] = s["rules_vs_gbm_embedded_disagreements_primary"]
        out["splits"][split] = d
    # derived counts (computed here from the confusion matrices; seed 0 = the deployable C model)
    cls = list(CLASSES)
    der: dict[str, Any] = {}
    for split in ("test_iid", "test_shift"):
        for m, v in out["splits"][split]["methods"].items():
            M = np.array(v["confusion"]["matrix"])
            n_cls = M.sum(axis=1)
            der[f"{split}.{m}.errors"] = int(M.sum() - np.trace(M))
            der[f"{split}.{m}.jump_recall"] = float(M[cls.index("phase_jump"), cls.index("phase_jump")] / n_cls[cls.index("phase_jump")])
            der[f"{split}.{m}.jump_to_normal"] = int(M[cls.index("phase_jump"), cls.index("normal")])
            der[f"{split}.{m}.jump_to_osc"] = int(M[cls.index("phase_jump"), cls.index("oscillation")])
            der[f"{split}.{m}.normal_flagged"] = int(n_cls[0] - M[0, 0])
            der[f"{split}.{m}.normal_to_osc"] = int(M[0, cls.index("oscillation")])
            der[f"{split}.{m}.jump_to_dip_swell"] = int(M[cls.index("phase_jump"), [1, 2, 3]].sum())
    out["derived"] = {k.replace(".", "__"): v for k, v in der.items()}
    out["iid_to_shift"] = r["iid_to_shift"]
    out["snr_by_bin_pooled"] = r["snr_by_bin_pooled"]
    out["embedded_cost"] = r["embedded_cost"]
    out["thresholds"] = r["frozen"]["rules"]["thresholds"]
    out["gbm_config"] = r["frozen"]["gbm_embedded"]["config"]
    out["gbm_unconstrained_config"] = r["frozen"]["gbm_unconstrained"]["config"]
    out["gbm_unconstrained_bytes"] = r["frozen"]["gbm_unconstrained"]["bytes_equivalent"]
    out["val_macro_f1"] = {"rules": r["frozen"]["rules"]["val_macro_f1"], "gbm_embedded": r["frozen"]["gbm_embedded"]["val_macro_f1_mean"],
                           "gbm_unconstrained": r["frozen"]["gbm_unconstrained"]["val_macro_f1_mean"]}
    return out


def site_data() -> dict:
    r = json.loads((ROOT / "reports" / "results.json").read_text())
    m3 = json.loads((ROOT / "reports" / "m3_gbm_val.json").read_text())
    m2b = _load(ROOT / "reports" / "m2_embedded_budget.json") or {}
    m6 = _load(ROOT / "reports" / "m6_sil.json") or {}
    m7 = _load(ROOT / "reports" / "m7_port_stream.json") or {}
    c28 = _load(ROOT / "reports" / "m7_c28x_size.json")
    m1 = _load(ROOT / "reports" / "m1_dataset_summary.json") or {}
    grid = [{"stages": g["n_estimators"], "leaves": g["max_leaf_nodes"], "lr": g["learning_rate"], "subsample": g["subsample"],
             "bytes": g["bytes_max"], "val_f1": g["val_f1_mean"]} for g in m3["embedded"]["grid"]]
    sel = m3["embedded"]["selected"]
    sil = {"version": m6.get("version"), "golden": {"n_pass": m6.get("golden", {}).get("n_pass"), "n": m6.get("golden", {}).get("n_files")},
           "sanitizer": m6.get("sanitizer", {}).get("status"), "memory": m6.get("memory", {}),
           "portability": m6.get("portability", {}).get("cross_compile", {}),
           "static_checks_pass": all(m6.get("portability", {}).get("static_checks", {"x": False}).values()), "splits": {}}
    for s, v in m6.get("splits", {}).items():
        f = v["features"]
        sil["splits"][s] = {"n": v["n_records"], "subset": v["subset"], "features_ok": f["n_features_within_tolerance_on_all_records"],
                            "n_features": f["n_features"], "outside": f["n_values_outside_tolerance"],
                            "outside_ungated": sum(f.get("outside_ungated_documented", {}).values()),
                            "rules_agree": v["rules"]["label_agree"], "gbm_a_bit_exact": v["gbm_a_same_c_features"]["bit_exact_labels"],
                            "gbm_a_max_dp": v["gbm_a_same_c_features"]["max_abs_proba_diff"], "gbm_b_agree": v["gbm_b_end_to_end"]["label_agree"],
                            "c_us": v["time"]["c_us_per_record"]}
    port = {}
    if m7:
        mm = m7["main_LL_calibrated"]
        port = {"summary": mm["summary"], "stream": mm["stream"], "per_event": mm["per_event"], "isr_cost": m7["isr_cost"],
                "decimator_check": m7["decimator_check"], "ln": m7["LN_input"], "port_struct_bytes": mm["port_struct_bytes"],
                "port_struct_words16": mm["port_struct_words16"], "duration_s": mm["duration_s"], "f_in_hz": mm["f_in_hz"],
                "seed": mm["seed"], "proc_delay_s": mm["proc_delay_s"]}
    c28x = None
    if c28:
        c28x = {"compiler": c28["version"], "flags": c28["flags"], "dropped": c28["flags_dropped_not_in_help"],
                "all_compiled": c28["all_compiled"],
                "variants": {k: {"total_words": v["total_words"],
                                 "files": {Path(f).name: {kk: e.get(kk) for kk in ("ok", "code_words", "const_words", "data_words")}
                                           for f, e in v["files"].items()}} for k, v in c28["variants"].items()}}
    counts = {s: m1.get(s, {}).get("n_records") for s in ("train", "val", "test_iid", "test_shift")}
    return _clean({
        "generator_version": GENERATOR_VERSION, "classes": list(CLASSES), "reasons": [REASONS[i] for i in range(8)],
        "features": [{"name": n, "unit": u, "tol": t, "def": d} for n, u, t, d in FEATURE_SPEC], "aux_names": AUX_NAMES,
        "results": results_subset(r), "size_vs_f1": {"grid": grid, "budget_bytes": m3["embedded"]["budget_bytes"],
                                                    "selected": {"bytes": sel.get("bytes_max"), "config": sel.get("config"),
                                                                 "val_f1": sel.get("val_f1_mean")},
                                                    "unconstrained_bytes": r["frozen"]["gbm_unconstrained"]["bytes_equivalent"],
                                                    "unconstrained_val_f1": r["frozen"]["gbm_unconstrained"]["val_macro_f1_mean"]},
        "budget": m2b.get("totals", {}), "sil": sil, "port": port, "c28x": c28x, "ladder": validation_ladder(),
        "dataset_counts": counts,
    })


# ------------------------------------------------------------------------------------------- explorer
def _predictions(split: str, fast: bool, fcfg: dict, th: dict, walker: Float32Walker) -> dict[str, Any]:
    allow = split in SEALED  # display only, after M4 (D-077)
    X, meta = load_waveforms(split, fast=fast, allow_test=allow)
    man = load_manifest(split, fast=fast, allow_test_manifest=allow)
    return {"X": X, "lsb": float(meta["lsb_pu"]), "man": man}


def _features_for(X: np.ndarray, lsb: float, fcfg: dict, th: dict, walker: Float32Walker, idx: np.ndarray) -> pd.DataFrame:
    rows = []
    for c0 in range(0, len(idx), 250):
        x = np.asarray(X[idx[c0:c0 + 250]])
        tr = compute_traces(x, lsb)
        F, _ = features_from_traces(tr, fcfg)
        rows.append(pd.DataFrame({k: np.asarray(v, dtype=np.float64) for k, v in F.items()}))
    F = pd.concat(rows, ignore_index=True)
    rules = classify(F, th)
    proba = walker.predict_proba(F[FEATURE_NAMES].to_numpy())
    F["rule_label"] = rules["rule_label"].to_numpy()
    F["rule_reason"] = rules["rule_reason"].to_numpy()
    F["gbm_label"] = [CLASSES[i] for i in np.argmax(walker.raw(F[FEATURE_NAMES].to_numpy()), axis=1)]
    F["gbm_pmax"] = proba.max(axis=1)
    return F


def select_records(data: dict, F: dict, fast: bool) -> list[tuple[str, int, str]]:
    """[(split, record, reason)] -- deterministic."""
    r = json.loads((ROOT / "reports" / "results.json").read_text())
    chosen: list[tuple[str, int, str]] = []
    seen = set()

    def add(split, rec, why):
        if (split, rec) not in seen and rec < len(data[split]["man"]):
            seen.add((split, rec))
            chosen.append((split, int(rec), why))
    if not fast:
        for e in r["disagreement_examples_test_shift"]:  # the records of reports/figures/m4_disagreements.png
            add("test_shift", int(e["record"]), "m4_disagreements figure")
    per_class_dis: dict[str, int] = {}
    pool = []
    for split in SEALED:
        f = F[split]
        y = data[split]["man"]["label"].to_numpy()
        for i in np.where(f["rule_label"].to_numpy() != f["gbm_label"].to_numpy())[0]:
            pool.append((split, int(i), y[i]))
    rng = np.random.default_rng(np.random.SeedSequence(8008))
    order = rng.permutation(len(pool)) if pool else []
    for k in order:
        if sum(1 for c in chosen if c[2] != "class fill") >= MAX_DISAGREE:
            break
        split, i, lab = pool[k]
        if per_class_dis.get(lab, 0) < 4:
            per_class_dis[lab] = per_class_dis.get(lab, 0) + 1
            add(split, i, "rules-vs-GBM disagreement")
    for lab in CLASSES:
        have = sum(1 for s, i, _ in chosen if data[s]["man"]["label"][i] == lab)
        k = 0
        cands = {s: [int(i) for i in rng.permutation(np.where(data[s]["man"]["label"].to_numpy() == lab)[0])] for s in SPLITS}
        while have < PER_CLASS and any(cands.values()):
            s = SPLITS[k % len(SPLITS)]
            k += 1
            if cands[s]:
                i = cands[s].pop(0)
                if (s, i) not in seen and F[s]["rule_label"].iloc[i] == F[s]["gbm_label"].iloc[i]:  # fill = agreeing records
                    add(s, i, "class fill")
                    have += 1
    return chosen


def record_file(split: str, i: int, why: str, data: dict, Frow: pd.Series, fcfg: dict, walker: Float32Walker) -> dict:
    x = np.asarray(data[split]["X"][i], dtype=np.int16)
    lsb = data[split]["lsb"]
    tr = compute_traces(x[None], lsb)
    Fd, ex = features_from_traces(tr, fcfg)
    F = pd.DataFrame({k: np.asarray(v, dtype=np.float64) for k, v in Fd.items()})
    feats = [float(F[n][0]) for n in FEATURE_NAMES]
    aux = [float(F[n][0]) for n in AUX_NAMES]
    proba = walker.predict_proba(np.array([feats]))[0]
    man = data[split]["man"].iloc[i]
    params = {c: man[c] for c in PARAM_COLS if c in man.index and not (isinstance(man[c], float) and np.isnan(man[c]))}
    v1, v2 = tr["v1"][0], tr["v2"][0]
    traces = {"urms": _b64(tr["urms"][0], "<f4"), "v1mag": _b64(np.abs(v1), "<f4"), "v2mag": _b64(np.abs(v2), "<f4"),
              "J_deg": _b64(np.degrees(tr["J"][0]), "<f4"), "f_dft": _b64(tr["f_dft"][0], "<f4"),
              "f_pll": _b64(tr["f_pll"][0], "<f4"), "e_mag": _b64(ex["e_mag"][0], "<f4"), "r_mag": _b64(ex["r_mag"][0], "<f4"),
              "r_ang_deg": _b64(np.degrees(ex["r_ang"][0]), "<f4")}
    rid = f"{split}_{i:04d}"
    return _clean({
        "id": rid, "split": split, "record": i, "why": why, "true": man["label"], "snr_db": man["snr_db"],
        "lsb": lsb, "x": _b64(x.reshape(-1), "<i2"), "traces": traces, "features": feats, "aux": aux,
        "rule_label": str(Frow["rule_label"]), "rule_reason": int(Frow["rule_reason"]),
        "rule_onset": float(classify(F, load_yaml(ROOT / "configs" / "frozen_methods.yaml")["rules"]["thresholds"])["rule_onset"][0]),
        "gbm_proba": proba.astype(np.float64), "gbm_label": CLASSES[int(np.argmax(walker.raw(np.array([feats]))[0]))],
        "params": params,
    })


def example_csvs(cfg_fast: bool) -> list[str]:
    """Two example uploads synthesised by the record generator (val seeds): LN 10 kHz and LL 5 kHz, volts."""
    from gridmon.generator.dataset import make_record_spec, split_seeds
    from gridmon.generator.synth import synth_phase_voltages
    from gridmon.io import load_dataset_config
    from gridmon.units import SQRT3
    cfg = load_dataset_config()
    seeds, _ = split_seeds(990001, 2)
    out_dir = WEB / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    un = 400.0  # nominal LL RMS (V)
    files = []
    rng = np.random.default_rng(np.random.SeedSequence(990001))
    for (label, fs, ln, name), seed in zip((("sag_unbal", 10000, True, "example_sag_unbal_LN_10kHz.csv"),
                                            ("phase_jump", 5000, False, "example_phase_jump_LL_5kHz.csv")), seeds):
        spec = make_record_spec(0, label, seed, cfg, "iid")
        t = np.arange(int(1.0 * fs)) / fs
        v = synth_phase_voltages(spec.pre, spec.ev, t)  # pu of nominal LN peak
        if ln:
            volts = v * (np.sqrt(2.0) * un / SQRT3)
            cols = "time_s,v_a,v_b,v_c"
        else:
            ll = np.stack([(v[0] - v[1]) / SQRT3, (v[1] - v[2]) / SQRT3, (v[2] - v[0]) / SQRT3])
            volts = ll * (np.sqrt(2.0) * un)
            cols = "time_s,v_ab,v_bc,v_ca"
        volts = volts + 0.002 * np.sqrt(2.0) * un * rng.standard_normal(volts.shape)
        lines = [cols] + [f"{tt:.6f},{a:.2f},{b:.2f},{c:.2f}" for tt, a, b, c in zip(t, *volts)]
        (out_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        files.append(f"{name} ({label}, t0 = {spec.ev['t0']:.3f} s)")
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args(argv)
    DATA.mkdir(parents=True, exist_ok=True)
    sd = site_data()
    (DATA / "site_data.js").write_text("window.GM_SITE = " + json.dumps(sd, separators=(",", ":")) + ";\n", encoding="utf-8")
    fcfg = load_baseline_config()["features"]
    th = load_yaml(ROOT / "configs" / "frozen_methods.yaml")["rules"]["thresholds"]
    walker = Float32Walker(load_model(ROOT / "models" / "gbm_embedded.json"))
    data = {s: _predictions(s, args.fast, fcfg, th, walker) for s in SPLITS}
    F = {s: _features_for(data[s]["X"], data[s]["lsb"], fcfg, th, walker, np.arange(len(data[s]["man"]))) for s in SPLITS}
    chosen = select_records(data, F, args.fast)
    exp_dir = DATA / "explorer"
    exp_dir.mkdir(parents=True, exist_ok=True)
    for old in exp_dir.glob("*.json"):
        old.unlink()
    index = []
    for split, i, why in chosen:
        Frow = F[split].iloc[i]
        rec = record_file(split, i, why, data, Frow, fcfg, walker)
        (exp_dir / f"{rec['id']}.json").write_text(json.dumps(rec, separators=(",", ":")), encoding="utf-8")
        index.append({k: rec[k] for k in ("id", "split", "record", "why", "true", "rule_label", "rule_reason", "gbm_label", "snr_db")}
                     | {"gbm_pmax": max(rec["gbm_proba"])})
    (DATA / "explorer_index.js").write_text("window.GM_EXPLORER = " + json.dumps(index, separators=(",", ":")) + ";\n",
                                            encoding="utf-8")
    ex = example_csvs(args.fast)
    import shutil
    shutil.copy2(ROOT / "reports" / "figures" / "m7_port_stream.png", DATA / "m7_port_stream.png")
    size = sum(p.stat().st_size for p in exp_dir.glob("*.json"))
    by = {c: sum(1 for r in index if r["true"] == c) for c in CLASSES}
    print(f"site_data.js {(DATA / 'site_data.js').stat().st_size / 1e3:.0f} KB; explorer {len(index)} records "
          f"({size / 1e6:.2f} MB; per class {by}; disagreements {sum(r['rule_label'] != r['gbm_label'] for r in index)}); "
          f"examples: {', '.join(ex)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
