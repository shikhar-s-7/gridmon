"""M2: conventional baseline on TRAIN/VAL only (features, rules, tuning, diagnostics, budget, figures).

Usage: python scripts/run_baseline.py [--fast]
Writes data[/fast]/{train,val}/features.csv, reports[/fast]/m2_baseline_val.json,
reports[/fast]/m2_embedded_budget.json and the M2 figures. Methods see only waveforms; labels and
manifests are used for evaluation and figure annotation only.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from gridmon import CLASSES
from gridmon.baseline_rules import classify, tune_thresholds
from gridmon.budget import estimate
from gridmon.features import FEATURE_NAMES, compute_traces, extract_features, tolerance_check
from gridmon.io import (data_root, load_baseline_config, load_labels, load_manifest, load_waveforms,
                        reports_root, write_json)
from gridmon.metrics import confusion_text, report
from gridmon.plots import m2_confusion, m2_naive_pll, m2_signal_chain_examples


def float32_check(X, F64: pd.DataFrame, fcfg, ths: dict, lsb: float) -> dict:
    """Recompute features in float32 and compare with float64 (labels and per-feature tolerances)."""
    F32 = extract_features(X, fcfg, lsb=lsb, dtype=np.float32)
    out = {"label_agreement": {}, "features": {}}
    for name, th in ths.items():
        a = classify(F64, th)["rule_label"].to_numpy()
        b = classify(F32, th)["rule_label"].to_numpy()
        out["label_agreement"][name] = {"agree": int((a == b).sum()), "n": len(a)}
    out["features"] = tolerance_check(F64, F32)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args(argv)
    fast = args.fast
    bcfg = load_baseline_config()
    fcfg = bcfg["features"]
    root, rep_dir = data_root(fast), reports_root(fast)
    t_all = time.perf_counter()

    # ---------------------------------------------------------------- features (waveforms only)
    feats: dict[str, pd.DataFrame] = {}
    runtimes: dict[str, float] = {}
    lsb = None
    for split in ("train", "val"):
        X, meta = load_waveforms(split, fast=fast)
        lsb = meta["lsb_pu"]
        t0 = time.perf_counter()
        F = extract_features(X, fcfg, lsb=lsb, progress=True)
        runtimes[f"features_{split}_s"] = round(time.perf_counter() - t0, 2)
        runtimes[f"features_{split}_ms_per_record"] = round(1e3 * (time.perf_counter() - t0) / len(F), 3)
        F.insert(0, "record_id", np.arange(len(F)))
        F.to_csv(root / split / "features.csv", index=False, float_format="%.9g")
        feats[split] = F
        print(f"features {split}: {len(F)} records, {runtimes[f'features_{split}_s']} s")
    y = {s: load_labels(s, fast=fast) for s in ("train", "val")}

    # ---------------------------------------------------------------- rules: default and train-tuned
    th_def = dict(bcfg["rules"])
    t0 = time.perf_counter()
    pred_def = {s: classify(feats[s], th_def) for s in ("train", "val")}
    runtimes["rules_val_ms_per_record"] = round(1e3 * (time.perf_counter() - t0) / (len(feats["train"]) + len(feats["val"])), 4)
    t0 = time.perf_counter()
    th_tuned, f1_train_tuned, n_combos = tune_thresholds(feats["train"], y["train"], th_def, bcfg["tuning_grid"])
    runtimes["tuning_s"] = round(time.perf_counter() - t0, 2)
    pred_tun = {s: classify(feats[s], th_tuned) for s in ("train", "val")}
    rep_def = {s: report(y[s], pred_def[s]["rule_label"].to_numpy()) for s in ("train", "val")}
    rep_tun = {s: report(y[s], pred_tun[s]["rule_label"].to_numpy()) for s in ("train", "val")}
    print(f"VAL macro-F1: default {rep_def['val']['macro_f1']:.3f}, tuned {rep_tun['val']['macro_f1']:.3f} "
          f"(train tuned {f1_train_tuned:.3f}, {n_combos} combos)")
    print(confusion_text(rep_def["val"]))

    # ---------------------------------------------------------------- evaluation-only manifest use
    man_va = load_manifest("val", fast=fast)
    onset = {}
    for name, pred in (("default", pred_def["val"]), ("tuned", pred_tun["val"])):
        d = {}
        ok = (pred["rule_label"].to_numpy() == y["val"]) & (y["val"] != "normal")
        err = pred["rule_onset"].to_numpy() - man_va["ev_t0"].to_numpy()
        for c in CLASSES[1:]:
            e = err[ok & (y["val"] == c)]
            e = e[np.isfinite(e)]
            d[c] = {"n": int(len(e)), "median_ms": float(np.median(e) * 1e3) if len(e) else None,
                    "p90_abs_ms": float(np.quantile(np.abs(e), 0.9) * 1e3) if len(e) else None}
        onset[name] = d
    osc = man_va[man_va["label"] == "oscillation"]
    osc_overlap = {"n_oscillation": int(len(osc)),
                   "analytic_min_ll_below_0.90": int((osc["mm_min_ll"] < 0.90).sum()),
                   "analytic_max_ll_above_1.10": int((osc["mm_max_ll"] > 1.10).sum()),
                   "either": int(((osc["mm_min_ll"] < 0.90) | (osc["mm_max_ll"] > 1.10)).sum())}

    # ---------------------------------------------------------------- diagnostics
    dcfg = bcfg["diagnostics"]
    t0 = time.perf_counter()
    ctrl = {}
    for s in ("train", "val"):
        X, meta = load_waveforms(s, fast=fast)
        ctrl[s] = extract_features(X, fcfg, lsb=meta["lsb_pu"], n_samples=750, ref_end=749, obs_start=0)[FEATURE_NAMES]
    used = [c for c in FEATURE_NAMES if ctrl["train"][c].notna().any()]
    clf = HistGradientBoostingClassifier(random_state=dcfg["control_seed"])
    clf.fit(ctrl["train"][used].to_numpy(), y["train"])
    rep_ctrl = report(y["val"], clf.predict(ctrl["val"][used].to_numpy()))
    runtimes["control_s"] = round(time.perf_counter() - t0, 2)
    control = {"macro_f1": rep_ctrl["macro_f1"], "accuracy": rep_ctrl["accuracy"], "features_used": used,
               "confusion": rep_ctrl["confusion"], "threshold": 0.25,
               "pass": rep_ctrl["macro_f1"] <= 0.25}
    print(f"pre-event-only control: macro-F1 {rep_ctrl['macro_f1']:.3f} ({'PASS' if control['pass'] else 'FAIL'} <= 0.25)")

    thr = dcfg["naive_pll_hz"]
    naive = {}
    for s in ("train", "val"):
        flag = feats[s]["dfpll_max"].to_numpy() > thr
        man = man_va if s == "val" else load_manifest("train", fast=fast)
        by_class = {c: {"flagged": int(flag[y[s] == c].sum()), "n": int((y[s] == c).sum()),
                        "rate": float(flag[y[s] == c].mean())} for c in CLASSES}
        jb = dcfg["jump_bins_deg"]
        pj = (y[s] == "phase_jump")
        size = np.abs(man["ev_dphi_deg"].to_numpy())
        by_jump = {}
        for lo, hi in zip(jb[:-1], jb[1:]):
            sel = pj & (size >= lo) & (size < hi if hi != jb[-1] else size <= hi)
            by_jump[f"{lo}-{hi}"] = {"flagged": int(flag[sel].sum()), "n": int(sel.sum()),
                                     "rate": float(flag[sel].mean()) if sel.any() else float("nan")}
        naive[s] = {"by_class": by_class, "by_jump_deg": by_jump}
    print("naive PLL false flags (val): " + ", ".join(
        f"{c} {100 * naive['val']['by_class'][c]['rate']:.0f}%" for c in ("sag_bal", "sag_unbal", "phase_jump", "normal")))

    # ---------------------------------------------------------------- float32 robustness (val)
    Xv, meta_v = load_waveforms("val", fast=fast)
    t0 = time.perf_counter()
    f32 = float32_check(Xv, feats["val"], fcfg, {"default": th_def, "tuned": th_tuned}, meta_v["lsb_pu"])
    runtimes["float32_check_s"] = round(time.perf_counter() - t0, 2)
    print("float32 label agreement (val): " + ", ".join(f"{k} {v['agree']}/{v['n']}" for k, v in f32["label_agreement"].items()))

    # ---------------------------------------------------------------- budget
    budget = estimate()
    write_json(rep_dir / "m2_embedded_budget.json", budget)
    tot = budget["totals"]
    print(f"budget: {tot['total_ops'] / 1e6:.2f} M ops/record, RAM {tot['ram_kb']} KB, const {tot['const_kb']} KB, flags {budget['flags']}")

    # ---------------------------------------------------------------- figures
    examples = []
    for c in ("sag_unbal", "phase_jump", "freq_event", "oscillation"):
        i = int(np.where(y["val"] == c)[0][0])
        tr = compute_traces(np.asarray(Xv[i:i + 1]), meta_v["lsb_pu"])
        title = f"{c} (val {i})"
        if c == "sag_unbal":
            title += f", type {man_va.loc[i, 'ev_sag_type']}{man_va.loc[i, 'ev_special']}"
        elif c == "phase_jump":
            title += f", {man_va.loc[i, 'ev_dphi_deg']:+.1f} deg"
        elif c == "freq_event":
            title += f", {man_va.loc[i, 'ev_rocof']:+.2f} Hz/s"
        else:
            title += f", f_m {man_va.loc[i, 'ev_f_m']:.1f} Hz"
        examples.append({"title": title, "t0": man_va.loc[i, "ev_t0"], "v": tr["v"][0], "urms": tr["urms"][0],
                         "v1": tr["v1"][0], "v2": tr["v2"][0], "f_dft": tr["f_dft"][0], "f_pll": tr["f_pll"][0],
                         "J": tr["J"][0], "rule": f"{pred_def['val'].loc[i, 'rule_label']} ({pred_def['val'].loc[i, 'rule_reason_name']})"})
    m2_signal_chain_examples(examples, fast)
    m2_confusion(rep_def["val"], rep_tun["val"], fast)
    m2_naive_pll({c: naive["val"]["by_class"][c] for c in ("normal", "sag_bal", "sag_unbal", "phase_jump")},
                 naive["val"]["by_jump_deg"], thr, fast)

    runtimes["total_s"] = round(time.perf_counter() - t_all, 1)
    out = {
        "split_evaluated": "val", "fast": fast,
        "thresholds": {"default": th_def, "tuned": th_tuned, "tuning": {"on": "train", "objective": "macro-F1",
                                                                         "n_combos": n_combos, "train_macro_f1": f1_train_tuned}},
        "val": {"default": rep_def["val"], "tuned": rep_tun["val"]},
        "train": {"default": rep_def["train"], "tuned": rep_tun["train"]},
        "onset_error_val": onset,
        "oscillation_dip_swell_overlap_val": osc_overlap,
        "diagnostics": {"pre_event_only_control": control, "naive_pll": {"threshold_hz": thr, **naive}},
        "float32_val": f32,
        "runtimes": runtimes,
    }
    write_json(rep_dir / "m2_baseline_val.json", out)
    print(f"M2 baseline done in {runtimes['total_s']} s -> {rep_dir / 'm2_baseline_val.json'}")
    return 0 if control["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
