"""M4: one-shot evaluation of the FROZEN methods on the sealed test sets (test_iid, test_shift).

Usage: python scripts/evaluate_test.py [--fast] [--dry-run] [--render-only]
This is the only script that passes allow_test=True. It refuses to run unless the working tree
matches the M4-freeze commit in every method-defining file and model (gridmon.frozen.verify).
--dry-run evaluates VAL in place of both test sets (pipeline check before the freeze; never loads a
test split; writes to reports/dryrun/). --fast evaluates the fast test subsets with models/fast (smoke
test; writes to reports/fast/).
Outputs: data/<split>/features.csv, reports/results.json, reports/RESULTS.md, reports/figures/m4_*.png.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from gridmon import CLASSES
from gridmon.budget import estimate
from gridmon.evaluation import (METHODS, FrozenMethods, breakdowns, compare, iid_to_shift, method_metrics,
                                naive_pll)
from gridmon.features import compute_traces, extract_features
from gridmon.frozen import FreezeError, freeze_commit, load_frozen, require_frozen
from gridmon.io import (ROOT, data_root, git_commit, load_baseline_config, load_labels, load_manifest, load_waveforms,
                        models_root, reports_root, write_json)
from gridmon.metrics import confusion_text
from gridmon.plots_ml import m4_confusions, m4_disagreements, m4_f1_by_snr, m4_shift_degradation
from gridmon.results_report import render_results_md


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="evaluate VAL as a stand-in (no test split is loaded)")
    ap.add_argument("--frozen", type=Path, default=None, help="(dry-run only) draft frozen yaml")
    ap.add_argument("--render-only", action="store_true",
                    help="rewrite RESULTS.md from an existing results.json (+ results_findings.md); loads no data")
    args = ap.parse_args(argv)
    fast, dry = args.fast, args.dry_run
    t_all = time.perf_counter()
    if args.render_only:
        import json
        rep_dir = reports_root(fast) / ("dryrun" if dry else "")
        res = json.loads((rep_dir / "results.json").read_text(encoding="utf-8"))
        findings = rep_dir / "results_findings.md"
        (rep_dir / "RESULTS.md").write_text(render_results_md(res, findings.read_text(encoding="utf-8") if findings.exists()
                                                              else None), encoding="utf-8")
        print(f"rendered {rep_dir / 'RESULTS.md'}")
        return 0

    # ------------------------------------------------------------ freeze check BEFORE any test data
    if dry:
        import yaml
        frozen = yaml.safe_load(args.frozen.read_text()) if args.frozen else load_frozen()
    else:
        try:
            frozen = require_frozen(check_models=not fast)
        except FreezeError as e:
            print(e)
            return 3
    commit, fcommit = git_commit(), freeze_commit()
    print(f"freeze OK (freeze commit {str(fcommit)[:7]}, HEAD {commit[:12]})" if not dry else "DRY RUN on val")
    splits = ("val", "val") if dry else ("test_iid", "test_shift")
    rep_dir = reports_root(fast) / ("dryrun" if dry else "")
    allow = not dry
    plan = frozen["evaluation_plan"]
    bcfg = load_baseline_config()
    methods = FrozenMethods(frozen, ROOT, model_dir_override=models_root(True) if fast else None)
    rng = np.random.default_rng(np.random.SeedSequence(int(plan["bootstrap_seed"])))

    res: dict = {"git_commit": commit, "freeze_commit": fcommit, "fast": fast, "dry_run": dry,
                 "splits": dict(zip(("test_iid", "test_shift"), splits)), "frozen": frozen, "per_split": {}}
    store: dict = {}
    timing: dict = {}
    for name, split in zip(("test_iid", "test_shift"), splits):
        Xw, meta = load_waveforms(split, fast=fast, allow_test=allow)
        t0 = time.perf_counter()
        F = extract_features(Xw, bcfg["features"], lsb=meta["lsb_pu"])  # waveforms only (rule A1.5)
        timing[f"features_{name}_ms_per_record"] = 1e3 * (time.perf_counter() - t0) / len(F)
        if not dry:
            Fo = F.copy()
            Fo.insert(0, "record_id", np.arange(len(F)))
            Fo.to_csv(data_root(fast) / split / "features.csv", index=False, float_format="%.9g")
        y = load_labels(split, fast=fast, allow_test=allow)            # evaluation only
        man = load_manifest(split, fast=fast, allow_test_manifest=True)  # breakdowns only
        P = methods.predict(F)
        for k, v in methods.timing.items():
            timing[f"{k}_{name}"] = v
        mm = {m: method_metrics(y, P[m]) for m in METHODS}
        d = {"n": int(len(y)), "data_meta": {k: meta[k] for k in ("split", "git_commit", "config_hash", "n_records")},
             "methods": mm, "comparisons": compare(y, P, plan, rng), "breakdowns": breakdowns(y, P, man, plan),
             "naive_pll": naive_pll(F, y, man, plan["naive_pll_hz"], plan["jump_bins_deg"]),
             "gbm_embedded_comparisons_per_record": {"mean": float(P["gbm_embedded_comparisons"].mean()),
                                                     "max": int(P["gbm_embedded_comparisons"].max())},
             "rules_vs_gbm_embedded_disagreements_primary": int((P["rules"][0] != P["gbm_embedded"][0]).sum())}
        # cross-check: float32 walker vs the sklearn pickles of the same seeds
        try:
            import joblib
            from gridmon.ml_gbm import model_input
            agree = []
            for mdl, w_lab in zip(frozen["gbm_embedded"]["models"], P["gbm_embedded"]):
                jp = (models_root(True) if fast else ROOT / "models") / f"gbm_embedded_seed{mdl['seed']}.joblib"
                clf = joblib.load(jp)
                agree.append(float((np.array([CLASSES[i] for i in clf.predict(model_input(F))]) == w_lab).mean()))
            d["gbm_embedded_walker_vs_sklearn_label_agreement"] = agree
        except FileNotFoundError:
            d["gbm_embedded_walker_vs_sklearn_label_agreement"] = None
        res["per_split"][name] = d
        store[name] = {"y": y, "P": P, "man": man, "X": Xw, "F": F, "lsb": meta["lsb_pu"]}
        print(f"{name} ({split}): macro-F1 rules {mm['rules']['macro_f1']['mean']:.4f} | GBM-emb "
              f"{mm['gbm_embedded']['macro_f1']['mean']:.4f} +/- {mm['gbm_embedded']['macro_f1']['sd']:.4f} | GBM-unc "
              f"{mm['gbm_unconstrained']['macro_f1']['mean']:.4f} +/- {mm['gbm_unconstrained']['macro_f1']['sd']:.4f}")
        c = d["comparisons"]["gbm_embedded_minus_rules"]
        print(f"   GBM-emb - rules: {c['bootstrap']['diff_seed_mean']:+.4f} CI95 [{c['bootstrap']['ci95_seed_mean'][0]:+.4f}, "
              f"{c['bootstrap']['ci95_seed_mean'][1]:+.4f}]; McNemar p (seeds) "
              + ", ".join(f"{x['p_value']:.3g}" for x in c["mcnemar_per_seed"]))
        for m in METHODS:
            print(f"   {m} (primary):\n" + confusion_text(mm[m]["primary"]))

    a, b = store["test_iid"], store["test_shift"]
    res["iid_to_shift"] = iid_to_shift(a["y"], a["P"], b["y"], b["P"], plan, rng)
    budget = estimate()
    emb_model = methods.walkers[0]
    res["embedded_cost"] = {
        "dsp_features_ops_per_record": budget["totals"]["total_ops"], "dsp_ram_kb": budget["totals"]["ram_kb"],
        "dsp_const_kb": budget["totals"]["const_kb"],
        "rules_comparisons_per_record": "<= 12 threshold comparisons",
        "gbm_model_bytes": int(frozen["gbm_embedded"]["models"][0]["size_bytes"]),
        "gbm_comparisons_per_record_mean": {s: res["per_split"][s]["gbm_embedded_comparisons_per_record"]["mean"] for s in res["per_split"]},
        "gbm_comparisons_worst_case": int(frozen["gbm_embedded"]["models"][0]["sum_max_depth"]),
        "gbm_ram_bytes": 4 * (emb_model.n_features + emb_model.K),
        "gbm_unconstrained_model_bytes_equivalent": int(frozen["gbm_unconstrained"]["bytes_equivalent"]),
        "python_timing_ms_per_record": {k: round(v, 4) for k, v in timing.items()},
    }

    # ------------------------------------------------------------ figures
    fig_dir = rep_dir / "figures" if dry else None  # dry-run figures stay out of the committed folder
    reps = {s: {m: res["per_split"][s]["methods"][m]["primary"] for m in METHODS} for s in res["per_split"]}
    m4_confusions(reps, fast, fig_dir)
    snr_iid, snr_shift = res["per_split"]["test_iid"]["breakdowns"]["snr"], res["per_split"]["test_shift"]["breakdowns"]["snr"]
    bins = snr_iid["bins"]
    snr_data = {}
    for m in METHODS:
        f1, n = [], []
        for g in bins:
            src = snr_shift if snr_shift[m][g]["n"] else snr_iid
            f1.append(src[m][g]["value"])
            n.append(src[m][g]["n"])
        snr_data[m] = {"f1": f1, "n": n}
    res["snr_by_bin_pooled"] = {"bins": bins, **snr_data}
    m4_f1_by_snr(bins, snr_data, fast, fig_dir)
    deg = {m: {"iid": res["per_split"]["test_iid"]["methods"][m]["macro_f1"]["mean"],
               "shift": res["per_split"]["test_shift"]["methods"][m]["macro_f1"]["mean"],
               "iid_sd": res["per_split"]["test_iid"]["methods"][m]["macro_f1"]["sd"],
               "shift_sd": res["per_split"]["test_shift"]["methods"][m]["macro_f1"]["sd"]} for m in METHODS}
    m4_shift_degradation(deg, fast, fig_dir)
    # disagreements on test_shift: first disagreement per true class, then fill (record order)
    yb, Pb = b["y"], b["P"]
    dis = np.where(Pb["rules"][0] != Pb["gbm_embedded"][0])[0]
    pick: list[int] = []
    for c in CLASSES:
        cand = [i for i in dis if yb[i] == c and i not in pick]
        if cand and len(pick) < 6:
            pick.append(int(cand[0]))
    for i in dis:
        if len(pick) >= 6:
            break
        if int(i) not in pick:
            pick.append(int(i))
    examples = []
    for i in pick:
        tr = compute_traces(np.asarray(b["X"][i:i + 1]), b["lsb"])
        r = b["man"].iloc[i]
        examples.append({"v": tr["v"][0], "urms": tr["urms"][0], "v1": tr["v1"][0], "J": tr["J"][0],
                         "t0": float(r.get("ev_t0", np.nan)) if r["label"] != "normal" else np.nan,
                         "title": f"test_shift #{i}: SNR {r['snr_db']:.1f} dB, transient {'yes' if r['tr_on'] else 'no'}",
                         "true": yb[i], "rules": f"{Pb['rules'][0][i]} ({Pb['rules_reason'][i]})",
                         "gbm": Pb["gbm_embedded"][0][i],
                         "p_gbm": float(Pb["gbm_embedded_proba"][0][i].max())})
    res["disagreement_examples_test_shift"] = [{"record": int(i), "true": e["true"], "rules": e["rules"], "gbm": e["gbm"],
                                                "p_gbm": e["p_gbm"]} for i, e in zip(pick, examples)]
    if examples:
        m4_disagreements(examples, fast, fig_dir)

    res["runtime_s"] = round(time.perf_counter() - t_all, 1)
    write_json(rep_dir / "results.json", res)
    findings = rep_dir / "results_findings.md"
    (rep_dir / "RESULTS.md").write_text(render_results_md(res, findings.read_text(encoding="utf-8") if findings.exists() else None),
                                        encoding="utf-8")
    print(f"M4 done in {res['runtime_s']} s -> {rep_dir / 'results.json'}, RESULTS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
