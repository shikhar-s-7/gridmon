"""S2E1: one-shot evaluation of the frozen arms on the new sealed sets (test2_shift, test2_novel).

Usage: python scripts/s2e1_evaluate.py [--fast] [--dry-run [--frozen DRAFT]] [--render-only]
The only S2E1 script that passes allow_test=True. It refuses to load a sealed split unless gridmon.s2e1.verify()
passes: the Stage 1 freeze (R, G0, shared code) and the S2E1 freeze (GW/GP models, S2E1 method files) both match
their freeze commits. --dry-run evaluates val_wide (for test2_shift) and preview_novel (for test2_novel) instead and
never loads a sealed split (pipeline check before the freeze; reports/dryrun/). --fast evaluates the fast subsets with
models/s2e1/fast (smoke test after the freeze; reports/fast/). --render-only rewrites S2E1_RESULTS.md from an existing
s2e1_results.json and reports/s2e1_findings.md (no data loaded).
Outputs: data/test2_*/features.csv, reports/s2e1_results.json, reports/S2E1_RESULTS.md, reports/figures/s2e1_*.png.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from gridmon import frozen as fz
from gridmon import s2e1
from gridmon.features import extract_features
from gridmon.io import (ROOT, data_root, git_commit, load_baseline_config, load_labels, load_manifest, load_waveforms,
                        reports_root, write_json)
from gridmon.metrics import confusion_text
from gridmon.ml_gbm import encode
from gridmon.s2e1_report import render
from gridmon.stats import stratified_indices

COMPARISONS = [("GW-R", "GW", "R", "primary"), ("GP-R", "GP", "R", "primary"), ("HW-R", "HW", "R", "secondary"),
               ("H0-R", "H0", "R", "secondary"), ("GW-G0", "GW", "G0", "secondary"),
               ("G0-R", "G0", "R", "additional (not pre-registered): replication of Stage 1")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="val_wide / preview_novel stand in; no sealed split loaded")
    ap.add_argument("--frozen", type=Path, default=None, help="(dry-run only) draft S2E1 frozen yaml")
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args(argv)
    fast, dry = args.fast, args.dry_run
    rep_dir = reports_root(fast) / ("dryrun" if dry else "")
    if args.render_only:
        res = json.loads((rep_dir / "s2e1_results.json").read_text(encoding="utf-8"))
        fp = rep_dir / "s2e1_findings.md"
        (rep_dir / "S2E1_RESULTS.md").write_text(render(res, fp.read_text(encoding="utf-8") if fp.exists() else None),
                                                 encoding="utf-8")
        print(f"rendered {rep_dir / 'S2E1_RESULTS.md'}")
        return 0
    t_all = time.perf_counter()

    # ------------------------------------------------------------ freeze check BEFORE any sealed data
    if dry:
        import yaml
        s2f = yaml.safe_load(args.frozen.read_text()) if args.frozen else s2e1.load_frozen()
    else:
        try:
            s2f = s2e1.require_frozen(check_models=not fast)
        except fz.FreezeError as e:
            print(e)
            return 3
    st1 = fz.load_frozen(ROOT)
    commit, fcommit = git_commit(), s2e1.freeze_commit()
    print("DRY RUN (val_wide, preview_novel)" if dry else f"freeze OK (S2E1-freeze {str(fcommit)[:7]}, HEAD {commit[:12]})")
    splits = {"test2_shift": "val_wide", "test2_novel": "preview_novel"} if dry else \
        {"test2_shift": "test2_shift", "test2_novel": "test2_novel"}
    allow = not dry
    plan = s2f["evaluation_plan"]
    arms = s2e1.Arms(st1, s2f, ROOT, model_dir_override=s2e1.models_root(True) if fast else None)  # G0: frozen models
    rng = np.random.default_rng(np.random.SeedSequence(int(plan["bootstrap"]["seed"])))
    n_boot = int(plan["bootstrap"]["resamples"])
    bcfg = load_baseline_config()
    res: dict = {"experiment": "S2E1", "git_commit": commit, "s2e1_freeze_commit": fcommit,
                 "stage1_freeze_commit": fz.freeze_commit(ROOT), "prereg_commit": s2f.get("prereg_commit"), "fast": fast,
                 "dry_run": dry, "splits": splits, "frozen": s2f, "per_set": {}}
    store: dict = {}
    for name, split in splits.items():
        Xw, meta = load_waveforms(split, fast=fast, allow_test=allow)
        t0 = time.perf_counter()
        F = extract_features(Xw, bcfg["features"], lsb=meta["lsb_pu"])  # waveforms only (rule A1.5)
        t_feat = 1e3 * (time.perf_counter() - t0) / len(F)
        if not dry:
            Fo = F.copy()
            Fo.insert(0, "record_id", np.arange(len(F)))
            Fo.to_csv(data_root(fast) / split / "features.csv", index=False, float_format="%.9g")
        y = load_labels(split, fast=fast, allow_test=allow)              # scoring only
        man = load_manifest(split, fast=fast, allow_test_manifest=True)    # breakdowns only
        P = arms.predict(F)
        yi = encode(y)
        enc = s2e1.encode_all(P)
        idx = stratified_indices(yi, n_boot, rng)                          # one resample set per sealed set, all arms
        boot = {a: np.stack([s2e1.macro_f1_boot(yi, p, idx) for p in enc[a]]) for a in s2e1.ARMS}
        comps = {c: {**s2e1.compare_arms(yi, enc[a], enc[b], boot, a, b), "role": role} for c, a, b, role in COMPARISONS}
        d = {"n": int(len(y)), "data_meta": {k: meta[k] for k in ("split", "git_commit", "config_hash", "n_records")},
             "features_ms_per_record": t_feat,
             "arms": {a: s2e1.arm_metrics(y, P[a]) for a in s2e1.ARMS},
             "hybrid_ml_share": {h: P[f"{h}_ml_share"] for h in ("H0", "HW")},
             "comparisons": comps, "breakdowns": s2e1.breakdowns(y, P, man, plan),
             "rules_reasons": {k: int(v) for k, v in zip(*np.unique(P["rules_reason"], return_counts=True))}}
        res["per_set"][name] = d
        store[name] = {"y": y, "P": P, "man": man}
        mm = d["arms"]
        print(f"{name} ({split}, n={len(y)}): " + " | ".join(
            f"{a} {mm[a]['macro_f1']['mean']:.4f}" + (f"+/-{mm[a]['macro_f1']['sd']:.4f}" if mm[a]["n_seeds"] > 1 else "")
            for a in s2e1.ARMS))
        for c in ("GW-R", "GP-R", "HW-R", "H0-R", "GW-G0"):
            x = comps[c]
            print(f"   {c}: {x['diff_seed_mean']:+.4f} CI95 [{x['ci95_seed_mean'][0]:+.4f}, {x['ci95_seed_mean'][1]:+.4f}] "
                  f"({x['reading']}); McNemar p " + ", ".join(f"{m['p_value']:.3g}" for m in x["mcnemar_per_seed"]))

    # ------------------------------------------------------------ unseen-nuisance survival (test2_novel)
    nv = store["test2_novel"]
    surv_cfg = plan["unseen_nuisance_survival"]
    margin = abs(float(str(surv_cfg["survives"]).split(">=")[1].strip()))
    if "ih_on" in nv["man"]:
        mask = nv["man"]["ih_on"].to_numpy().astype(bool)
        yi = encode(nv["y"])
        res["survival_test2_novel"] = {a: s2e1.survival(yi, [encode(p) for p in nv["P"][a]], mask, n_boot, rng, margin)
                                       for a in s2e1.ARMS}
        print("survival (with - without interharmonic): " + "; ".join(
            f"{a} {v['delta']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] {v['verdict']}"
            for a, v in res["survival_test2_novel"].items()))

    # ------------------------------------------------------------ decision rule and hypotheses
    dec = {}
    for a in ("GW", "GP", "HW", "H0"):
        cis = {s: res["per_set"][s]["comparisons"][f"{a}-R"]["ci95_seed_mean"] for s in splits}
        dec[a] = {"ci95_by_set": cis, "verdict": s2e1.decision(cis),
                  "role": "primary" if a in s2f["decision_rules"]["primary_arms"] else "secondary"}
    sv = res.get("survival_test2_novel", {})
    hyp = {
        "H1": {"text": s2f["hypotheses"]["H1"], "supported": dec["GW"]["verdict"] in ("ML matches", "ML better")},
        "H2": {"text": s2f["hypotheses"]["H2"], "supported": dec["GP"]["verdict"] in ("ML matches", "ML better")},
        "H3": {"text": s2f["hypotheses"]["H3"],
               "supported": res["per_set"]["test2_shift"]["comparisons"]["GW-G0"]["ci95_seed_mean"][0] > 0},
        "H4": {"text": s2f["hypotheses"]["H4"], "supported": dec["HW"]["verdict"] in ("ML matches", "ML better")},
        "H5": {"text": s2f["hypotheses"]["H5"], "supported": sv.get("R", {}).get("verdict") == "survives"},
        "H6": {"text": s2f["hypotheses"]["H6"], "supported": sv.get("GW", {}).get("verdict") == "survives"},
    }
    res["decision"], res["hypotheses"] = dec, hyp
    print("decision rule: " + "; ".join(f"{a}: {v['verdict']}" for a, v in dec.items()))
    print("hypotheses: " + ", ".join(f"{h} {'supported' if v['supported'] else 'not supported'}" for h, v in hyp.items()))
    for name in splits:
        for a in ("R", "GW", "GP"):
            print(f"   {name} {a} (seed 0):\n" + confusion_text(res["per_set"][name]["arms"][a]["primary"]))

    # ------------------------------------------------------------ figures, report
    from gridmon.s2e1_plots import confusions, f1_by_snr, novel_breakdown
    fig_dir = (rep_dir / "figures") if (dry or fast) else (ROOT / "reports" / "figures")
    res["figures"] = [confusions(res, fig_dir).name, f1_by_snr(res, fig_dir).name, novel_breakdown(res, fig_dir).name]
    res["runtime_s"] = round(time.perf_counter() - t_all, 1)
    write_json(rep_dir / "s2e1_results.json", res)
    fp = rep_dir / "s2e1_findings.md"
    (rep_dir / "S2E1_RESULTS.md").write_text(render(res, fp.read_text(encoding="utf-8") if fp.exists() else None),
                                             encoding="utf-8")
    print(f"S2E1 evaluation done in {res['runtime_s']} s -> {rep_dir / 's2e1_results.json'}, S2E1_RESULTS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
