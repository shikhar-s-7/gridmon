"""S2E1: train and select GW and GP on train_wide / val_wide only (no sealed split is touched).

Usage: python scripts/s2e1_train.py [--fast] [--force]
For each trained arm (configs/s2e1.yaml arms GW: all 37 features, GP: physics subset):
1. the Stage 1 GBM-embedded grid and selection rule, unchanged (gridmon.ml_gbm, configs/ml.yaml), on train_wide,
   selected by mean val_wide macro-F1 under the 64 KB budget; refit of the selected configuration for seeds 0-2;
2. export: models[/fast]/s2e1/gbm_<arm>_seed<k>.json (GP remapped to the 37-feature vector) and, if every seed fits
   64 KB, the seed-0 C tables in c/generated_s2e1[/fast]/<arm>/; equivalence float32 walker vs sklearn on val_wide;
   C SIL of the seed-0 tables on val_wide (C core built against those tables; labels vs the walker);
3. pre-event-only control (selected configuration on features of the first 750 samples, train_wide -> val_wide);
4. val_wide reference numbers for R, G0 and the hybrids (information only; selection uses the trained arm's F1 only).
Writes reports[/fast]/s2e1_train.json. Full mode refuses to retrain once configs/s2e1_frozen.yaml exists (--force).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from gridmon import CLASSES
from gridmon import s2e1
from gridmon.baseline_rules import classify
from gridmon.export_gbm import Float32Walker, export_gbc, load_model, threshold_check, write_c_tables, write_model
from gridmon.features import FEATURE_NAMES, extract_features
from gridmon.frozen import load_frozen as load_stage1_frozen
from gridmon.io import (ROOT, data_root, git_commit, load_baseline_config, load_labels, load_ml_config, load_waveforms,
                        reports_root, write_json)
from gridmon.metrics import report
from gridmon.ml_gbm import decode, encode, fit_seeds, grid_embedded, model_input, nonfinite_counts, select_embedded, summarise

EMB_KEYS = ["n_estimators", "max_leaf_nodes", "learning_rate", "subsample"]


def build_sil_for(arm_dir: Path, tag: str) -> Path:
    """C SIL shared library built against an S2E1 model-table folder instead of c/generated/ (same flags)."""
    from gridmon import c_build as cb
    cc = cb.find_compiler()
    out = cb.BUILD_DIR / cb.shared_lib_name(f"gridmon_sil_s2e1_{tag}")
    srcs = [s for s in cb.CORE_SOURCES if s.parent.name != "generated"] + [arm_dir / "gm_model_gbm.c",
                                                                            cb.C_DIR / "host" / "gm_sil.c"]
    inc = [i for i in cb.INCLUDES if not i.endswith("generated")] + ["-I" + str(arm_dir)]
    pic = [] if sys.platform.startswith("win") else ["-fPIC"]
    cmd = cc + cb.WARN_FLAGS + cb.OPT_FLAGS + pic + ["-shared", "-DGM_TRACES", "-DGM_REPORT_NAN=1"] + inc + \
        [str(s) for s in srcs] + ["-o", str(out), "-lm"]
    return cb._run(cmd, out, srcs, True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain although S2E1 is frozen (breaks the S2E1 freeze)")
    ap.add_argument("--no-c", action="store_true", help="skip the C SIL check of the exported tables")
    args = ap.parse_args(argv)
    fast = args.fast
    if not fast and not args.force and (ROOT / s2e1.S2E1_FROZEN).exists():
        print(f"S2E1 arms are frozen ({s2e1.S2E1_FROZEN}): not retraining (use --force; s2e1_evaluate.py will then refuse)")
        return 0
    t_all = time.perf_counter()
    s2, mcfg = s2e1.load_config(), load_ml_config()
    seeds, budget = list(mcfg["seeds"]), int(mcfg["budget_bytes"])
    root = data_root(fast)
    F = {s: pd.read_csv(root / s / "features.csv") for s in ("train_wide", "val_wide")}
    X = {s: model_input(F[s]) for s in F}
    ys = {s: load_labels(s, fast=fast) for s in F}
    y = {s: encode(ys[s]) for s in F}
    nonfinite = {s: nonfinite_counts(F[s]) for s in F}
    print(f"train_wide {X['train_wide'].shape}, val_wide {X['val_wide'].shape}; non-finite -> 0.0 (D-040): {nonfinite}")

    # pre-event-only control features (first 750 samples, OBS = REF), shared by both arms
    t0 = time.perf_counter()
    bcfg = load_baseline_config()
    ctrl = {}
    for s in F:
        Xw, meta = load_waveforms(s, fast=fast)
        ctrl[s] = model_input(extract_features(Xw, bcfg["features"], lsb=meta["lsb_pu"], n_samples=750, ref_end=749,
                                               obs_start=0))
    ctrl_s = round(time.perf_counter() - t0, 1)

    # references on val_wide (information only)
    st1 = load_stage1_frozen()
    rules_val = classify(F["val_wide"], st1["rules"]["thresholds"])["rule_label"].to_numpy()
    g0 = [Float32Walker(load_model(ROOT / m["path"])) for m in st1["gbm_embedded"]["models"]]
    ref = {"R": report(ys["val_wide"], rules_val)["macro_f1"],
           "G0": [report(ys["val_wide"], decode(w.predict(X["val_wide"])))["macro_f1"] for w in g0]}
    print(f"val_wide reference: R {ref['R']:.4f}; G0 {np.mean(ref['G0']):.4f} (seeds {np.round(ref['G0'], 4).tolist()})")

    out: dict = {"fast": fast, "git_commit": git_commit(), "sklearn_version": sklearn.__version__, "trained_on": "train_wide",
                 "selected_on": "val_wide", "classes": list(CLASSES), "ml_config": "configs/ml.yaml (unchanged)",
                 "budget_bytes": budget, "tie_tol": mcfg["tie_tol"], "seeds": seeds,
                 "input_mapping": {"nan_fill": 0.0, "nonfinite_counts": nonfinite, "decision": "D-040"},
                 "val_reference": ref, "arms": {}, "runtimes": {"control_features_s": ctrl_s}}
    ok = True
    for arm in s2e1.TRAINED_ARMS:
        t0 = time.perf_counter()
        subset = s2e1.arm_subset(arm, s2)
        cols = s2e1.feature_columns(subset, s2)
        names = [FEATURE_NAMES[c] for c in cols]
        Xtr, Xva = X["train_wide"][:, cols], X["val_wide"][:, cols]
        fits = grid_embedded(mcfg, Xtr, y["train_wide"], Xva, y["val_wide"])
        summ = summarise(fits, EMB_KEYS).sort_values(EMB_KEYS).reset_index(drop=True)
        sel = select_embedded(summ, budget, mcfg["tie_tol"])
        grid_s = round(time.perf_counter() - t0, 1)
        clfs = fit_seeds("embedded", sel["config"], mcfg, Xtr, y["train_wide"])
        models, equiv, walkers = [], [], []
        for seed, clf in zip(seeds, clfs):
            m = export_gbc(clf, names, list(CLASSES), Xtr[:256], extra={
                "method": f"S2E1 {arm} (GBM-embedded recipe, {subset} features)", "seed": seed,
                "hyperparameters": {**sel["config"], **mcfg["embedded"]["fixed"]},
                "trained_on": "train_wide" + (" (fast)" if fast else ""), "git_commit": git_commit()})
            if subset != "all37":
                m = s2e1.remap_model_features(m, cols)
            path = s2e1.model_path(arm, seed, fast)
            sha = write_model(m, path)
            w = Float32Walker(m)
            walkers.append(w)
            p_sk, p_w = clf.predict_proba(Xva), w.predict_proba(X["val_wide"])
            lab_sk, lab_w = clf.predict(Xva), w.predict(X["val_wide"])
            tc = threshold_check(clf, Xva)
            e = {"seed": seed, "max_abs_proba_diff": float(np.abs(p_w - p_sk).max()),
                 "n_label_disagree": int((lab_w != lab_sk).sum()), "threshold_check": tc,
                 "pass": bool(np.abs(p_w - p_sk).max() <= 1e-3 and (lab_w == lab_sk).all() and tc["mismatches"] == 0)}
            equiv.append(e)
            models.append({"seed": seed, "path": path.relative_to(ROOT).as_posix(), "sha256": sha,
                           "size_bytes": m["size_bytes"], "n_nodes": m["n_nodes"], "n_trees": m["n_trees"],
                           "max_depth": m["max_depth"], "sum_max_depth": m["sum_max_depth"]})
            ok &= e["pass"]
        fits_budget = all(mm["size_bytes"] <= budget for mm in models)
        c_info: dict = {"exported": False}
        if fits_budget:
            cdir = s2e1.c_generated_root(fast) / arm.lower()
            write_c_tables(load_model(ROOT / models[0]["path"]), models[0]["sha256"], cdir,
                           source=models[0]["path"])
            c_info = {"exported": True, "dir": cdir.relative_to(ROOT).as_posix(), "seed": seeds[0]}
            if not args.no_c:
                from gridmon.c_sil import CCore
                Xw, meta = load_waveforms("val_wide", fast=fast)
                core = CCore(build_sil_for(cdir, arm.lower() + ("_fast" if fast else "")), build=False)
                c = core.process(np.asarray(Xw), float(meta["lsb_pu"]))
                Fc = c["features"][FEATURE_NAMES].to_numpy(dtype=np.float64)
                raw_py = walkers[0].raw(Fc)
                c_info["sil_val_wide"] = {
                    "n": int(len(Fc)), "raw_bit_exact_on_c_features": bool(np.array_equal(raw_py, c["gbm_raw"])),
                    "labels_equal_on_c_features": int((np.argmax(raw_py, axis=1) == c["gbm_label"]).sum()),
                    "end_to_end_label_agreement": float((walkers[0].predict(X["val_wide"]) == c["gbm_label"]).mean())}
                ok &= c_info["sil_val_wide"]["raw_bit_exact_on_c_features"]
        # val metrics
        labs = [decode(w.predict(X["val_wide"])) for w in walkers]
        probas = [w.predict_proba(X["val_wide"]) for w in walkers]
        reps = [report(ys["val_wide"], lab) for lab in labs]
        hyb = [report(ys["val_wide"], s2e1.hybrid(lab, p, rules_val, float(s2["hybrid"]["threshold"])))["macro_f1"]
               for lab, p in zip(labs, probas)]
        # pre-event-only control
        cm = fit_seeds("embedded", sel["config"], mcfg, ctrl["train_wide"][:, cols], y["train_wide"])
        cf1 = [report(ys["val_wide"], decode(c.predict(ctrl["val_wide"][:, cols])))["macro_f1"] for c in cm]
        thr = float(s2["evaluation"]["pre_event_only_control"]["pass_if_mean_macro_f1_le"])
        control = {"macro_f1": cf1, "macro_f1_mean": float(np.mean(cf1)), "macro_f1_sd": float(np.std(cf1, ddof=1)),
                   "chance": 1 / len(CLASSES), "threshold": thr, "pass": bool(np.mean(cf1) <= thr)}
        ok &= control["pass"]
        imp = clfs[0].feature_importances_
        order = np.argsort(imp)[::-1][:10]
        f1v = np.array([r["macro_f1"] for r in reps])
        out["arms"][arm] = {
            "feature_subset": subset, "n_features_used": len(cols), "features": names,
            "grid": summ.to_dict(orient="records"), "fits": fits.to_dict(orient="records"), "selected": sel,
            "models": models, "fits_budget": fits_budget, "c_tables": c_info, "equivalence_val_wide": equiv,
            "val_wide": {"macro_f1": f1v.tolist(), "macro_f1_mean": float(f1v.mean()), "macro_f1_sd": float(f1v.std(ddof=1)),
                         "false_alarm_rate": [r["false_alarm_rate"] for r in reps], "miss_rate": [r["miss_rate"] for r in reps],
                         "report_primary": reps[0], "hybrid_macro_f1": hyb,
                         "share_max_proba_ge_threshold": [float((p.max(axis=1) >= 0.5).mean()) for p in probas]},
            "pre_event_only_control": control,
            "impurity_importance_seed0_top10": [{"feature": names[i], "importance": float(imp[i])} for i in order],
            "cost": {"model_bytes": models[0]["size_bytes"], "comparisons_per_record_mean_val":
                     float(walkers[0].path_lengths(X["val_wide"]).mean()),
                     "comparisons_worst_case": models[0]["sum_max_depth"]},
        }
        out["runtimes"][f"{arm}_grid_s"] = grid_s
        out["runtimes"][f"{arm}_total_s"] = round(time.perf_counter() - t0, 1)
        print(f"{arm} ({subset}, {len(cols)} features): {len(fits)} fits, {sel['n_feasible']}/{len(summ)} configs fit; "
              f"selected {sel['config']} val F1 {sel['val_f1_mean']:.4f} +/- {sel['val_f1_sd']:.4f}, {sel['bytes_max']} B "
              f"({sel['n_tied']} within tie_tol); equivalence {'ok' if all(e['pass'] for e in equiv) else 'FAIL'}; "
              f"C {c_info.get('sil_val_wide', c_info)}; hybrid val F1 {np.round(hyb, 4).tolist()}; control "
              f"{control['macro_f1_mean']:.3f} ({'PASS' if control['pass'] else 'FAIL'}) [{out['runtimes'][f'{arm}_total_s']} s]")
    out["runtimes"]["total_s"] = round(time.perf_counter() - t_all, 1)
    write_json(reports_root(fast) / "s2e1_train.json", out)
    print(f"S2E1 training done in {out['runtimes']['total_s']} s -> {reports_root(fast) / 's2e1_train.json'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
