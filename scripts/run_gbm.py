"""M3: gradient boosting on TRAIN/VAL (no test split is touched).

Usage: python scripts/run_gbm.py [--fast]
1. GBM-embedded grid (sklearn GradientBoostingClassifier; 3 seeds), size of every grid point, selection
   under the 64 KB budget; refit of the selected configuration for every seed.
2. Export: models[/fast]/gbm_embedded.json (seed 0; C tables in c/generated[/fast]/) and
   gbm_embedded_seed<k>.json for the other seeds; sklearn pickles (*.joblib) for cross-checks.
   Equivalence on VAL: float32 walker vs sklearn predict_proba; threshold branching check.
3. GBM-unconstrained reference grid (HistGradientBoostingClassifier, early stopping; 3 seeds).
4. Pre-event-only control (selected config), permutation importance (val), figures, reports[/fast]/m3_gbm_val.json.
Inputs: data[/fast]/{train,val}/features.csv (scripts/run_baseline.py) and the train-tuned rule thresholds
from reports[/fast]/m2_baseline_val.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from gridmon import CLASSES
from gridmon.baseline_rules import classify
from gridmon.export_gbm import (Float32Walker, export_gbc, sha256_file, threshold_check, write_c_tables,
                                write_model)
from gridmon.features import FEATURE_NAMES, extract_features
from gridmon.io import (ROOT, c_generated_root, data_root, git_commit, load_baseline_config, load_labels, load_ml_config,
                        load_waveforms, models_root, reports_root, write_json)
from gridmon.metrics import confusion_text, report
from gridmon.ml_gbm import (decode, encode, fit_seeds, grid_embedded, grid_unconstrained, model_input,
                            nonfinite_counts, select_embedded, select_unconstrained, summarise)
from gridmon.plots_ml import m3_feature_importance, m3_size_vs_f1, m3_val_confusion

EMB_KEYS = ["n_estimators", "max_leaf_nodes", "learning_rate", "subsample"]


def model_paths(fast: bool, seeds: list[int]) -> list:
    root = models_root(fast)
    return [root / ("gbm_embedded.json" if i == 0 else f"gbm_embedded_seed{s}.json") for i, s in enumerate(seeds)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain even if the methods are frozen (breaks the M4 freeze)")
    args = ap.parse_args(argv)
    fast = args.fast
    if not fast and not args.force and (ROOT / "configs" / "frozen_methods.yaml").exists():
        print("methods are frozen (configs/frozen_methods.yaml, M4): not retraining, frozen models kept "
              "(use --force to retrain; evaluate_test.py will then refuse until a new freeze)")
        return 0
    mcfg, bcfg = load_ml_config(), load_baseline_config()
    seeds = list(mcfg["seeds"])
    root, rep_dir, mod_dir = data_root(fast), reports_root(fast), models_root(fast)
    mod_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    rt: dict[str, float] = {}

    F = {s: pd.read_csv(root / s / "features.csv") for s in ("train", "val")}
    X = {s: model_input(F[s]) for s in F}
    ys = {s: load_labels(s, fast=fast) for s in F}
    y = {s: encode(ys[s]) for s in F}
    nonfinite = {s: nonfinite_counts(F[s]) for s in F}
    print(f"train {X['train'].shape}, val {X['val'].shape}; non-finite -> 0.0 (D-040): {nonfinite}")

    # ---------------------------------------------------------------- 1. GBM-embedded grid
    t0 = time.perf_counter()
    fits = grid_embedded(mcfg, X["train"], y["train"], X["val"], y["val"])
    rt["embedded_grid_s"] = round(time.perf_counter() - t0, 1)
    summ = summarise(fits, EMB_KEYS).sort_values(EMB_KEYS).reset_index(drop=True)
    sel = select_embedded(summ, mcfg["budget_bytes"], mcfg["tie_tol"])
    cfg_sel = sel["config"]
    print(f"embedded grid: {len(fits)} fits in {rt['embedded_grid_s']} s; {sel['n_feasible']}/{len(summ)} configs <= "
          f"{mcfg['budget_bytes']} B; selected {cfg_sel} val F1 {sel['val_f1_mean']:.4f} +/- {sel['val_f1_sd']:.4f}, "
          f"{sel['bytes_max']} B (best feasible {sel['best_feasible_f1']:.4f}, {sel['n_tied']} within tie_tol)")

    t0 = time.perf_counter()
    emb = fit_seeds("embedded", cfg_sel, mcfg, X["train"], y["train"])
    rt["embedded_refit_s"] = round(time.perf_counter() - t0, 1)

    # ---------------------------------------------------------------- 2. export + equivalence
    exported, equiv = [], []
    walkers = []
    n_self = int(mcfg["selftest_rows"])
    for i, (seed, clf) in enumerate(zip(seeds, emb)):
        p_sk = clf.predict_proba(X["val"])
        selftest = {"source": "first val rows (features after D-040 mapping, float32 values)",
                    "x": X["val"][:n_self].astype(np.float32).astype(np.float64).tolist(),
                    "sklearn_proba": p_sk[:n_self].tolist(), "sklearn_label": clf.predict(X["val"][:n_self]).tolist()}
        model = export_gbc(clf, FEATURE_NAMES, list(CLASSES), X["train"][:256], extra={
            "method": "GBM-embedded", "seed": seed, "hyperparameters": {**cfg_sel, **mcfg["embedded"]["fixed"]},
            "trained_on": "train" + (" (fast)" if fast else ""), "git_commit": git_commit(), "selftest": selftest})
        path = model_paths(fast, seeds)[i]
        sha = write_model(model, path)
        joblib.dump(clf, mod_dir / f"gbm_embedded_seed{seed}.joblib", compress=3)
        w = Float32Walker(model)
        walkers.append(w)
        p_w = w.predict_proba(X["val"])
        lab_w, lab_sk = w.predict(X["val"]), clf.predict(X["val"])
        dis = np.where(lab_w != lab_sk)[0]
        tc = threshold_check(clf, X["val"])
        e = {"seed": seed, "max_abs_proba_diff": float(np.abs(p_w - p_sk).max()),
             "label_agreement": float((lab_w == lab_sk).mean()), "n_disagree": int(len(dis)),
             "disagreements": [{"val_index": int(k), "walker": CLASSES[lab_w[k]], "sklearn": CLASSES[lab_sk[k]],
                                "sklearn_proba_top2": sorted(p_sk[k].tolist())[-2:]} for k in dis[:10]],
             "threshold_check": tc,
             "pass": bool(np.abs(p_w - p_sk).max() <= 1e-3 and len(dis) == 0 and tc["mismatches"] == 0)}
        equiv.append(e)
        exported.append({"seed": seed, "path": path.relative_to(ROOT).as_posix(),
                         "sha256": sha, "size_bytes": model["size_bytes"], "n_nodes": model["n_nodes"],
                         "n_trees": model["n_trees"], "max_depth": model["max_depth"],
                         "sum_max_depth": model["sum_max_depth"],
                         "joblib": f"gbm_embedded_seed{seed}.joblib",
                         "joblib_sha256": sha256_file(mod_dir / f"gbm_embedded_seed{seed}.joblib")})
        print(f"  seed {seed}: {path.name} {model['size_bytes']} B, {model['n_nodes']} nodes; equivalence max|dp| "
              f"{e['max_abs_proba_diff']:.2e}, labels {100 * e['label_agreement']:.2f} %, threshold mismatches "
              f"{tc['mismatches']}/{tc['comparisons']} ({tc['thresholds_stepped_down']} thresholds stepped down)")
        if i == 0:
            h, c = write_c_tables(model, sha, c_generated_root(fast), source=f"models/{'fast/' if fast else ''}{path.name}")
            print(f"  C tables -> {h.parent}")

    # ---------------------------------------------------------------- 3. GBM-unconstrained reference
    t0 = time.perf_counter()
    fits_u = grid_unconstrained(mcfg, X["train"], y["train"], X["val"], y["val"])
    rt["unconstrained_grid_s"] = round(time.perf_counter() - t0, 1)
    summ_u = summarise(fits_u, ["max_leaf_nodes", "learning_rate"])
    summ_u = summ_u.merge(fits_u.groupby(["max_leaf_nodes", "learning_rate"])["n_iter"].agg(["min", "max"])
                          .rename(columns={"min": "n_iter_min", "max": "n_iter_max"}).reset_index())
    sel_u = select_unconstrained(summ_u)
    unc = fit_seeds("unconstrained", sel_u["config"], mcfg, X["train"], y["train"])
    unc_files = []
    for seed, clf in zip(seeds, unc):
        p = mod_dir / f"gbm_unconstrained_seed{seed}.joblib"
        joblib.dump(clf, p, compress=3)
        unc_files.append({"seed": seed, "path": p.name, "sha256": sha256_file(p), "n_iter": int(clf.n_iter_),
                          "size_kb": round(p.stat().st_size / 1024, 1)})
    print(f"unconstrained: selected {sel_u['config']} val F1 {sel_u['val_f1_mean']:.4f} +/- {sel_u['val_f1_sd']:.4f}, "
          f"{sel_u['bytes_max'] / 1024:.0f} KB equivalent ({rt['unconstrained_grid_s']} s)")

    # ---------------------------------------------------------------- val reports
    th = json.loads((rep_dir / "m2_baseline_val.json").read_text(encoding="utf-8"))["thresholds"]["tuned"]
    rep_rules = report(ys["val"], classify(F["val"], th)["rule_label"].to_numpy())
    t0 = time.perf_counter()
    lab_emb = [w.predict(X["val"]) for w in walkers]
    rt["walker_ms_per_record"] = round(1e3 * (time.perf_counter() - t0) / (len(walkers) * len(X["val"])), 4)
    t0 = time.perf_counter()
    for clf in emb:
        clf.predict(X["val"])
    rt["sklearn_gbc_ms_per_record"] = round(1e3 * (time.perf_counter() - t0) / (len(emb) * len(X["val"])), 4)
    reps_emb = [report(ys["val"], decode(lab)) for lab in lab_emb]
    reps_unc = [report(ys["val"], decode(c.predict(X["val"]))) for c in unc]
    paths = [walkers[0].path_lengths(X["val"])]

    def seed_stats(reps):
        f1 = np.array([r["macro_f1"] for r in reps])
        return {"macro_f1": f1.tolist(), "macro_f1_mean": float(f1.mean()), "macro_f1_sd": float(f1.std(ddof=1)),
                "false_alarm_rate": [r["false_alarm_rate"] for r in reps], "miss_rate": [r["miss_rate"] for r in reps],
                "per_class_f1_mean": {c: float(np.mean([r["per_class"][c]["f1"] for r in reps])) for c in CLASSES}}

    print(f"VAL macro-F1: rules {rep_rules['macro_f1']:.4f}; GBM-embedded {seed_stats(reps_emb)['macro_f1_mean']:.4f} "
          f"+/- {seed_stats(reps_emb)['macro_f1_sd']:.4f}; GBM-unconstrained {seed_stats(reps_unc)['macro_f1_mean']:.4f}")
    print(confusion_text(reps_emb[0]))

    # ---------------------------------------------------------------- 4. pre-event-only control
    t0 = time.perf_counter()
    ctrl = {}
    for s in ("train", "val"):
        Xw, meta = load_waveforms(s, fast=fast)
        Fc = extract_features(Xw, bcfg["features"], lsb=meta["lsb_pu"], n_samples=750, ref_end=749, obs_start=0)
        ctrl[s] = (model_input(Fc), nonfinite_counts(Fc))
    ctrl_models = fit_seeds("embedded", cfg_sel, mcfg, ctrl["train"][0], y["train"])
    ctrl_f1 = [report(ys["val"], decode(c.predict(ctrl["val"][0])))["macro_f1"] for c in ctrl_models]
    rt["control_s"] = round(time.perf_counter() - t0, 1)
    control = {"config": cfg_sel, "macro_f1": ctrl_f1, "macro_f1_mean": float(np.mean(ctrl_f1)),
               "macro_f1_sd": float(np.std(ctrl_f1, ddof=1)), "chance": 1 / len(CLASSES), "threshold": 0.25,
               "pass": bool(np.mean(ctrl_f1) <= 0.25), "nonfinite_train": ctrl["train"][1],
               "note": "features of the first 750 samples (OBS = REF); non-finite -> 0.0 (all-NaN columns become constant)"}
    print(f"pre-event-only control (GBM-embedded config): macro-F1 {control['macro_f1_mean']:.3f} +/- "
          f"{control['macro_f1_sd']:.3f} ({'PASS' if control['pass'] else 'FAIL'} <= 0.25)")

    # ---------------------------------------------------------------- permutation importance (val)
    t0 = time.perf_counter()
    pi = permutation_importance(emb[0], X["val"], y["val"], scoring=mcfg["permutation_importance"]["scoring"],
                                n_repeats=mcfg["permutation_importance"]["n_repeats"],
                                random_state=mcfg["permutation_importance"]["seed"], n_jobs=mcfg["n_jobs"])
    rt["permutation_importance_s"] = round(time.perf_counter() - t0, 1)
    order = np.argsort(pi.importances_mean)[::-1]
    importance = [{"feature": FEATURE_NAMES[i], "mean": float(pi.importances_mean[i]), "sd": float(pi.importances_std[i])}
                  for i in order]

    # ---------------------------------------------------------------- figures
    m3_size_vs_f1(summ.to_dict(orient="records"), summ_u.to_dict(orient="records"), sel, rep_rules["macro_f1"],
                  mcfg["budget_bytes"], fast)
    m3_feature_importance(FEATURE_NAMES, pi.importances_mean, pi.importances_std,
                          f"M3: permutation importance on VAL, GBM-embedded seed {seeds[0]} (top 15)", fast)
    m3_val_confusion(rep_rules, reps_emb[0], fast)

    rt["total_s"] = round(time.perf_counter() - t_all, 1)
    out = {
        "split_evaluated": "val", "trained_on": "train", "fast": fast, "git_commit": git_commit(),
        "sklearn_version": sklearn.__version__, "features": FEATURE_NAMES, "classes": list(CLASSES),
        "input_mapping": {"nan_fill": 0.0, "nonfinite_counts": nonfinite, "decision": "D-040"},
        "embedded": {
            "estimator": "GradientBoostingClassifier", "fixed": mcfg["embedded"]["fixed"],
            "budget_bytes": mcfg["budget_bytes"], "tie_tol": mcfg["tie_tol"], "seeds": seeds,
            "size_format": "14 B/node + 2 B/tree + 4 B/class (docs/MODEL_FORMAT.md)",
            "grid": summ.to_dict(orient="records"), "fits": fits.to_dict(orient="records"),
            "selected": sel, "models": exported, "equivalence_val": equiv,
            "val": {**seed_stats(reps_emb), "primary_seed": seeds[0], "report_primary": reps_emb[0]},
            "cost": {"model_bytes": exported[0]["size_bytes"], "comparisons_per_record_mean": float(paths[0].mean()),
                     "comparisons_per_record_max_val": int(paths[0].max()),
                     "comparisons_per_record_worst_case": exported[0]["sum_max_depth"],
                     "ram_bytes": 4 * (len(FEATURE_NAMES) + len(CLASSES)),
                     "softmax": f"{len(CLASSES)} expf + {len(CLASSES)} div"},
        },
        "unconstrained": {
            "estimator": "HistGradientBoostingClassifier", "fixed": mcfg["unconstrained"]["fixed"], "seeds": seeds,
            "grid": summ_u.to_dict(orient="records"), "fits": fits_u.to_dict(orient="records"), "selected": sel_u,
            "models": unc_files, "val": {**seed_stats(reps_unc), "report_primary": reps_unc[0]},
        },
        "rules_val": {"thresholds": th, "report": rep_rules},
        "pre_event_only_control": control,
        "permutation_importance_val": {"model": f"GBM-embedded seed {seeds[0]} (sklearn)", **mcfg["permutation_importance"],
                                       "ranking": importance},
        "runtimes": rt,
    }
    write_json(rep_dir / "m3_gbm_val.json", out)
    print(f"M3 done in {rt['total_s']} s -> {rep_dir / 'm3_gbm_val.json'}")
    ok = all(e["pass"] for e in equiv) and control["pass"]
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
