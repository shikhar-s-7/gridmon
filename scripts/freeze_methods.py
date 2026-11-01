"""M4-freeze: write configs/frozen_methods.yaml from the M2/M3 outputs (TRAIN/VAL only).

Usage: python scripts/freeze_methods.py
The method files, models and M2/M3 reports must be committed (feature_code_commit = HEAD).
Records the train-tuned rule thresholds (D-036), the selected GBM-embedded configuration, seeds and
the SHA-256 of every exported model file, the GBM-unconstrained configuration and model files, the
SHA-256 of every method-defining file (gridmon.frozen.METHOD_GLOBS), the feature-code commit, and the
pre-registered evaluation plan. Commit the result as "M4-freeze: ..." BEFORE running evaluate_test.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from gridmon.frozen import FROZEN_PATH, file_hashes, sha256
from gridmon.io import ROOT, load_baseline_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", type=Path, default=None, help="write a draft (no clean-tree check) for evaluate_test --dry-run")
    args = ap.parse_args(argv)
    files = list(file_hashes(ROOT)) + ["models", "reports/m2_baseline_val.json", "reports/m3_gbm_val.json"]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *files], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    if dirty and not args.draft:
        print("refusing to freeze: method files, models or M2/M3 reports have uncommitted changes:\n" + dirty)
        return 2
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    m2 = json.loads((ROOT / "reports" / "m2_baseline_val.json").read_text(encoding="utf-8"))
    m3 = json.loads((ROOT / "reports" / "m3_gbm_val.json").read_text(encoding="utf-8"))
    if m2.get("fast") or m3.get("fast"):
        print("refusing to freeze fast-mode outputs")
        return 2
    bdiag = load_baseline_config()["diagnostics"]
    emb, unc = m3["embedded"], m3["unconstrained"]
    th = m2["thresholds"]["tuned"]
    frozen = {
        "description": "M4 freeze (PROMPT_02 A1.4): everything that defines the methods. Written by scripts/freeze_methods.py "
                       "from TRAIN/VAL results only; verified by gridmon.frozen.verify before any test split is loaded.",
        "feature_code_commit": head,
        "rules": {"thresholds": {k: float(v) for k, v in th.items()}, "source": "reports/m2_baseline_val.json "
                  "thresholds.tuned (TRAIN-tuned, D-036)", "train_macro_f1": m2["thresholds"]["tuning"]["train_macro_f1"],
                  "val_macro_f1": m2["val"]["tuned"]["macro_f1"]},
        "gbm_embedded": {
            "estimator": "sklearn.ensemble.GradientBoostingClassifier", "config": emb["selected"]["config"],
            "fixed": emb["fixed"], "seeds": emb["seeds"], "primary_seed": emb["seeds"][0],
            "evaluation": "float32 reference walker (gridmon.export_gbm.Float32Walker) on the exported JSON",
            "input_nan_fill": 0.0, "val_macro_f1_mean": emb["val"]["macro_f1_mean"],
            "models": [{"seed": m["seed"], "path": m["path"], "sha256": m["sha256"], "size_bytes": m["size_bytes"],
                        "n_trees": m["n_trees"], "sum_max_depth": m["sum_max_depth"]} for m in emb["models"]],
            "sklearn_pickles": [{"seed": m["seed"], "path": f"models/{m['joblib']}", "sha256": m["joblib_sha256"]}
                                for m in emb["models"]],
        },
        "gbm_unconstrained": {
            "estimator": "sklearn.ensemble.HistGradientBoostingClassifier", "config": unc["selected"]["config"],
            "fixed": unc["fixed"], "seeds": unc["seeds"], "val_macro_f1_mean": unc["val"]["macro_f1_mean"],
            "bytes_equivalent": unc["selected"]["bytes_max"],
            "models": [{"seed": m["seed"], "path": f"models/{m['path']}", "sha256": m["sha256"]} for m in unc["models"]],
        },
        "evaluation_plan": {
            "splits": ["test_iid", "test_shift"], "primary_metric": "macro_f1",
            "bootstrap_resamples": 2000, "bootstrap_stratified_by_class": True, "bootstrap_seed": 20261104,
            "mcnemar": "exact binomial, two-sided, per seed", "snr_bins_db": [25, 30, 35, 40, 50, 60],
            "jump_bins_deg": bdiag["jump_bins_deg"], "naive_pll_hz": bdiag["naive_pll_hz"],
            "severity_bins": "gridmon.evaluation.SEVERITY",
        },
        "method_files": file_hashes(ROOT),
        "sklearn_version": m3["sklearn_version"],
    }
    for m in frozen["gbm_embedded"]["models"] + frozen["gbm_unconstrained"]["models"]:
        assert sha256(ROOT / m["path"]) == m["sha256"], m["path"]
    out = args.draft or ROOT / FROZEN_PATH
    out.write_text("# " + frozen["description"] + "\n" + yaml.safe_dump(frozen, sort_keys=False, width=110), encoding="utf-8")
    print(f"wrote {out}: {len(frozen['method_files'])} method files, "
          f"{len(frozen['gbm_embedded']['models']) + len(frozen['gbm_unconstrained']['models'])} model files; feature code {head[:7]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
