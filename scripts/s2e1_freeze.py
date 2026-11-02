"""S2E1-freeze: write configs/s2e1_frozen.yaml from reports/s2e1_train.json (train_wide / val_wide results only).

Usage: python scripts/s2e1_freeze.py [--draft PATH]
Records: the S2E1-prereg commit, the Stage 1 freeze it relies on (R, G0 and the shared DSP/feature/model code; must
verify), the data provenance (seeds, config hashes, generating commit), the selected GW/GP configurations, seeds and
the SHA-256 of every exported model, the hybrid threshold, the S2E1 method files' SHA-256 and the pre-registered
evaluation plan, decision rules and hypotheses. Commit the result as "S2E1-freeze: ..." BEFORE s2e1_evaluate.py runs;
s2e1_evaluate.py refuses to load test2_* unless gridmon.s2e1.verify() passes against that commit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from gridmon import frozen as fz
from gridmon import s2e1
from gridmon.io import ROOT


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", type=Path, default=None, help="write a draft (no clean-tree check) for --dry-run")
    args = ap.parse_args(argv)
    problems = fz.verify(ROOT)
    if problems:
        print("refusing to freeze S2E1: the Stage 1 freeze does not verify:\n  " + "\n  ".join(problems))
        return 2
    tr = json.loads((ROOT / "reports" / "s2e1_train.json").read_text(encoding="utf-8"))
    data = json.loads((ROOT / "reports" / "s2e1_data_summary.json").read_text(encoding="utf-8"))
    if tr.get("fast") or data.get("fast"):
        print("refusing to freeze fast-mode outputs")
        return 2
    model_files = [m["path"] for a in s2e1.TRAINED_ARMS for m in tr["arms"][a]["models"]]
    watch = list(s2e1.S2E1_METHOD_FILES) + model_files + ["reports/s2e1_train.json", "reports/s2e1_data_summary.json",
                                                          "c/generated_s2e1", "scripts/s2e1_evaluate.py"]
    dirty = git("status", "--porcelain", "--", *watch)
    if dirty and not args.draft:
        print("refusing to freeze: S2E1 method files, models, reports or the evaluation script are not committed:\n" + dirty)
        return 2
    s2 = s2e1.load_config()
    st1 = fz.load_frozen(ROOT)
    prereg = git("log", "-1", "--format=%H", "--grep=^S2E1-prereg")
    arms: dict = {
        "R": {"source": "configs/frozen_methods.yaml rules.thresholds (Stage 1, unchanged)",
              "thresholds": st1["rules"]["thresholds"]},
        "G0": {"source": "configs/frozen_methods.yaml gbm_embedded (Stage 1, unchanged)",
               "models": [{k: m[k] for k in ("seed", "path", "sha256", "size_bytes")} for m in st1["gbm_embedded"]["models"]]},
    }
    for a in s2e1.TRAINED_ARMS:
        t = tr["arms"][a]
        arms[a] = {"recipe": "Stage 1 GBM-embedded (gridmon.ml_gbm, configs/ml.yaml unchanged)", "trained_on": "train_wide",
                   "selected_on": "val_wide", "feature_subset": t["feature_subset"], "n_features_used": t["n_features_used"],
                   "features": t["features"], "config": t["selected"]["config"], "seeds": tr["seeds"],
                   "primary_seed": tr["seeds"][0], "val_wide_macro_f1_mean": t["val_wide"]["macro_f1_mean"],
                   "pre_event_only_control_mean": t["pre_event_only_control"]["macro_f1_mean"],
                   "evaluation": "float32 reference walker (gridmon.export_gbm.Float32Walker) on the exported JSON",
                   "models": [{k: m[k] for k in ("seed", "path", "sha256", "size_bytes", "n_trees", "sum_max_depth")}
                              for m in t["models"]],
                   "c_tables": t["c_tables"].get("dir")}
    frozen = {
        "description": "S2E1 freeze (PROMPT_05 A1.3): everything that defines the S2E1 arms. Written by scripts/s2e1_freeze.py "
                       "from train_wide/val_wide results only; verified by gridmon.s2e1.verify (with the Stage 1 freeze) "
                       "before test2_shift / test2_novel are loaded.",
        "prereg_commit": prereg, "train_code_commit": git("rev-parse", "HEAD"),
        "stage1_freeze": {"path": fz.FROZEN_PATH, "freeze_commit": fz.freeze_commit(ROOT), "verified_at_s2e1_freeze": True},
        "data": {k: {f: v[f] for f in ("split_seed", "nuisance_config", "n_records", "config_hash", "git_commit", "sealed")}
                 for k, v in data["splits"].items()},
        "arms": arms,
        "hybrid": {"threshold": float(s2["hybrid"]["threshold"]), "H0": "G0 label if max proba >= threshold else R",
                   "HW": "GW label if max proba >= threshold else R", "per_seed": True},
        "evaluation_plan": s2["evaluation"], "decision_rules": s2["decision_rules"], "hypotheses": s2["hypotheses"],
        "method_files": s2e1.method_file_hashes(ROOT),
    }
    out = args.draft or (ROOT / s2e1.S2E1_FROZEN)
    out.write_text("# S2E1 freeze (PROMPT_05): written by scripts/s2e1_freeze.py; verified by gridmon.s2e1.verify.\n"
                   + yaml.safe_dump(frozen, sort_keys=False, width=120), encoding="utf-8")
    print(f"wrote {out}" + ("" if args.draft else " -- commit it as 'S2E1-freeze: ...' before scripts/s2e1_evaluate.py"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
