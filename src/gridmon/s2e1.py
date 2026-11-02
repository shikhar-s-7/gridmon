"""S2E1 (Stage 2, experiment 1): configuration, data checks, arms, freeze and statistics.

Pre-registration: docs/S2E1_PREREG.md and configs/s2e1.yaml. Stage 1 methods are used unchanged:
R = frozen rules (configs/frozen_methods.yaml), G0 = frozen GBM-embedded (seeds 0-2). New arms:
GW / GP = the GBM-embedded recipe (gridmon.ml_gbm, configs/ml.yaml, unchanged) trained on train_wide and
selected on val_wide, with all 37 features (GW) or the pre-registered physics subset (GP); H0 / HW =
confidence-gated hybrids: ML label if its max softmax probability >= 0.5, else the rules' label.
GP models are exported with feature indices remapped to the full 37-feature vector, so the float32
walker and the C tables take the same input as every other model (excluded features are never read).
Methods see only feature tables computed from the int16 waveforms; manifests are used for scoring,
breakdowns and data checks only (rule A1.5).
"""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import CLASSES
from . import frozen as fz
from .baseline_rules import classify
from .evaluation import SEVERITY, bin_labels
from .export_gbm import Float32Walker, load_model
from .features import FEATURE_NAMES
from .io import ROOT, load_dataset_config, load_yaml
from .metrics import confusion, prf, report
from .ml_gbm import decode, encode, model_input
from .stats import mcnemar_exact, stratified_indices

S2E1_CONFIG = "configs/s2e1.yaml"
S2E1_FROZEN = "configs/s2e1_frozen.yaml"
SEALED = ("test2_shift", "test2_novel")
NONSEALED = ("train_wide", "val_wide", "preview_wide", "preview_novel")
ARMS = ("R", "G0", "GW", "GP", "H0", "HW")
TRAINED_ARMS = ("GW", "GP")
# S2E1-specific method-defining files (hashed at the S2E1 freeze, on top of the Stage 1 freeze, which covers
# DSP, features, rules, units, ml_gbm, export_gbm, configs/baseline.yaml, configs/ml.yaml, Stage 1 models).
S2E1_METHOD_FILES = ("src/gridmon/s2e1.py", "configs/s2e1.yaml")
K = len(CLASSES)


# ------------------------------------------------------------------------------------------- configuration
def load_config(root: Path = ROOT) -> dict[str, Any]:
    return load_yaml(Path(root) / S2E1_CONFIG)


def dataset_config(root: Path = ROOT) -> dict[str, Any]:
    """configs/dataset.yaml (Stage 1, unchanged) with the S2E1 nuisances and splits.

    The Stage 1 'shift' nuisance is used by reference (test2_shift = Stage 1 test_shift configuration);
    every other key (record, ADC, AA FIR, events, margins, chunking, redraws) is Stage 1's."""
    base = load_dataset_config()
    s2 = load_config(root)
    cfg = copy.deepcopy(base)
    cfg["nuisance"] = {"shift": copy.deepcopy(base["nuisance"]["shift"]), **copy.deepcopy(s2["nuisance"])}
    cfg["splits"] = copy.deepcopy(s2["splits"])
    return cfg


def feature_columns(subset: str, s2: dict[str, Any] | None = None) -> list[int]:
    """Indices into FEATURE_NAMES of a pre-registered feature subset ('all37' or 'physics')."""
    s2 = s2 or load_config()
    if subset == "all37":
        return list(range(len(FEATURE_NAMES)))
    names = s2["feature_subsets"][subset]
    return [FEATURE_NAMES.index(n) for n in names]


def arm_subset(arm: str, s2: dict[str, Any] | None = None) -> str:
    s2 = s2 or load_config()
    return s2["arms"][arm]["features"]


def models_root(fast: bool = False) -> Path:
    return ROOT / "models" / "s2e1" / ("fast" if fast else "")


def c_generated_root(fast: bool = False) -> Path:
    return ROOT / "c" / "generated_s2e1" / ("fast" if fast else "")


def model_path(arm: str, seed: int, fast: bool = False) -> Path:
    return models_root(fast) / f"gbm_{arm.lower()}_seed{seed}.json"


def remap_model_features(model: dict[str, Any], cols: list[int]) -> dict[str, Any]:
    """Model trained on X[:, cols] -> same trees reading the full 37-feature vector (feature k -> cols[k])."""
    m = copy.deepcopy(model)
    feat = np.asarray(m["nodes"]["feature"], dtype=np.int64)
    m["nodes"]["feature"] = [int(cols[f]) if f >= 0 else -1 for f in feat]
    m["n_features"] = len(FEATURE_NAMES)
    m["feature_names"] = list(FEATURE_NAMES)
    m["features_used_subset"] = [FEATURE_NAMES[c] for c in cols]
    return m


# ------------------------------------------------------------------------------------------- data checks
def verify_margins(man: pd.DataFrame, mc: dict[str, Any], max_redraws: int) -> dict[str, Any]:
    """Label margins (brief B5 + D-034) re-checked from the manifest's analytic margin metrics, per class.

    The interharmonic (S2E1 novel nuisance) is a measurement nuisance like the harmonics: the margins
    are defined on the fundamental phasors and are unaffected by it (label unchanged by design)."""
    eps = 1e-12
    out: dict[str, Any] = {}
    for c in CLASSES:
        d = man[man["label"] == c]
        if c in ("sag_bal", "sag_unbal"):
            ok = d["mm_min_ll_ev"] <= mc["sag_min_ll"]
        elif c == "swell":
            ok = d["mm_max_ll_ev"] >= mc["swell_max_ll"]
        elif c == "normal":
            lo, hi = mc["normal_ll"]
            ok = ((d["mm_min_ll"] >= lo) & (d["mm_max_ll"] <= hi) & (d["mm_ang_deg"] <= mc["normal_max_ang_deg"] + eps)
                  & (d["drift"].abs() <= mc["normal_max_drift"] + eps))
        elif c == "phase_jump":
            ok = (d["ev_dphi_deg"].abs() >= mc["jump_min_deg"] - eps) & (d["mm_dv1"] <= mc["jump_max_dv1"] + eps)
        elif c == "freq_event":
            ok = d["mm_df_end"].abs() >= mc["freq_min_df"]
        else:
            lo, hi = mc["osc_ll"]
            ok = (d["ev_depth"] >= mc["osc_min_depth"] - eps) & (d["mm_min_ll"] >= lo) & (d["mm_max_ll"] <= hi)
        fallback = (d["n_reject"] > max_redraws) if c == "oscillation" else pd.Series(False, index=d.index)
        out[c] = {"n": int(len(d)), "n_margin_ok": int(ok.sum()), "n_fail": int((~ok).sum()),
                  "n_fail_not_d039_fallback": int((~ok & ~fallback).sum()), "n_d039_fallback": int(fallback.sum()),
                  "max_redraws_used": int(d["n_reject"].max()) if len(d) else 0}
    out["all_hold_except_d039_fallback"] = all(v["n_fail_not_d039_fallback"] == 0 for k, v in out.items()
                                               if isinstance(v, dict))
    return out


def verify_interharmonic(man: pd.DataFrame, icfg: dict[str, Any] | None) -> dict[str, Any]:
    """Interharmonic draws vs the pre-registered configuration (range, harmonic exclusion, share)."""
    if icfg is None:
        return {"configured": False, "manifest_columns": bool("ih_on" in man)}
    on = man["ih_on"].to_numpy().astype(bool)
    f, a = man["ih_f"].to_numpy(float)[on], man["ih_amp_pct"].to_numpy(float)[on]
    dist = np.abs(f - 50.0 * np.round(f / 50.0))
    beat = np.abs(f - 50.0)
    lo, hi = icfg["freq_hz"]
    return {"configured": True, "n": int(len(man)), "n_on": int(on.sum()), "share_on": float(on.mean()),
            "share_on_by_class": {c: float(on[man["label"].to_numpy() == c].mean()) for c in CLASSES},
            "f_min": float(f.min()) if on.any() else None, "f_max": float(f.max()) if on.any() else None,
            "f_in_range": bool(((f >= lo) & (f <= hi)).all()),
            "min_distance_to_harmonic_hz": float(dist.min()) if on.any() else None,
            "harmonic_exclusion_holds": bool((dist >= icfg["exclude_harmonic_hz"] - 1e-9).all()),
            "amp_pct_min": float(a.min()) if on.any() else None, "amp_pct_max": float(a.max()) if on.any() else None,
            "n_beat_lt_25hz": int((beat < 25.0).sum()), "n_beat_ge_25hz": int((beat >= 25.0).sum())}


# ------------------------------------------------------------------------------------------- arms
def hybrid(ml_labels: np.ndarray, ml_proba: np.ndarray, rules_labels: np.ndarray, threshold: float) -> np.ndarray:
    """Confidence-gated hybrid: ML label where max_k p_k >= threshold, else the rules' label."""
    use_ml = ml_proba.max(axis=1) >= threshold
    return np.where(use_ml, ml_labels, rules_labels)


class Arms:
    """R, G0 (Stage 1 frozen) and GW, GP (S2E1 frozen) plus the hybrids H0, HW."""

    def __init__(self, stage1: dict[str, Any], s2frozen: dict[str, Any], root: Path = ROOT,
                 model_dir_override: Path | None = None, stage1_model_dir_override: Path | None = None):
        self.thresholds = dict(stage1["rules"]["thresholds"])
        self.hybrid_threshold = float(s2frozen["hybrid"]["threshold"])

        def walkers(models: list[dict[str, Any]], override: Path | None) -> list[Float32Walker]:
            return [Float32Walker(load_model((override / Path(m["path"]).name) if override else root / m["path"]))
                    for m in models]

        self.walkers = {"G0": walkers(stage1["gbm_embedded"]["models"], stage1_model_dir_override)}
        for arm in TRAINED_ARMS:
            self.walkers[arm] = walkers(s2frozen["arms"][arm]["models"], model_dir_override)
        self.seeds = {a: [int(m["seed"]) for m in (stage1["gbm_embedded"]["models"] if a == "G0"
                                                   else s2frozen["arms"][a]["models"])] for a in self.walkers}

    def predict(self, F: pd.DataFrame) -> dict[str, Any]:
        """Labels per arm (list over seeds; R has one), ML probabilities, rule reasons."""
        r = classify(F, self.thresholds)
        rules = r["rule_label"].to_numpy()
        X = model_input(F)
        out: dict[str, Any] = {"R": [rules], "rules_reason": r["rule_reason_name"].to_numpy(), "proba": {}}
        for arm, ws in self.walkers.items():
            raws = [w.raw(X) for w in ws]
            out["proba"][arm] = [Float32Walker.softmax32(x) for x in raws]
            out[arm] = [decode(np.argmax(x, axis=1)) for x in raws]
        for h, ml in (("H0", "G0"), ("HW", "GW")):
            out[h] = [hybrid(lab, p, rules, self.hybrid_threshold) for lab, p in zip(out[ml], out["proba"][ml])]
            out[f"{h}_ml_share"] = [float((p.max(axis=1) >= self.hybrid_threshold).mean()) for p in out["proba"][ml]]
        return out


# ------------------------------------------------------------------------------------------- freeze
def method_file_hashes(root: Path = ROOT) -> dict[str, str]:
    return {f: fz.sha256(Path(root) / f) for f in S2E1_METHOD_FILES}


def load_frozen(root: Path = ROOT) -> dict[str, Any]:
    p = Path(root) / S2E1_FROZEN
    if not p.exists():
        raise fz.FreezeError(f"{S2E1_FROZEN} not found: S2E1 arms are not frozen (run scripts/s2e1_freeze.py)")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def freeze_commit(root: Path = ROOT) -> str | None:
    """Last commit that touched configs/s2e1_frozen.yaml (the S2E1-freeze commit), or None."""
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%H", "--", S2E1_FROZEN], cwd=root, capture_output=True,
                             text=True, check=True).stdout.strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def verify(root: Path = ROOT, frozen: dict[str, Any] | None = None, check_models: bool = True,
           check_git: bool = True) -> list[str]:
    """Problems (empty = OK): the Stage 1 freeze (R, G0, shared DSP/feature/model code) AND the S2E1 freeze
    (S2E1 method files, GW/GP model files, all unchanged since the S2E1-freeze commit)."""
    root = Path(root)
    problems = [f"stage 1: {p}" for p in fz.verify(root, check_models=check_models, check_git=check_git)]
    try:
        frozen = frozen or load_frozen(root)
    except fz.FreezeError as e:
        return problems + [str(e)]
    for f, h in frozen.get("method_files", {}).items():
        p = root / f
        if not p.exists():
            problems.append(f"missing S2E1 method file {f}")
        elif fz.sha256(p) != h:
            problems.append(f"S2E1 method file changed since the freeze: {f}")
    for f in S2E1_METHOD_FILES:
        if f not in frozen.get("method_files", {}):
            problems.append(f"S2E1 method file not listed in the freeze: {f}")
    model_paths = [m["path"] for a in TRAINED_ARMS for m in frozen.get("arms", {}).get(a, {}).get("models", [])]
    if check_models:
        for a in TRAINED_ARMS:
            for m in frozen.get("arms", {}).get(a, {}).get("models", []):
                p = root / m["path"]
                if not p.exists():
                    problems.append(f"missing S2E1 model file {m['path']}")
                elif fz.sha256(p) != m["sha256"]:
                    problems.append(f"S2E1 model file changed since the freeze: {m['path']}")
    if check_git:
        fc = freeze_commit(root)
        if fc is None:
            problems.append("no S2E1-freeze commit found (configs/s2e1_frozen.yaml is not committed)")
        else:
            paths = list(frozen.get("method_files", {})) + [S2E1_FROZEN] + (model_paths if check_models else [])
            r = subprocess.run(["git", "diff", "--name-only", fc, "--", *paths], cwd=root, capture_output=True, text=True)
            if r.returncode != 0:
                problems.append(f"git diff against the S2E1-freeze commit failed: {r.stderr.strip()[:200]}")
            for x in [x for x in r.stdout.split() if x]:
                problems.append(f"working tree differs from the S2E1-freeze commit {fc[:7]}: {x}")
    return problems


def require_frozen(root: Path = ROOT, **kw) -> dict[str, Any]:
    problems = verify(root, **kw)
    if problems:
        raise fz.FreezeError("refusing to evaluate the S2E1 sealed sets:\n  " + "\n  ".join(problems))
    return load_frozen(root)


# ------------------------------------------------------------------------------------------- statistics
def macro_f1_boot(y: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Macro-F1 of encoded predictions on every resample row of idx [B, N] (vectorised confusion counts)."""
    B, N = idx.shape
    flat = (y[idx] * K + pred[idx]) + (np.arange(B) * K * K)[:, None]
    C = np.bincount(flat.ravel(), minlength=B * K * K).reshape(B, K, K).astype(float)
    tp = np.einsum("bkk->bk", C)
    col, row = C.sum(axis=1), C.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(col > 0, tp / col, 0.0)
        r = np.where(row > 0, tp / row, 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return f.mean(axis=1)


def full_macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(prf(confusion(y, pred, K))[2].mean())


def ci95(d: np.ndarray) -> list[float]:
    lo, hi = np.percentile(d, [2.5, 97.5])
    return [float(lo), float(hi)]


def reading(ci: list[float]) -> str:
    """'above 0' / 'includes 0' / 'below 0' for a CI of (arm - reference)."""
    return "above 0" if ci[0] > 0 else ("below 0" if ci[1] < 0 else "includes 0")


def compare_arms(y: np.ndarray, preds_a: list[np.ndarray], preds_b: list[np.ndarray], boot: dict[str, np.ndarray],
                 name_a: str, name_b: str) -> dict[str, Any]:
    """Paired comparison on one set: seed-mean macro-F1 difference with the shared stratified resamples
    (boot: per-arm [S, B] macro-F1 arrays), per-seed CIs, exact McNemar per seed (seed k vs seed k, or vs R)."""
    fa, fb = boot[name_a], boot[name_b]
    d = fa.mean(axis=0) - fb.mean(axis=0)
    full = float(np.mean([full_macro_f1(y, p) for p in preds_a]) - np.mean([full_macro_f1(y, p) for p in preds_b]))
    per_seed, mc = [], []
    for s, pa in enumerate(preds_a):
        sb = s if len(preds_b) > 1 else 0
        per_seed.append({"diff": full_macro_f1(y, pa) - full_macro_f1(y, preds_b[sb]), "ci95": ci95(fa[s] - fb[sb])})
        mc.append(mcnemar_exact(pa == y, preds_b[sb] == y))
    ci = ci95(d)
    return {"a": name_a, "b": name_b, "diff_seed_mean": full, "ci95_seed_mean": ci, "reading": reading(ci),
            "frac_resamples_diff_le_0": float((d <= 0).mean()), "per_seed": per_seed, "mcnemar_per_seed": mc}


def survival(y: np.ndarray, preds: list[np.ndarray], mask: np.ndarray, n_boot: int, rng: np.random.Generator,
             margin: float) -> dict[str, Any]:
    """Unseen-nuisance survival: delta = macro-F1(mask) - macro-F1(~mask), seed-mean; unpaired bootstrap
    stratified by class within each subset. survives: CI lower >= -margin; fails: CI upper < 0 and
    delta <= -margin; otherwise inconclusive."""
    y1, y0 = y[mask], y[~mask]
    i1, i0 = stratified_indices(y1, n_boot, rng), stratified_indices(y0, n_boot, rng)
    f1 = np.mean([macro_f1_boot(y1, p[mask], i1) for p in preds], axis=0)
    f0 = np.mean([macro_f1_boot(y0, p[~mask], i0) for p in preds], axis=0)
    full1 = float(np.mean([full_macro_f1(y1, p[mask]) for p in preds]))
    full0 = float(np.mean([full_macro_f1(y0, p[~mask]) for p in preds]))
    delta = full1 - full0
    ci = ci95(f1 - f0)
    verdict = "survives" if ci[0] >= -margin else ("fails" if (ci[1] < 0 and delta <= -margin) else "inconclusive")
    return {"n_with": int(mask.sum()), "n_without": int((~mask).sum()), "f1_with": full1, "f1_without": full0,
            "delta": delta, "ci95": ci, "margin": margin, "verdict": verdict}


def decision(ci_by_set: dict[str, list[float]]) -> str:
    """Pre-registered decision rule on the CIs of (arm - R) over both sealed sets."""
    cis = list(ci_by_set.values())
    if any(c[1] < 0 for c in cis):
        return "rules better"
    if all(c[0] > 0 for c in cis):
        return "ML better"
    return "ML matches"


# ------------------------------------------------------------------------------------------- metrics, breakdowns
def arm_metrics(y: np.ndarray, preds: list[np.ndarray]) -> dict[str, Any]:
    """Per-seed reports, mean/sd of headline metrics, per-class F1 (seed mean), critical confusions (seed mean)."""
    reps = [report(y, p) for p in preds]

    def ms(key: str) -> dict[str, Any]:
        v = np.array([r[key] for r in reps])
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "per_seed": v.tolist()}

    crit = list(reps[0]["critical_confusions"])
    return {"macro_f1": ms("macro_f1"), "accuracy": ms("accuracy"), "false_alarm_rate": ms("false_alarm_rate"),
            "miss_rate": ms("miss_rate"),
            "per_class_f1_mean": {c: float(np.mean([r["per_class"][c]["f1"] for r in reps])) for c in CLASSES},
            "critical_confusions_mean": {k: float(np.mean([r["critical_confusions"][k] for r in reps])) for k in crit},
            "primary": reps[0], "n_seeds": len(reps)}


def bins_metric(y: np.ndarray, preds: list[np.ndarray], groups: np.ndarray, names: list[str],
                kind: str = "macro_f1") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for g in names:
        sel = groups == g
        n = int(sel.sum())
        if n == 0:
            out[g] = {"n": 0, "value": None}
        elif kind == "macro_f1":
            out[g] = {"n": n, "value": float(np.mean([report(y[sel], p[sel])["macro_f1"] for p in preds])),
                      "accuracy": float(np.mean([(y[sel] == p[sel]).mean() for p in preds])),
                      "errors_primary": int((y[sel] != preds[0][sel]).sum())}
        else:
            out[g] = {"n": n, "value": float(np.mean([(y[sel] == p[sel]).mean() for p in preds])),
                      "errors_primary": int((y[sel] != preds[0][sel]).sum())}
    return out


def breakdowns(y: np.ndarray, P: dict[str, Any], man: pd.DataFrame, ev: dict[str, Any]) -> dict[str, Any]:
    """SNR, severity, transient and interharmonic breakdowns for every arm (manifest used for grouping only)."""
    out: dict[str, Any] = {}
    snr_lab, snr_names = bin_labels(man["snr_db"].to_numpy(), ev["snr_bins_db"])
    out["snr"] = {"bins": snr_names, **{a: bins_metric(y, P[a], snr_lab, snr_names) for a in ARMS}}
    sev = {}
    for c, (col, tf, edges, unit) in SEVERITY.items():
        x = man[col].to_numpy(dtype=float)
        x = np.abs(x) if tf == "abs" else (100 * x if tf == "pct" else x)
        lab, names = bin_labels(x, edges)
        lab = np.where(y == c, lab, "")
        sev[c] = {"variable": unit, "bins": names, **{a: bins_metric(y, P[a], lab, names, "recall") for a in ARMS}}
    out["severity_recall"] = sev
    tr = man["tr_on"].to_numpy().astype(int) if "tr_on" in man else np.zeros(len(y), int)
    trl = np.where(tr == 1, "with_transient", "without_transient")
    out["transient"] = {"n_with": int(tr.sum()), **{a: bins_metric(y, P[a], trl, ["without_transient", "with_transient"])
                                                     for a in ARMS}}
    if "ih_on" in man:
        ih = man["ih_on"].to_numpy().astype(bool)
        ihl = np.where(ih, "with_interharmonic", "without_interharmonic")
        names = ["without_interharmonic", "with_interharmonic"]
        out["interharmonic"] = {"n_with": int(ih.sum()), **{a: bins_metric(y, P[a], ihl, names) for a in ARMS}}
        beat = np.abs(man["ih_f"].to_numpy(dtype=float) - 50.0)
        prim = np.where(~ih, "", np.where(beat < 25.0, "beat_lt_25", "beat_ge_25"))
        out["interharmonic_beat_primary"] = {"bins": ["beat_lt_25", "beat_ge_25"], "variable": "|f_ih - 50| (Hz)",
                                             **{a: bins_metric(y, P[a], prim, ["beat_lt_25", "beat_ge_25"], "recall")
                                                for a in ARMS}}
        lab, names = bin_labels(np.where(ih, beat, np.nan), ev["breakdowns"]["interharmonic_beat_secondary_edges_hz"])
        out["interharmonic_beat_secondary"] = {"bins": names, "variable": "|f_ih - 50| (Hz)",
                                               **{a: bins_metric(y, P[a], lab, names, "recall") for a in ARMS}}
        # what the errors on interharmonic records turn into (primary seed)
        out["interharmonic_errors_primary"] = {
            a: {f"{t}->{p}": int(n) for (t, p), n in pd.Series(list(zip(y[ih & (y != P[a][0])], P[a][0][ih & (y != P[a][0])])))
                .value_counts().items()} for a in ARMS}
    return out


def encode_all(P: dict[str, Any]) -> dict[str, list[np.ndarray]]:
    return {a: [encode(p) for p in P[a]] for a in ARMS}
