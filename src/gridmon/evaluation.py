"""M4 evaluation library: run the frozen methods on a feature table and compute the B9 metrics.

Methods see only the feature table (computed from waveforms); the manifest is used here for metrics
and breakdowns only (rule A1.5). Conventions: ML methods are reported as mean +/- sd over the frozen
seeds; confusion matrices and per-record figures use the primary seed (the exported C model);
breakdown tables use the seed-mean of the per-bin metric.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import CLASSES
from .baseline_rules import classify
from .export_gbm import Float32Walker, load_model
from .metrics import report
from .ml_gbm import decode, encode, model_input
from .stats import mcnemar_exact, paired_bootstrap, unpaired_bootstrap_change

METHODS = ("rules", "gbm_embedded", "gbm_unconstrained")
K = len(CLASSES)


class FrozenMethods:
    """The three frozen methods, loaded from configs/frozen_methods.yaml (paths relative to root)."""

    def __init__(self, frozen: dict[str, Any], root: Path, model_dir_override: Path | None = None):
        import joblib

        self.frozen = frozen
        self.thresholds = dict(frozen["rules"]["thresholds"])
        emb, unc = frozen["gbm_embedded"], frozen["gbm_unconstrained"]
        self.seeds = list(emb["seeds"])

        def path(p: str) -> Path:
            return (model_dir_override / Path(p).name) if model_dir_override else root / p

        self.walkers = [Float32Walker(load_model(path(m["path"]))) for m in emb["models"]]
        self.hgb = [joblib.load(path(m["path"])) for m in unc["models"]]
        self.timing: dict[str, float] = {}

    def predict(self, F: pd.DataFrame) -> dict[str, Any]:
        """Labels (string arrays; lists over seeds for ML) and GBM-embedded probabilities."""
        n = len(F)
        t0 = time.perf_counter()
        rules = classify(F, self.thresholds)
        self.timing["rules_ms_per_record"] = 1e3 * (time.perf_counter() - t0) / n
        X = model_input(F)
        t0 = time.perf_counter()
        emb_raw = [w.raw(X) for w in self.walkers]
        self.timing["gbm_embedded_walker_ms_per_record"] = 1e3 * (time.perf_counter() - t0) / (n * len(self.walkers))
        t0 = time.perf_counter()
        unc_lab = [c.predict(X) for c in self.hgb]
        self.timing["gbm_unconstrained_ms_per_record"] = 1e3 * (time.perf_counter() - t0) / (n * len(self.hgb))
        return {
            "rules": [rules["rule_label"].to_numpy()],
            "rules_reason": rules["rule_reason_name"].to_numpy(),
            "gbm_embedded": [decode(np.argmax(r, axis=1)) for r in emb_raw],
            "gbm_embedded_proba": [Float32Walker.softmax32(r) for r in emb_raw],
            "gbm_unconstrained": [decode(lab) for lab in unc_lab],
            "gbm_embedded_comparisons": self.walkers[0].path_lengths(X),
        }


def method_metrics(y: np.ndarray, preds: list[np.ndarray]) -> dict[str, Any]:
    """Per-seed reports, seed mean/sd of headline metrics, primary-seed report."""
    reps = [report(y, p) for p in preds]
    f1 = np.array([r["macro_f1"] for r in reps])

    def ms(key):
        v = np.array([r[key] for r in reps])
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "per_seed": v.tolist()}

    crit_keys = list(reps[0]["critical_confusions"])
    return {
        "macro_f1": ms("macro_f1"), "accuracy": ms("accuracy"), "false_alarm_rate": ms("false_alarm_rate"),
        "miss_rate": ms("miss_rate"),
        "per_class_mean": {c: {m: float(np.mean([r["per_class"][c][m] for r in reps])) for m in ("precision", "recall", "f1")}
                           for c in CLASSES},
        "critical_confusions_mean": {k: float(np.mean([r["critical_confusions"][k] for r in reps])) for k in crit_keys},
        "primary": reps[0], "n_seeds": len(reps), "macro_f1_seeds": f1.tolist(),
    }


def _bins_metric(y: np.ndarray, preds: list[np.ndarray], groups: np.ndarray, labels: list[str], kind: str) -> dict:
    """Per-bin metric (seed mean): kind 'macro_f1' (all classes in the bin) or 'recall'."""
    out = {}
    for g in labels:
        sel = groups == g
        n = int(sel.sum())
        if n == 0:
            out[g] = {"n": 0, "value": None}
            continue
        if kind == "macro_f1":
            vals = [report(y[sel], p[sel])["macro_f1"] for p in preds]
            acc = [float((y[sel] == p[sel]).mean()) for p in preds]
            out[g] = {"n": n, "value": float(np.mean(vals)), "accuracy": float(np.mean(acc))}
        else:
            vals = [float((y[sel] == p[sel]).mean()) for p in preds]
            out[g] = {"n": n, "value": float(np.mean(vals)), "errors_primary": int((y[sel] != preds[0][sel]).sum())}
    return out


def bin_labels(x: np.ndarray, edges: list[float], fmt: str = "{:g}") -> tuple[np.ndarray, list[str]]:
    """Assign values to [e_i, e_{i+1}) bins (last bin closed); returns (labels per record, bin names)."""
    names = [f"{fmt.format(a)}-{fmt.format(b)}" for a, b in zip(edges[:-1], edges[1:])]
    lab = np.full(len(x), "", dtype=object)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        sel = (x >= a) & ((x <= b) if last else (x < b))
        lab[sel] = names[i]
    return lab, names


SEVERITY = {  # class -> (manifest column, transform, bin edges, unit)
    "sag_bal": ("mm_min_ll_ev", None, [0.0, 0.4, 0.7, 0.87], "min LL fundamental in the dip (pu)"),
    "sag_unbal": ("mm_min_ll_ev", None, [0.0, 0.4, 0.7, 0.87], "min LL fundamental in the dip (pu)"),
    "swell": ("mm_max_ll_ev", None, [1.13, 1.2, 1.3, 1.45], "max LL fundamental in the swell (pu)"),
    "phase_jump": ("ev_dphi_deg", "abs", [6.0, 12.0, 20.0, 30.0], "abs(dphi) (deg)"),
    "freq_event": ("ev_rocof", "abs", [0.5, 1.0, 2.0, 4.0], "abs(ROCOF) (Hz/s)"),
    "oscillation": ("ev_depth", "pct", [2.0, 4.0, 6.0, 8.0], "modulation depth (%)"),
}


def breakdowns(y: np.ndarray, P: dict[str, Any], man: pd.DataFrame, plan: dict[str, Any]) -> dict[str, Any]:
    """SNR, severity and shift-transient breakdowns for every method."""
    out: dict[str, Any] = {}
    snr_lab, snr_names = bin_labels(man["snr_db"].to_numpy(), plan["snr_bins_db"])
    out["snr"] = {"bins": snr_names, **{m: _bins_metric(y, P[m], snr_lab, snr_names, "macro_f1") for m in METHODS}}
    sev = {}
    for c, (col, tf, edges, unit) in SEVERITY.items():
        sel = y == c
        x = man[col].to_numpy(dtype=float)
        x = np.abs(x) if tf == "abs" else (100 * x if tf == "pct" else x)
        lab, names = bin_labels(x, edges)
        lab = np.where(sel, lab, "")
        sev[c] = {"variable": unit, "bins": names,
                  **{m: _bins_metric(y, P[m], lab, names, "recall") for m in METHODS}}
    out["severity_recall"] = sev
    tr = man["tr_on"].to_numpy().astype(int) if "tr_on" in man else np.zeros(len(y), int)
    trl = np.where(tr == 1, "with_transient", "without_transient")
    out["shift_transient"] = {"n_with": int(tr.sum()),
                              **{m: _bins_metric(y, P[m], trl, ["without_transient", "with_transient"], "macro_f1")
                                 for m in METHODS}}
    return out


def naive_pll(F: pd.DataFrame, y: np.ndarray, man: pd.DataFrame, thr: float, jump_bins: list[float]) -> dict[str, Any]:
    flag = F["dfpll_max"].to_numpy() > thr
    by_class = {c: {"flagged": int(flag[y == c].sum()), "n": int((y == c).sum()), "rate": float(flag[y == c].mean())}
                for c in CLASSES}
    lab, names = bin_labels(np.abs(man["ev_dphi_deg"].fillna(0).to_numpy()), jump_bins)
    pj = y == "phase_jump"
    by_jump = {g: {"flagged": int(flag[pj & (lab == g)].sum()), "n": int((pj & (lab == g)).sum())} for g in names}
    return {"threshold_hz": thr, "by_class": by_class, "by_jump_deg": by_jump}


def compare(y: np.ndarray, P: dict[str, Any], plan: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Paired bootstrap CIs and McNemar tests between methods on one split."""
    yi = encode(y)
    enc = {m: [encode(p) for p in P[m]] for m in METHODS}
    nb = int(plan["bootstrap_resamples"])
    out = {}
    for a, b in (("gbm_embedded", "rules"), ("gbm_unconstrained", "rules"), ("gbm_unconstrained", "gbm_embedded")):
        bs = paired_bootstrap(yi, enc[a], enc[b][0], K, nb, rng)
        mc = [mcnemar_exact(p == yi, enc[b][0] == yi) for p in enc[a]]
        out[f"{a}_minus_{b}"] = {"bootstrap": bs, "mcnemar_per_seed": mc,
                                 "note": f"B = {b} primary seed" if b != "rules" else "B = rules"}
    return out


def iid_to_shift(y1, P1, y2, P2, plan, rng) -> dict[str, Any]:
    out = {}
    for m in METHODS:
        out[m] = unpaired_bootstrap_change(encode(y1), [encode(p) for p in P1[m]], encode(y2), [encode(p) for p in P2[m]],
                                           K, int(plan["bootstrap_resamples"]), rng)
    return out
