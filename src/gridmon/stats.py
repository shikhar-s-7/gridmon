"""Statistics for the sealed evaluation (M4): stratified paired bootstrap of macro-F1 differences,
exact McNemar test, unpaired bootstrap of iid -> shift changes.

Bootstrap: resample record indices with replacement WITHIN each true class (class counts preserved),
apply the same resample to every method (paired), recompute macro-F1. 95 % CI = 2.5/97.5 percentiles.
McNemar (exact binomial): b = #(A correct, B wrong), c = #(A wrong, B correct);
p = min(1, 2 * BinomCDF(min(b, c); b + c, 0.5)); p = 1 if b + c = 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import binom

from .metrics import confusion, prf


def stratified_indices(y: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """[n_boot, N] resampled indices, sampling with replacement within each class of y."""
    N = len(y)
    out = np.empty((n_boot, N), dtype=np.int64)
    pos = 0
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        out[:, pos:pos + len(idx)] = idx[rng.integers(0, len(idx), size=(n_boot, len(idx)))]
        pos += len(idx)
    return out


def macro_f1_rows(y: np.ndarray, pred: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    """Macro-F1 for each row of resample indices idx [B, N] (y, pred: class indices)."""
    out = np.empty(idx.shape[0])
    for b in range(idx.shape[0]):
        out[b] = prf(confusion(y[idx[b]], pred[idx[b]], k))[2].mean()
    return out


def paired_bootstrap(y: np.ndarray, preds_a: list[np.ndarray], pred_b: np.ndarray, k: int, n_boot: int,
                     rng: np.random.Generator) -> dict[str, Any]:
    """CI of macro-F1(A) - macro-F1(B); A may have several seeds (difference of the seed-mean F1, plus
    per-seed CIs). Same stratified resamples for every method."""
    idx = stratified_indices(y, n_boot, rng)
    fb = macro_f1_rows(y, pred_b, idx, k)
    fa = np.stack([macro_f1_rows(y, p, idx, k) for p in preds_a])  # [S, B]
    full_b = prf(confusion(y, pred_b, k))[2].mean()
    full_a = np.array([prf(confusion(y, p, k))[2].mean() for p in preds_a])
    d_mean = fa.mean(axis=0) - fb

    def ci(d):
        lo, hi = np.percentile(d, [2.5, 97.5])
        return [float(lo), float(hi)]

    return {"n_boot": int(n_boot), "stratified": True,
            "diff_seed_mean": float(full_a.mean() - full_b), "ci95_seed_mean": ci(d_mean),
            "frac_resamples_diff_le_0": float((d_mean <= 0).mean()),
            "per_seed": [{"diff": float(full_a[s] - full_b), "ci95": ci(fa[s] - fb)} for s in range(len(preds_a))],
            "excludes_zero": bool(ci(d_mean)[0] > 0 or ci(d_mean)[1] < 0)}


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, Any]:
    """Exact two-sided McNemar test on paired per-record correctness."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    p = 1.0 if n == 0 else float(min(1.0, 2.0 * binom.cdf(min(b, c), n, 0.5)))
    return {"a_correct_b_wrong": b, "a_wrong_b_correct": c, "n_discordant": n, "p_value": p}


def unpaired_bootstrap_change(y1: np.ndarray, p1: list[np.ndarray], y2: np.ndarray, p2: list[np.ndarray], k: int,
                              n_boot: int, rng: np.random.Generator) -> dict[str, Any]:
    """CI of macro-F1(split 2) - macro-F1(split 1) (seed-mean), independent stratified resamples."""
    i1 = stratified_indices(y1, n_boot, rng)
    i2 = stratified_indices(y2, n_boot, rng)
    f1 = np.mean([macro_f1_rows(y1, p, i1, k) for p in p1], axis=0)
    f2 = np.mean([macro_f1_rows(y2, p, i2, k) for p in p2], axis=0)
    full = (np.mean([prf(confusion(y2, p, k))[2].mean() for p in p2])
            - np.mean([prf(confusion(y1, p, k))[2].mean() for p in p1]))
    lo, hi = np.percentile(f2 - f1, [2.5, 97.5])
    return {"change": float(full), "ci95": [float(lo), float(hi)], "n_boot": int(n_boot)}
