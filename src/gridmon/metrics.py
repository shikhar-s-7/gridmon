"""Classification metrics: confusion matrix, per-class P/R/F1, macro-F1, false alarms, misses,
safety-critical confusions."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import CLASSES

CRITICAL: list[tuple[str, str]] = [("phase_jump", "freq_event"), ("freq_event", "phase_jump"),
                                   ("sag_bal", "normal"), ("sag_unbal", "normal")]


def confusion(y_true_idx: np.ndarray, y_pred_idx: np.ndarray, k: int) -> np.ndarray:
    """C[i, j] = count(true i, predicted j)."""
    return np.bincount(y_true_idx * k + y_pred_idx, minlength=k * k).reshape(k, k)


def prf(C: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(C).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(C.sum(0) > 0, tp / C.sum(0), 0.0)
        r = np.where(C.sum(1) > 0, tp / C.sum(1), 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return p, r, f


def macro_f1(y_true_idx: np.ndarray, y_pred_idx: np.ndarray, k: int) -> float:
    return float(prf(confusion(y_true_idx, y_pred_idx, k))[2].mean())


def report(y_true: np.ndarray, y_pred: np.ndarray, classes: tuple[str, ...] = CLASSES) -> dict[str, Any]:
    """Full metric dict (JSON-serialisable) for string labels."""
    k = len(classes)
    yt = np.array([classes.index(c) for c in y_true])
    yp = np.array([classes.index(c) for c in y_pred])
    C = confusion(yt, yp, k)
    p, r, f = prf(C)
    i_norm = classes.index("normal")
    n_norm = C[i_norm].sum()
    n_event = C.sum() - n_norm
    crit = {f"{a}->{b}": int(C[classes.index(a), classes.index(b)]) for a, b in CRITICAL}
    crit["sag_*->normal"] = crit["sag_bal->normal"] + crit["sag_unbal->normal"]
    return {
        "n": int(C.sum()), "accuracy": float(np.trace(C) / C.sum()), "macro_f1": float(f.mean()),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(C[i].sum())}
                      for i, c in enumerate(classes)},
        "confusion": {"classes": list(classes), "matrix": C.tolist(), "rows": "true", "cols": "predicted"},
        "false_alarm_rate": float((n_norm - C[i_norm, i_norm]) / max(n_norm, 1)),
        "miss_rate": float(C[np.arange(k) != i_norm, i_norm].sum() / max(n_event, 1)),
        "critical_confusions": crit,
    }


def confusion_text(rep: dict[str, Any]) -> str:
    """Fixed-width text table of a confusion matrix (rows true, cols predicted)."""
    cls = rep["confusion"]["classes"]
    short = [c[:9] for c in cls]
    C = rep["confusion"]["matrix"]
    lines = ["true \\ pred  " + " ".join(f"{s:>9s}" for s in short)]
    for c, row in zip(short, C):
        lines.append(f"{c:<12s} " + " ".join(f"{v:9d}" for v in row))
    return "\n".join(lines)
