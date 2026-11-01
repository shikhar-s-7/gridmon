"""M3/M4 report figures (matplotlib, Agg). Same style and palette as plots.py.

Method colours (fixed order, never cycled): rules = C1 (blue), GBM-embedded = C2 (orange),
GBM-unconstrained = C3 (aqua). Figures read manifests for annotation only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .plots import C1, C2, C3, GRID, INK, INK2, _fig_dir, _style, plt

METHOD_COLORS = {"rules": C1, "gbm_embedded": C2, "gbm_unconstrained": C3}
METHOD_NAMES = {"rules": "rules", "gbm_embedded": "GBM-embedded", "gbm_unconstrained": "GBM-unconstrained"}


def _save(fig, name: str, fast: bool) -> Path:
    out = _fig_dir(fast) / name
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def confusion_panel(ax, rep: dict[str, Any], title: str) -> None:
    """Counts with row-normalised colour (recall), rows true / cols predicted."""
    C = np.asarray(rep["confusion"]["matrix"], dtype=float)
    cls = rep["confusion"]["classes"]
    Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
    ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] > 0:
                ax.text(j, i, f"{int(C[i, j])}", ha="center", va="center", fontsize=7,
                        color="white" if Cn[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(cls)), cls, rotation=35, ha="right")
    ax.set_yticks(range(len(cls)), cls)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, loc="left", fontsize=8)


def _rep_title(name: str, rep: dict[str, Any]) -> str:
    return (f"{name}: macro-F1 {rep['macro_f1']:.3f}, false alarms {100 * rep['false_alarm_rate']:.1f} %, "
            f"misses {100 * rep['miss_rate']:.1f} %")


# ----------------------------------------------------------------------------------------------
# M3
# ----------------------------------------------------------------------------------------------

def m3_size_vs_f1(grid: list[dict], unc: list[dict], selected: dict, rules_f1: float, budget: int,
                  fast: bool = False) -> Path:
    """Val macro-F1 (mean +/- sd over seeds) vs exported size; 64 KB budget line; rules line."""
    _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    leaf_col = {4: C1, 8: C2, 16: C3}
    markers = {(0.1, 0.8): "o", (0.1, 1.0): "s", (0.2, 0.8): "^", (0.2, 1.0): "D"}
    for g in grid:
        kb = g["bytes_max"] / 1024
        mk = markers[(g["learning_rate"], g["subsample"])]
        ax.errorbar(kb, g["val_f1_mean"], yerr=g["val_f1_sd"], fmt=mk, ms=5, color=leaf_col[g["max_leaf_nodes"]],
                    mec="white", mew=0.6, elinewidth=0.8, capsize=0)
    for u in unc:
        ax.errorbar(u["bytes_max"] / 1024, u["val_f1_mean"], yerr=u["val_f1_sd"], fmt="X", ms=7, color=INK2,
                    mec="white", mew=0.6, elinewidth=0.8)
    sel = selected
    ax.plot(sel["bytes_max"] / 1024, sel["val_f1_mean"], "o", ms=13, mfc="none", mec=INK, mew=1.2)
    c = sel["config"]
    ax.annotate(f"selected: {c['n_estimators']} stages x {c['max_leaf_nodes']} leaves, lr {c['learning_rate']}, "
                f"subsample {c['subsample']}\n{sel['bytes_max'] / 1024:.1f} KB, val macro-F1 {sel['val_f1_mean']:.4f}",
                (sel["bytes_max"] / 1024, sel["val_f1_mean"]), xytext=(12, -30), textcoords="offset points",
                fontsize=7, color=INK, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6})
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    from matplotlib.transforms import blended_transform_factory
    ax.set_xscale("log")
    ticks = [t for t in (5, 10, 20, 50, 100, 200, 500, 1000, 2000) if t >= 0.8 * ax.get_xlim()[0] and t <= 1.2 * ax.get_xlim()[1]]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.axvline(budget / 1024, color=INK2, ls="--", lw=0.9)
    tr_x = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(budget / 1024, 0.01, " 64 KB budget", color=INK2, fontsize=7, va="bottom", transform=tr_x)
    ax.axhline(rules_f1, color=C1, ls=":", lw=1.2)
    tr_y = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.01, rules_f1, f"rules (train-tuned) {rules_f1:.4f}", color=INK2, fontsize=7, va="bottom", transform=tr_y)
    ax.set_xlabel("exported model size (KB, log scale; 14 B/node + 2 B/tree + 4 B/class)")
    ax.set_ylabel("val macro-F1 (mean +/- sd over 3 seeds)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker="o", color=leaf_col[k], label=f"{k} leaves") for k in (4, 8, 16)]
    handles += [Line2D([], [], ls="", marker=m, color=INK2, mfc="none", label=f"lr {lr}, subsample {ss}")
                for (lr, ss), m in markers.items()]
    handles += [Line2D([], [], ls="", marker="X", color=INK2, label="GBM-unconstrained (HGB, reference)")]
    ax.legend(handles=handles, loc="lower right", fontsize=7, ncol=2)
    ax.set_title("M3: GBM-embedded grid on VAL (sizes 20-100 stages per marker family)", loc="left")
    fig.tight_layout()
    return _save(fig, "m3_size_vs_f1.png", fast)


def m3_feature_importance(names: list[str], mean: np.ndarray, sd: np.ndarray, title: str, fast: bool = False,
                          top: int = 15) -> Path:
    _style()
    order = np.argsort(mean)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    y = np.arange(len(order))[::-1]
    ax.barh(y, mean[order], xerr=sd[order], color=C2, height=0.6, error_kw={"elinewidth": 0.8, "ecolor": INK2})
    ax.set_yticks(y, [names[i] for i in order])
    ax.set_xlabel("drop in val macro-F1 when the feature is permuted (mean +/- sd, 10 repeats)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, "m3_feature_importance.png", fast)


def m3_val_confusion(rep_rules: dict, rep_gbm: dict, fast: bool = False) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    confusion_panel(axes[0], rep_rules, _rep_title("rules (train-tuned)", rep_rules))
    confusion_panel(axes[1], rep_gbm, _rep_title("GBM-embedded (exported model, float32 walker)", rep_gbm))
    fig.suptitle("M3: VAL confusion, rules vs GBM-embedded (counts; colour = row-normalised recall)", x=0.01,
                 ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "m3_val_confusion.png", fast)


# ----------------------------------------------------------------------------------------------
# M4
# ----------------------------------------------------------------------------------------------

def m4_confusions(reps: dict[str, dict[str, dict]], fast: bool = False) -> Path:
    """2 x 3: rows test_iid / test_shift, cols rules / GBM-embedded / GBM-unconstrained."""
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    for i, split in enumerate(("test_iid", "test_shift")):
        for j, m in enumerate(("rules", "gbm_embedded", "gbm_unconstrained")):
            confusion_panel(axes[i, j], reps[split][m], f"{split}: " + _rep_title(METHOD_NAMES[m], reps[split][m]))
    fig.suptitle("M4: sealed test sets (ML: primary seed 0; counts; colour = row-normalised recall)", x=0.01,
                 ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "m4_confusions.png", fast)


def m4_f1_by_snr(bins: list[str], data: dict[str, dict[str, list]], fast: bool = False) -> Path:
    """Macro-F1 per SNR bin and method (test_iid and test_shift pooled by bin), with record counts."""
    _style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.26
    x = np.arange(len(bins))
    for k, m in enumerate(("rules", "gbm_embedded", "gbm_unconstrained")):
        d = data[m]
        ax.bar(x + (k - 1) * w, d["f1"], width=w - 0.02, color=METHOD_COLORS[m], label=METHOD_NAMES[m])
    for xi, n in zip(x, data["rules"]["n"]):
        ax.text(xi, 0.0, f"n={n}", ha="center", va="bottom", fontsize=7, color="white")
    lo = min(min(v["f1"]) for v in data.values())
    ax.set_ylim(max(0.0, lo - 0.05), 1.0)
    ax.set_xticks(x, bins)
    ax.set_xlabel("SNR bin (dB; < 35 dB only occurs in test_shift)")
    ax.set_ylabel("macro-F1")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("M4: macro-F1 by SNR (test_iid + test_shift records pooled per bin)", loc="left")
    fig.tight_layout()
    return _save(fig, "m4_f1_by_snr.png", fast)


def m4_shift_degradation(vals: dict[str, dict[str, Any]], fast: bool = False) -> Path:
    """Macro-F1 on test_iid vs test_shift per method (ML: mean +/- sd over seeds)."""
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m in ("rules", "gbm_embedded", "gbm_unconstrained"):
        v = vals[m]
        ax.errorbar([0, 1], [v["iid"], v["shift"]], yerr=[v.get("iid_sd", 0.0), v.get("shift_sd", 0.0)],
                    color=METHOD_COLORS[m], marker="o", ms=7, lw=2, capsize=3, label=METHOD_NAMES[m])
        ax.text(1.04, v["shift"], f"{v['shift']:.3f} ({v['shift'] - v['iid']:+.3f})", fontsize=7, color=INK2,
                va="center")
    ax.set_xticks([0, 1], ["test_iid", "test_shift"])
    ax.set_xlim(-0.2, 1.45)
    ax.set_ylabel("macro-F1")
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title("M4: iid -> shift degradation (labels: shift F1 and change)", loc="left")
    fig.tight_layout()
    return _save(fig, "m4_shift_degradation.png", fast)


def m4_disagreements(examples: list[dict[str, Any]], fast: bool = False) -> Path:
    """Up to 6 test_shift records where rules and GBM-embedded disagree: waveform, Urms(1/2), |V1|, J."""
    _style()
    n = len(examples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 2.3 * n + 0.6), squeeze=False)
    from .units import FS
    for r, ex in enumerate(examples):
        t = np.arange(ex["v"].shape[-1]) / FS
        a = axes[r]
        for c, col in enumerate((C1, C2, C3)):
            a[0].plot(t, ex["v"][c], color=col, lw=0.5)
        a[0].set_ylabel("v LL (pu)")
        tu = (np.arange(ex["urms"].shape[-1]) * 50 + 50) / FS
        for c, col in enumerate((C1, C2, C3)):
            a[1].plot(tu, ex["urms"][c], color=col, lw=0.9)
        a[1].axhline(0.90, color=INK2, ls="--", lw=0.6)
        a[1].axhline(1.10, color=INK2, ls="--", lw=0.6)
        a[1].set_ylabel("Urms(1/2) (pu)")
        tm = (5 * np.arange(ex["v1"].shape[-1]) + 4) / FS
        a[2].plot(tm, np.abs(ex["v1"]), color=C1, lw=0.9)
        a[2].set_ylabel("|V1| (pu)")
        a[3].plot(tm, np.degrees(ex["J"]), color=C1, lw=0.9)
        a[3].set_ylabel("J (deg)")
        for ax in a:
            if np.isfinite(ex.get("t0", np.nan)):
                ax.axvline(ex["t0"], color=INK2, ls=":", lw=0.8)
            ax.set_xlim(0, 1)
        a[0].set_title(ex["title"], loc="left", fontsize=8)
        a[1].set_title(f"true {ex['true']} | rules {ex['rules']} | GBM-emb {ex['gbm']} (p={ex['p_gbm']:.2f})",
                       loc="left", fontsize=8)
        if r == n - 1:
            for ax in a:
                ax.set_xlabel("t (s)")
    fig.suptitle("M4: test_shift records where rules and GBM-embedded disagree (dotted line: true onset; "
                 "channels ab/bc/ca)", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "m4_disagreements.png", fast)


__all__ = ["m3_size_vs_f1", "m3_feature_importance", "m3_val_confusion", "m4_confusions", "m4_f1_by_snr",
           "m4_shift_degradation", "m4_disagreements", "GRID"]
