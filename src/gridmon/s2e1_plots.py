"""S2E1 report figures (matplotlib, Agg; same style as plots.py / plots_ml.py).

Arm colours = the website's series order (--series-1..4) + reference slots 5-6, fixed, never cycled (light mode,
checked with the dataviz validator: adjacent CVD dE >= 9.2, normal-vision >= 27.6; aqua and magenta are below 3:1
contrast, so every series also has its own marker and a direct label, and every number is in reports/S2E1_RESULTS.md
tables): R blue, G0 orange (= Stage 1 GBM-embedded), GW aqua, GP violet, HW magenta, H0 green. One y-axis per panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .plots import GRID, INK, INK2, _style, plt
from .plots_ml import confusion_panel

ARM_COLORS = {"R": "#2a78d6", "G0": "#eb6834", "GW": "#1baf7a", "GP": "#6250d6", "HW": "#e87ba4", "H0": "#008300"}
ARM_MARKERS = {"R": "o", "G0": "s", "GW": "D", "GP": "^", "HW": "v", "H0": "P"}
ARM_NAMES = {"R": "R rules", "G0": "G0 Stage 1 GBM", "GW": "GW wide GBM", "GP": "GP physics GBM",
             "HW": "HW hybrid (GW)", "H0": "H0 hybrid (G0)"}
ARMS = ("R", "G0", "GW", "GP", "HW", "H0")
SET_NAMES = {"test2_shift": "test2_shift (Stage 1 shift config, new seed)",
             "test2_novel": "test2_novel (wide + unseen interharmonic)"}


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _label(res: dict[str, Any], name: str) -> str:
    split = res["splits"][name]
    return SET_NAMES[name] if split == name else f"{name} stand-in: {split} (dry run)"


def confusions(res: dict[str, Any], out_dir: Path) -> Path:
    """Rows: the two sealed sets; columns: R, G0, GW, GP, HW (seed 0)."""
    _style()
    cols = ("R", "G0", "GW", "GP", "HW")
    fig, axes = plt.subplots(2, len(cols), figsize=(3.0 * len(cols), 6.4), constrained_layout=True)
    for i, name in enumerate(("test2_shift", "test2_novel")):
        for j, a in enumerate(cols):
            m = res["per_set"][name]["arms"][a]
            rep = m["primary"]
            title = f"{ARM_NAMES[a]}\nmacro-F1 {rep['macro_f1']:.3f}" + (" (seed 0)" if m["n_seeds"] > 1 else "")
            confusion_panel(axes[i, j], rep, title)
        short = name if res["splits"][name] == name else f"{res['splits'][name]} (dry run)"
        axes[i, 0].set_ylabel(f"{short}\ntrue", fontsize=8, color=INK)
    fig.suptitle("S2E1 confusion matrices [SIM] (rows true, columns predicted; colour = row share)", fontsize=10, color=INK)
    return _save(fig, out_dir, "s2e1_confusions.png")


def _end_labels(ax, ys: dict[str, float], x: float) -> None:
    """Direct labels at the right end: arms whose end values coincide (< 1 % of the axis span) share one label;
    groups are nudged apart vertically (>= 4 % of the span)."""
    lo, hi = ax.get_ylim()
    tol, gap = 0.01 * (hi - lo), 0.04 * (hi - lo)
    groups: list[tuple[float, list[str]]] = []
    for a, v in sorted(ys.items(), key=lambda kv: kv[1]):
        if groups and v - groups[-1][0] < tol:
            groups[-1][1].append(a)
        else:
            groups.append((v, [a]))
    placed: list[float] = []
    for v, names in groups:
        p = v if not placed else max(v, placed[-1] + gap)
        placed.append(p)
        ax.annotate(", ".join(names), xy=(x, v), xytext=(x + 0.1, p), fontsize=7, color=INK, va="center",
                    arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.6})


def f1_by_snr(res: dict[str, Any], out_dir: Path) -> Path:
    """Macro-F1 per SNR bin and arm (seed mean), one panel per sealed set (bins with records only)."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 2]})
    for ax, name in zip(axes, ("test2_shift", "test2_novel")):
        snr = res["per_set"][name]["breakdowns"]["snr"]
        bins = [b for b in snr["bins"] if snr["R"][b]["n"]]
        x = np.arange(len(bins))
        ends = {}
        for a in ARMS:
            v = [snr[a][b]["value"] for b in bins]
            ax.plot(x, v, color=ARM_COLORS[a], marker=ARM_MARKERS[a], ms=6, lw=2, label=ARM_NAMES[a])
            ends[a] = v[-1]
        ax.set_xticks(x, [f"{b} dB\nn={snr['R'][b]['n']}" for b in bins])
        ax.set_xlim(-0.3, len(bins) - 0.2)
        ax.set_ylabel("macro-F1 (seed mean)")
        ax.set_title(_label(res, name), fontsize=9, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        _end_labels(ax, ends, x[-1])
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=6, fontsize=7, frameon=False)
    fig.suptitle("S2E1 macro-F1 by SNR bin [SIM]", fontsize=10, color=INK)
    return _save(fig, out_dir, "s2e1_f1_by_snr.png")


def novel_breakdown(res: dict[str, Any], out_dir: Path) -> Path:
    """test2_novel: (a) macro-F1 without vs with the interharmonic per arm; (b) recall on interharmonic records by
    |f_ih - 50| (secondary bins, n shown)."""
    _style()
    bd = res["per_set"]["test2_novel"]["breakdowns"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    ax = axes[0]
    if "interharmonic" in bd:
        ih = bd["interharmonic"]
        y = np.arange(len(ARMS))[::-1]
        for yi, a in zip(y, ARMS):
            w0, w1 = ih[a]["without_interharmonic"]["value"], ih[a]["with_interharmonic"]["value"]
            ax.plot([w0, w1], [yi, yi], color=GRID, lw=2, zorder=1)
            ax.scatter([w0], [yi], s=46, facecolor="white", edgecolor=ARM_COLORS[a], lw=1.6, marker=ARM_MARKERS[a], zorder=2)
            ax.scatter([w1], [yi], s=46, color=ARM_COLORS[a], marker=ARM_MARKERS[a], edgecolor="white", lw=1.0, zorder=3)
            ax.annotate(f"{w1 - w0:+.3f}", xy=(max(w0, w1), yi), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=7, color=INK2)
        ax.set_yticks(y, [ARM_NAMES[a] for a in ARMS])
        ax.set_xlabel("macro-F1 (seed mean): open = without, filled = with interharmonic")
        n0, n1 = ih["R"]["without_interharmonic"]["n"], ih["R"]["with_interharmonic"]["n"]
        ax.set_title(f"(a) without (n={n0}) vs with (n={n1}) the interharmonic", fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6)
        lo = min(min(ih[a][k]["value"] for k in ("without_interharmonic", "with_interharmonic")) for a in ARMS)
        ax.set_xlim(lo - 0.02, 1.02)
    ax = axes[1]
    if "interharmonic_beat_secondary" in bd:
        sec = bd["interharmonic_beat_secondary"]
        bins = [b for b in sec["bins"] if sec["R"][b]["n"]]
        x = np.arange(len(bins))
        ends = {}
        for a in ARMS:
            v = [sec[a][b]["value"] for b in bins]
            ax.plot(x, v, color=ARM_COLORS[a], marker=ARM_MARKERS[a], ms=6, lw=2, label=ARM_NAMES[a])
            ends[a] = v[-1]
        ax.set_xticks(x, [f"{b} Hz\nn={sec['R'][b]['n']}" for b in bins])
        ax.set_xlim(-0.3, len(bins) - 0.2)
        ax.set_ylabel("accuracy on interharmonic records (seed mean)")
        ax.set_title("(b) by beat frequency |f_ih - 50 Hz|", fontsize=9, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        _end_labels(ax, ends, x[-1])
        h, lab = ax.get_legend_handles_labels()
        fig.legend(h, lab, loc="outside lower center", ncol=6, fontsize=7, frameon=False)
    fig.suptitle("S2E1 test2_novel: the unseen interharmonic [SIM]", fontsize=10, color=INK)
    return _save(fig, out_dir, "s2e1_novel_breakdown.png")
