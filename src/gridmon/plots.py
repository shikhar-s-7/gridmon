"""Report figures (matplotlib, Agg). Figures read manifests for annotation only; methods never do."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import CLASSES  # noqa: E402
from .dsp.rms import rms_one_cycle_sliding  # noqa: E402
from .io import load_manifest, load_split, reports_root  # noqa: E402
from .units import FS, LSB_PU  # noqa: E402

# Reference categorical palette (first three slots validate all-pairs), text inks, surface.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
CH_COLORS = (C1, C2, C3)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 8,
        "axes.titlesize": 9, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 0.9,
    })


def _fig_dir(fast: bool) -> Path:
    d = reports_root(fast) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------------------------
# M1
# ----------------------------------------------------------------------------------------------

def m1_examples_per_class(fast: bool = False) -> Path:
    _style()
    X, man, meta = load_split("train", fast=fast)
    t = np.arange(X.shape[-1]) / FS
    fig, axes = plt.subplots(len(CLASSES), 2, figsize=(11, 13), sharex=True,
                             gridspec_kw={"width_ratios": [1.6, 1]})
    for r, lab in enumerate(CLASSES):
        i = int(man.index[man["label"] == lab][0])
        v = X[i].astype(float) * LSB_PU
        rms = rms_one_cycle_sliding(v)
        t0 = man.loc[i, "ev_t0"] if lab != "normal" else np.nan
        for c, name in enumerate(("v_ab", "v_bc", "v_ca")):
            axes[r, 0].plot(t, v[c], color=CH_COLORS[c], lw=0.6, label=name)
            axes[r, 1].plot(t, rms[c], color=CH_COLORS[c], lw=0.9, label=name)
        for ax in axes[r]:
            if np.isfinite(t0):
                ax.axvline(t0, color=INK, ls="--", lw=0.8)
        axes[r, 1].axhline(0.9, color=INK2, lw=0.6, ls=":")
        axes[r, 1].axhline(1.1, color=INK2, lw=0.6, ls=":")
        axes[r, 0].set_ylabel(f"{lab}\nv (pu peak)")
        axes[r, 1].set_ylabel("1-cycle RMS (pu)")
        info = f"record {i}" + (f", onset {t0:.3f} s" if np.isfinite(t0) else ", hard negatives only")
        axes[r, 0].set_title(f"{lab} ({info})", loc="left")
    axes[0, 1].legend(loc="lower right", ncol=3)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle("M1: one train record per class (3 LL waveforms, one-cycle sliding RMS; dashed = onset)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    out = _fig_dir(fast) / "m1_examples_per_class.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _hist(ax, series: list[tuple[str, np.ndarray]], title: str, xlabel: str, bins: int = 30) -> None:
    allv = np.concatenate([s for _, s in series if len(s)])
    allv = allv[np.isfinite(allv)]
    edges = np.histogram_bin_edges(allv, bins=bins) if len(allv) else bins
    for k, (name, s) in enumerate(series):
        s = s[np.isfinite(s)]
        ax.hist(s, bins=edges, color=CH_COLORS[k], alpha=0.75 if len(series) > 1 else 1.0,
                label=f"{name} (n={len(s)})", edgecolor=SURFACE, linewidth=0.5)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=7)


def m1_param_distributions(fast: bool = False) -> Path:
    _style()
    m = load_manifest("train", fast=fast)
    g = {lab: m[m["label"] == lab] for lab in CLASSES}
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.5))
    ax = axes.ravel()
    su = g["sag_unbal"]
    fam = su["ev_sag_type"].isin(["C", "D"])
    _hist(ax[0], [("sag_bal", g["sag_bal"]["ev_V"].to_numpy())], "sag_bal: characteristic V", "V (pu)")
    _hist(ax[1], [("types C/D", su.loc[fam, "ev_V"].to_numpy()), ("types F/G", su.loc[~fam, "ev_V"].to_numpy())],
          "sag_unbal: characteristic V", "V (pu)")
    _hist(ax[2], [("sag_bal", g["sag_bal"]["ev_dur"].to_numpy()), ("sag_unbal", su["ev_dur"].to_numpy())],
          "sags: duration", "s")
    _hist(ax[3], [("sag_bal", g["sag_bal"]["mm_min_ll_ev"].to_numpy()), ("sag_unbal", su["mm_min_ll_ev"].to_numpy())],
          "sags: realised min LL (margin <= 0.87)", "pu")
    _hist(ax[4], [("swell", g["swell"]["ev_S"].to_numpy())], "swell: S", "x U_pre")
    _hist(ax[5], [("swell", g["swell"]["mm_max_ll_ev"].to_numpy())], "swell: realised max LL (margin >= 1.13)", "pu")
    _hist(ax[6], [("phase_jump", g["phase_jump"]["ev_dphi_deg"].to_numpy())], "phase_jump: angle step", "deg")
    _hist(ax[7], [("freq_event", g["freq_event"]["ev_rocof"].to_numpy())], "freq_event: ROCOF", "Hz/s")
    _hist(ax[8], [("freq_event", g["freq_event"]["mm_df_end"].to_numpy())],
          "freq: f(end) - f_pre (margin |.| >= 0.25)", "Hz")
    _hist(ax[9], [("oscillation", 100 * g["oscillation"]["ev_depth"].to_numpy())], "oscillation: depth parameter", "%")
    _hist(ax[10], [("oscillation", g["oscillation"]["ev_f_m"].to_numpy())], "oscillation: modulation frequency", "Hz")
    _hist(ax[11], [("oscillation", 100 * g["oscillation"]["mm_osc_end"].to_numpy())],
          "osc: depth x envelope at end", "%")
    fig.suptitle("M1: event parameter distributions (train)", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    out = _fig_dir(fast) / "m1_param_distributions.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def m1_shift_preview(fast: bool = False) -> Path:
    _style()
    a = load_manifest("preview_iid", fast=fast)
    b = load_manifest("preview_shift", fast=fast)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    ax = axes.ravel()

    def both(col, scale=1.0):
        return [("iid (9009)", scale * a[col].to_numpy()), ("shift (9010)", scale * b[col].to_numpy())]

    def gains(df, kind):
        return np.concatenate([df[f"{kind}_{c}"].to_numpy() for c in ("ab", "bc", "ca")])

    _hist(ax[0], both("snr_db"), "noise SNR", "dB")
    _hist(ax[1], both("f_pre"), "pre-event frequency", "Hz")
    _hist(ax[2], both("h5_pct"), "5th harmonic", "% of fundamental")
    _hist(ax[3], both("h7_pct"), "7th harmonic", "% of fundamental")
    _hist(ax[4], both("unb", 100.0), "background unbalance |V2|/|V1|", "%")
    _hist(ax[5], [("iid (9009)", 100 * gains(a, "gain")), ("shift (9010)", 100 * gains(b, "gain"))],
          "sensor gain error (all channels)", "%")
    _hist(ax[6], [("iid (9009)", 100 * gains(a, "offset")), ("shift (9010)", 100 * gains(b, "offset"))],
          "sensor offset (all channels)", "% of nominal peak")
    tr = b[b["tr_on"] == 1]
    _hist(ax[7], [("shift transient amp", tr["tr_amp"].to_numpy())],
          f"unseen transient ({len(tr)}/{len(b)} shift records)", "initial amplitude (pu)")
    fig.suptitle("M1: iid vs shift nuisance distributions (200-record PREVIEW draws; sealed test sets not used)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    out = _fig_dir(fast) / "m1_shift_preview.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def m1_figures(fast: bool = False) -> list[Path]:
    return [m1_examples_per_class(fast), m1_param_distributions(fast), m1_shift_preview(fast)]


__all__ = ["m1_figures", "pd"]
