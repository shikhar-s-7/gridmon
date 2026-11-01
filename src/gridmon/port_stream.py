"""M7 host simulation of the C2000 port layer on a continuous stream (integration test, not an accuracy
result: 14 events). Used by scripts/run_port_stream.py and tests/test_port_sil.py.

Timing conventions: decimated (ring) sample n is FIR output k = n, whose linear-phase centre is analog
time t = n/5000 + gd, gd = (L-1)/(2 f_in); it becomes available to the background at input index
D n + L - 1, i.e. at real time n/5000 + 2 gd. Latency = availability time of the trigger sample - true
onset (plus <= 1 ms background scheduling). A capture belongs to an event if its trigger lies in
[T - 0.1 s, E + 1 s] (E = end of the event's window); further captures there are 'secondary'.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import CLASSES
from .generator.stream import StreamSpec, calibration, frequency, make_stream, sequence_phasors, synth_stream
from .io import ROOT, load_dataset_config
from .port_sim import PortSim

PORT_DIR = ROOT / "firmware" / "c2000"


def fir_table() -> list[dict[str, int]]:
    """Polyphase FIR parameters per f_in from the generated firmware/c2000/gm_port_fir.c."""
    txt = (PORT_DIR / "gm_port_fir.c").read_text()
    out = []
    for m in re.finditer(r"\{(\d+), (\d+), (\d+), (\d+), (\d+), gm_port_fir_\w+\}", txt):
        f, D, K, pad, L = (int(x) for x in m.groups())
        out.append({"f_in_hz": f, "D": D, "K": K, "pad": pad, "L": L})
    return out


def fir_taps(f_in: int) -> np.ndarray:
    """float32 taps h[i] reconstructed from the polyphase table (C[r][m] = h_ext[r + m D])."""
    p = next(x for x in fir_table() if x["f_in_hz"] == f_in)
    txt = (PORT_DIR / "gm_port_fir.c").read_text()
    body = txt.split(f"gm_port_fir_{f_in // 1000}k[{p['D'] * p['K']}] = {{")[1].split("};")[0]
    c = np.array([float.fromhex(s.strip().rstrip("f")) for s in body.split(",") if s.strip()]).reshape(p["D"], p["K"])
    h_ext = np.zeros(p["D"] * p["K"])
    for r in range(p["D"]):
        for m in range(p["K"]):
            h_ext[r + m * p["D"]] = c[r, m]
    return h_ext[p["pad"]:][::-1]


def isr_cost() -> list[dict[str, Any]]:
    """MACs per input sample (3 channels) of gm_port_push for every supported f_in."""
    out = []
    for p in fir_table():
        h = fir_taps(p["f_in_hz"]).astype(np.float64)
        w = np.exp(-2j * np.pi * np.outer(np.array([50.0, 1000.0, 2500.0]), np.arange(len(h))) / p["f_in_hz"])
        f = np.linspace(2500.0, p["f_in_hz"] / 2, 4000)
        stop = np.abs(np.exp(-2j * np.pi * np.outer(f, np.arange(len(h))) / p["f_in_hz"]) @ h).max()
        gains = np.abs(w @ h)
        out.append({**p, "fir_macs_per_input_sample": 3 * p["K"], "calibration_mults": 3, "ln_to_ll_mults": 3,
                    "total_macs_per_input_sample": 3 * p["K"] + 6, "useful_macs_avg": round(3 * p["L"] / p["D"], 2),
                    "macs_per_s": (3 * p["K"] + 6) * p["f_in_hz"], "gain_50hz_float32_taps": float(gains[0]),
                    "gain_1khz_db": float(20 * np.log10(gains[1])), "stopband_atten_db": float(-20 * np.log10(stop))})
    return out


def decimator_check(f_in: int = 50000, seconds: float = 0.5, seed: int = 1) -> dict[str, Any]:
    """Port decimator (C, float32, int16 LSB 2^-14) vs the float64 reference y[k] = sum_i h[i] x[D k + L - 1 - i]."""
    rng = np.random.default_rng(seed)
    n = int(seconds * f_in)
    t = np.arange(n) / f_in
    x = np.stack([np.cos(2 * np.pi * 50.3 * t + ph) + 0.05 * np.cos(2 * np.pi * 650 * t + ph) for ph in (0, -2.1, 2.1)])
    x = (x + 0.01 * rng.standard_normal(x.shape)).astype(np.float32)
    sim = PortSim(f_in)
    y = sim.decimate(x.T).astype(np.float64) * 2.0 ** -14
    h = fir_taps(f_in)
    p = next(q for q in fir_table() if q["f_in_hz"] == f_in)
    L, D = len(h), p["D"]
    m = (n - L) // D + 1
    idx = D * np.arange(m)[:, None] + L - 1 - np.arange(L)[None, :]
    ref = np.stack([x[c].astype(np.float64)[idx] @ h for c in range(3)], axis=1)
    k = min(m, y.shape[0])
    d = np.abs(y[:k] - ref[:k]) / 2.0 ** -14
    return {"f_in_hz": f_in, "n_outputs": int(k), "max_abs_diff_lsb": float(d.max()), "mean_abs_diff_lsb": float(d.mean())}


def run_stream(seed: int = 7007, duration_s: float = 120.0, f_in: int = 50000, ln: bool = False,
               calibrate: bool = True, bg_every_s: float = 0.001, proc_delay_s: float = 0.05,
               spec: StreamSpec | None = None) -> dict[str, Any]:
    """Generate the stream, feed it sample by sample through the port (host build) and evaluate."""
    cfg = load_dataset_config()
    t0 = time.perf_counter()
    spec = spec or make_stream(cfg, seed=seed, duration_s=duration_s, fs_in=f_in)
    g, o = calibration(spec) if calibrate else ([1.0] * 3, [0.0] * 3)
    sim = PortSim(f_in, ln, g, o, bg_every=int(round(bg_every_s * f_in)), proc_delay=int(round(proc_delay_s * f_in)))
    tel: list[dict[str, Any]] = []
    for x in synth_stream(spec, ln=ln):
        tel += sim.feed(x)
    runtime = time.perf_counter() - t0
    p = next(q for q in fir_table() if q["f_in_hz"] == f_in)
    gd = (p["L"] - 1) / 2 / f_in
    for d in tel:
        d["t_trigger_s"] = d["t_trigger"] / 5000 + gd                       # analog time of the trigger sample
        d["t_trigger_avail_s"] = (p["D"] * d["t_trigger"] + p["L"] - 1) / f_in  # real time it is available
        d["t_record_s"] = d["t_record"] / 5000 + gd
    events, used = [], set()
    for e in spec.events:
        mine = [i for i, d in enumerate(tel) if e.T - 0.1 <= d["t_trigger_s"] <= e.E + 1.0]
        used |= set(mine)
        for j, i in enumerate(mine):
            tel[i]["role"] = "primary" if j == 0 else "secondary"
        r: dict[str, Any] = {"label": e.label, "onset_s": round(e.T, 4), "window_end_s": round(e.E, 4),
                             "params": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in e.ev.items()},
                             "triggered": bool(mine), "n_secondary_captures": max(0, len(mine) - 1)}
        if mine:
            d = tel[mine[0]]
            r.update({"cause": d["cause_names"], "latency_s": round(d["t_trigger_avail_s"] - e.T, 4),
                      "onset_in_record_s": round(e.T - d["t_record_s"], 4), "onset_est_in_record_s": round(d["onset_est_s"], 4),
                      "rule_label": d["rule_label_name"], "rule_reason": d["rule_reason_name"],
                      "rule_correct": d["rule_label_name"] == e.label, "gbm_label": d["gbm_label_name"],
                      "gbm_max_proba": round(float(d["gbm_max_proba"]), 3), "gbm_agrees_with_truth": d["gbm_label_name"] == e.label,
                      "secondary": [{"t_s": round(tel[i]["t_trigger_s"], 3), "cause": tel[i]["cause_names"],
                                     "rule_label": tel[i]["rule_label_name"], "gbm_label": tel[i]["gbm_label_name"]}
                                    for i in mine[1:]]})
        events.append(r)
    false = []
    for i, d in enumerate(tel):
        if i in used:
            continue
        d["role"] = "false"
        where = "normal operation"
        for e in spec.episodes:
            if e.kind == "gap" and e.S <= d["t_trigger_s"] <= e.E + 1.0:
                where = "gap hard negative"
        false.append({"t_s": round(d["t_trigger_s"], 3), "cause": d["cause_names"], "where": where,
                      "rule_label": d["rule_label_name"], "gbm_label": d["gbm_label_name"]})
    ev_real = [r for r in events if r["label"] != "normal"]
    ev_norm = [r for r in events if r["label"] == "normal"]
    pos = [r["onset_in_record_s"] for r in ev_real if r["triggered"]]
    summary = {
        "events": len(ev_real), "events_triggered": sum(r["triggered"] for r in ev_real),
        "normal_events": len(ev_norm), "normal_events_triggered": sum(r["triggered"] for r in ev_norm),
        "rule_label_correct": sum(bool(r.get("rule_correct")) for r in ev_real),
        "gbm_agrees_with_truth": sum(bool(r.get("gbm_agrees_with_truth")) for r in ev_real),
        "false_triggers_outside_events": len(false), "secondary_captures": sum(r["n_secondary_captures"] for r in events),
        "onset_in_record_s_min": min(pos) if pos else None, "onset_in_record_s_max": max(pos) if pos else None,
        "onset_in_record_s_mean": float(np.mean(pos)) if pos else None,
        "latency_s_by_cause": {c: [r["latency_s"] for r in ev_real if r["triggered"] and r["cause"][0] == c]
                               for c in ("urms", "jump", "freq", "env")},
        "overruns": sim.overruns, "resyncs": tel[-1]["resyncs"] if tel else 0, "clipped": tel[-1]["clipped"] if tel else 0,
        "captures": len(tel), "triggers": sim.triggers,
    }
    return {"seed": spec.seed, "duration_s": spec.duration_s, "f_in_hz": f_in, "input": "LN" if ln else "LL",
            "calibrated": calibrate, "background_every_s": bg_every_s, "proc_delay_s": proc_delay_s,
            "stream": {"snr_db": spec.pre.snr_db, "u_pre": spec.pre.u_pre, "f_pre": spec.pre.f_pre,
                       "gain": list(spec.pre.gain), "offset": list(spec.pre.offset), "gap_s": spec.gap_s,
                       "n_gap_hard_negative_episodes": sum(e.kind == "gap" for e in spec.episodes)},
            "summary": summary, "per_event": events, "false_triggers": false,
            "port_struct_bytes": sim.sizeof_port, "port_struct_words16": sim.port_words, "runtime_s": round(runtime, 1),
            "_spec": spec, "_telemetry": tel}


def figure(res: dict[str, Any], path: Path) -> Path:
    """|V1| and frequency over the stream with event spans, trigger markers and labels (two panels, one axis each)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec: StreamSpec = res["_spec"]
    t = np.arange(0.0, spec.duration_s, 0.002)
    V1, _ = sequence_phasors(spec, t)
    f, _ = frequency(spec, t)
    ink, ink2, blue, grid = "#0b0b0b", "#52514e", "#2a78d6", "#e4e3df"
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        for e in spec.events:
            ax.axvspan(e.S, e.E, color=grid, lw=0, zorder=0)
        ax.grid(axis="y", color=grid, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=ink2, labelsize=8)
    a1.plot(t, np.abs(V1), color=blue, lw=1.0)
    a2.plot(t, f, color=blue, lw=1.0)
    a1.set_ylabel("|V1| (pu, model)", color=ink2, fontsize=9)
    a2.set_ylabel("f (Hz, model)", color=ink2, fontsize=9)
    a2.set_xlabel("time (s)", color=ink2, fontsize=9)
    top = max(1.32, float(np.abs(V1).max()) + 0.06)
    a1.set_ylim(max(0.0, float(np.abs(V1).min()) - 0.08), top + 0.62)
    for e, r in zip(spec.events, res["per_event"]):
        lab = r["label"]
        if r["triggered"]:
            txt = f"{lab} -> {r['rule_label']}" + ("" if r["rule_correct"] or lab == "normal" else " (x)")
        else:
            txt = f"{lab}: no trigger"
        a1.text(e.T + 0.6, top + 0.04, txt, rotation=90, fontsize=7, color=ink, va="bottom", ha="center")
    tel = res["_telemetry"]
    for d in tel:
        primary = d.get("role") == "primary"
        a1.plot(d["t_trigger_s"], top - 0.03, marker="v", ms=8 if primary else 6, color=ink if primary else "none",
                markeredgecolor=ink, mew=1.2, zorder=5)
    from matplotlib.lines import Line2D
    a1.legend(handles=[Line2D([], [], marker="v", ls="", color=ink, ms=8, label="capture (primary)"),
                       Line2D([], [], marker="v", ls="", color="none", markeredgecolor=ink, ms=6, label="secondary capture"),
                       matplotlib.patches.Patch(color=grid, label="event window")],
              loc="lower right", fontsize=7, frameon=False, ncol=3)
    s = res["summary"]
    a1.set_title(f"M7 port-layer stream [SIM]: {spec.duration_s:.0f} s at {res['f_in_hz'] // 1000} kHz, seed {spec.seed}; "
                 f"{s['events_triggered']}/{s['events']} events captured, rules {s['rule_label_correct']}/{s['events']}, "
                 f"{s['normal_events_triggered']}/{s['normal_events']} normal episodes triggered, "
                 f"{s['false_triggers_outside_events']} false triggers, {s['overruns']} overruns (labels: truth -> rule label)",
                 fontsize=9, color=ink, loc="left", pad=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def public(res: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in res.items() if not k.startswith("_")}


def class_counts(spec: StreamSpec) -> dict[str, int]:
    return {c: sum(e.label == c for e in spec.events) for c in CLASSES}
