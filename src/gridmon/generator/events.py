"""Event models (brief B6.3), normal-class hard negatives (B6.2) and label margins (B5).

An event is a flat dict: {'label': ..., <parameters>}. Angles are stored in radians here and
converted to degrees for the manifest. The fundamental is described by sequence phasors
V1(t), V2(t) (complex pu, relative to the rotating reference theta(t)) and a frequency
trajectory f(t); `fundamental` and `freq_theta` evaluate them on any time grid.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..units import ll_fundamental_mags
from .nuisance import HARMONICS, PreEvent
from .phasors import dip_sequence

Event = dict[str, Any]

SAG_TYPES = {"CD": ("C", "D"), "FG": ("F", "G")}


def _u(rng: np.random.Generator, lohi) -> float:
    lo, hi = lohi
    return float(rng.uniform(lo, hi))


def _sign(rng: np.random.Generator) -> float:
    return 1.0 if rng.random() < 0.5 else -1.0


# ----------------------------------------------------------------------------------------------
# Drawing event parameters (fixed draw order per class; all fields drawn even if unused)
# ----------------------------------------------------------------------------------------------

def draw_event(label: str, rng: np.random.Generator, ecfg: dict[str, Any],
               onset: tuple[float, float]) -> Event:
    """Draw event (or hard-negative) parameters for one record of class `label`."""
    if label == "normal":
        return _draw_normal(rng, ecfg["normal"])
    c = ecfg[label]
    ev: Event = {"label": label, "t0": _u(rng, onset)}
    if label in ("sag_bal", "sag_unbal"):
        ev["V"] = _u(rng, c["V"])
        ev["psi"] = np.radians(_u(rng, c["psi_deg"]))
        ev["dur"] = _u(rng, c["duration_s"])
        ev["rec_abrupt"] = bool(rng.random() < c["p_abrupt"])
        ev["rec_level"] = _u(rng, c["rec_level"])
        ev["rec_tau"] = _u(rng, c["rec_tau_s"])
        if label == "sag_bal":
            ev["sag_type"], ev["special"] = "A", "a"
        else:
            family = "CD" if rng.random() < 0.5 else "FG"
            combo = int(rng.integers(6))  # sign (2) x special phase (3), uniform
            ev["sag_type"] = SAG_TYPES[family][combo // 3]
            ev["special"] = "abc"[combo % 3]
    elif label == "swell":
        ev["S"] = _u(rng, c["S"])
        ev["ang"] = np.radians(_sign(rng) * _u(rng, c["angle_deg"]))
        ev["v2x"] = _u(rng, c["extra_v2"])
        ev["v2x_ang"] = _u(rng, (0.0, 2 * np.pi))
        ev["dur"] = _u(rng, c["duration_s"])
    elif label == "phase_jump":
        ev["dphi"] = np.radians(_sign(rng) * _u(rng, c["dphi_deg"]))
        ev["dv1"] = _u(rng, c["dv1"])
    elif label == "freq_event":
        mag = _u(rng, c["rocof_hz_per_s"])
        ev["rocof"] = -mag if rng.random() < c["p_negative"] else mag
        ev["df_target"] = _u(rng, c["df_hz"])
        ev["dv1"] = _u(rng, c["dv1"])
    elif label == "oscillation":
        ev["depth"] = _u(rng, c["depth"])
        ev["ang_amp"] = np.radians(_u(rng, c["ang_deg"]))
        ev["f_m"] = _u(rng, c["f_m_hz"])
        ev["sigma"] = _u(rng, c["sigma_per_s"])
        ev["phi_m"] = _u(rng, (0.0, 2 * np.pi))
        ev["phi_rel"] = _u(rng, (0.0, 2 * np.pi))
        ev["ramp"] = float(c["ramp_s"])
    else:
        raise ValueError(f"unknown label {label!r}")
    return ev


def quiet_normal() -> Event:
    """Normal event with every hard negative off (clean records for tests and golden unit signals)."""
    return {"label": "normal", "hn_step_on": False, "hn_step": 0.0, "hn_step_t": 1.0,
            "hn_ang_on": False, "hn_ang": 0.0, "hn_ang_t": 1.0,
            "hn_harm_on": False, "hn_harm_h": 5, "hn_harm_add": 0.0, "hn_harm_t": 1.0,
            "hn_ramp_on": False, "hn_ramp": 0.0, "hn_ramp_t": 1.0, "hn_ramp_dur": 0.5}


def _draw_normal(rng: np.random.Generator, c: dict[str, Any]) -> Event:
    """Independent hard negatives, each at a random time >= 0.2 s."""
    ev: Event = {"label": "normal"}
    s = c["v1_step"]
    ev["hn_step_on"] = bool(rng.random() < s["p"])
    ev["hn_step"] = _sign(rng) * _u(rng, s["mag"])
    ev["hn_step_t"] = _u(rng, c["time_s"])
    a = c["ang_step"]
    ev["hn_ang_on"] = bool(rng.random() < a["p"])
    ev["hn_ang"] = np.radians(_sign(rng) * _u(rng, a["deg"]))
    ev["hn_ang_t"] = _u(rng, c["time_s"])
    h = c["harm_step"]
    ev["hn_harm_on"] = bool(rng.random() < h["p"])
    ev["hn_harm_h"] = 5 if rng.random() < 0.5 else 7
    ev["hn_harm_add"] = _u(rng, h["add_pct"]) / 100.0
    ev["hn_harm_t"] = _u(rng, c["time_s"])
    r = c["v1_ramp"]
    ev["hn_ramp_on"] = bool(rng.random() < r["p"])
    ev["hn_ramp"] = _sign(rng) * _u(rng, r["mag"])
    ev["hn_ramp_t"] = _u(rng, c["time_s"])
    ev["hn_ramp_dur"] = _u(rng, r["dur_s"])
    return ev


# ----------------------------------------------------------------------------------------------
# Evaluating trajectories
# ----------------------------------------------------------------------------------------------

def _recovery(ev: Event, t: np.ndarray) -> np.ndarray:
    """Recovery multiplier after the dip end te = t0 + dur: 1 (abrupt) or
    g(t) = 1 - (1 - r) exp(-(t - te)/tau) (to r*U_pre, then exponential to U_pre)."""
    te = ev["t0"] + ev["dur"]
    if ev["rec_abrupt"]:
        return np.ones_like(t)
    return 1.0 - (1.0 - ev["rec_level"]) * np.exp(-np.maximum(t - te, 0.0) / ev["rec_tau"])


def osc_envelope(ev: Event, t: np.ndarray) -> np.ndarray:
    """E(t) = w(t) exp(sigma (t - t0)) for t >= t0, w = raised-cosine ramp-in over ev['ramp'] s."""
    tau = t - ev["t0"]
    w = np.where(tau < ev["ramp"], 0.5 * (1.0 - np.cos(np.pi * np.clip(tau, 0.0, None) / ev["ramp"])), 1.0)
    return np.where(tau >= 0.0, w * np.exp(ev["sigma"] * np.clip(tau, 0.0, None)), 0.0)


def fundamental(pre: PreEvent, ev: Event, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sequence phasors V1(t), V2(t) (complex pu relative to theta(t)) on time grid t (s)."""
    t = np.asarray(t, dtype=float)
    lab = ev["label"]
    v1p, v2b = pre.v1_pre, pre.v2_bg
    V1 = np.full(t.shape, v1p, dtype=complex)
    V2 = np.full(t.shape, v2b, dtype=complex)
    if lab == "normal":
        mag = np.ones_like(t)
        if ev["hn_step_on"]:
            mag = mag + ev["hn_step"] * (t >= ev["hn_step_t"])
        if ev["hn_ramp_on"]:
            mag = mag + ev["hn_ramp"] * np.clip((t - ev["hn_ramp_t"]) / ev["hn_ramp_dur"], 0.0, 1.0)
        ang = ev["hn_ang"] * (t >= ev["hn_ang_t"]) if ev["hn_ang_on"] else 0.0
        V1 = v1p * mag * np.exp(1j * ang)
        return V1, V2
    t0 = ev["t0"]
    on = t >= t0
    if lab in ("sag_bal", "sag_unbal"):
        te = t0 + ev["dur"]
        during = on & (t < te)
        after = t >= te
        r1, r2 = dip_sequence(ev["sag_type"], ev["V"], ev["psi"], ev["special"])
        g = _recovery(ev, t)
        V1 = np.where(during, v1p * r1, np.where(after, v1p * g, V1))
        if lab == "sag_bal":  # background V2 scales with V1
            vs = ev["V"] * np.exp(1j * ev["psi"])
            V2 = np.where(during, v2b * vs, np.where(after, v2b * g, V2))
        else:  # background V2 added
            V2 = np.where(during, v2b + v1p * r2, V2)
    elif lab == "swell":
        during = on & (t < t0 + ev["dur"])
        V1 = np.where(during, v1p * ev["S"] * np.exp(1j * ev["ang"]), V1)
        V2 = np.where(during, v2b + v1p * ev["v2x"] * np.exp(1j * ev["v2x_ang"]), V2)
    elif lab == "phase_jump":
        V1 = np.where(on, v1p * (1.0 + ev["dv1"]) * np.exp(1j * ev["dphi"]), V1)
    elif lab == "freq_event":
        V1 = np.where(on, v1p * (1.0 + ev["dv1"]), V1)
    elif lab == "oscillation":
        E = osc_envelope(ev, t)
        ph = 2 * np.pi * ev["f_m"] * (t - t0) + ev["phi_m"]
        V1 = v1p * (1.0 + ev["depth"] * E * np.sin(ph)) * np.exp(1j * ev["ang_amp"] * E * np.sin(ph + ev["phi_rel"]))
    else:
        raise ValueError(lab)
    return V1, V2


def freq_theta(pre: PreEvent, ev: Event, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frequency f(t) (Hz) and reference phase theta(t) = theta0 + 2*pi*integral(f dt) (rad), exact.

    f(t) = f_pre + drift*t + r*clip(t - t0, 0, Tr), Tr = df_target/|r| (freq_event only), so
    theta(t) = theta0 + 2*pi*(f_pre*t + drift*t^2/2 + r*R(t)) with R the integral of the ramp:
    R = 0 (t < t0); (t-t0)^2/2 (t0 <= t < t0+Tr); Tr^2/2 + Tr*(t - t0 - Tr) (t >= t0+Tr).
    """
    t = np.asarray(t, dtype=float)
    f = pre.f_pre + pre.drift * t
    phase = pre.f_pre * t + 0.5 * pre.drift * t * t
    if ev["label"] == "freq_event":
        r = ev["rocof"]
        Tr = ev["df_target"] / abs(r)
        tau = np.clip(t - ev["t0"], 0.0, None)
        f = f + r * np.minimum(tau, Tr)
        R = np.where(tau < Tr, 0.5 * tau * tau, 0.5 * Tr * Tr + Tr * (tau - Tr))
        phase = phase + r * R
    return f, pre.theta0 + 2 * np.pi * phase


def harmonic_levels(pre: PreEvent, ev: Event, t: np.ndarray) -> list[np.ndarray]:
    """Harmonic amplitudes A_h(t) = level_h * U_pre (pu peak), h = 5, 7, 11, 13 (+ normal step)."""
    t = np.asarray(t, dtype=float)
    out = []
    for h, lvl in zip(HARMONICS, pre.harm_level):
        a = np.full(t.shape, lvl)
        if ev["label"] == "normal" and ev["hn_harm_on"] and ev["hn_harm_h"] == h:
            a = a + ev["hn_harm_add"] * (t >= ev["hn_harm_t"])
        out.append(pre.u_pre * a)
    return out


# ----------------------------------------------------------------------------------------------
# Label margins (analytic, from fundamental phasors incl. background)
# ----------------------------------------------------------------------------------------------

def event_window(ev: Event, duration: float = 1.0) -> tuple[float, float]:
    """Interval over which the event is 'present' for margin purposes."""
    lab = ev["label"]
    if lab == "normal":
        return 0.0, duration
    if lab in ("sag_bal", "sag_unbal", "swell"):
        return ev["t0"], min(ev["t0"] + ev["dur"], duration)
    return ev["t0"], duration


def margin_metrics(pre: PreEvent, ev: Event, grid_hz: float = 2000.0, duration: float = 1.0) -> dict[str, float]:
    """Realised margin metrics from analytic phasors on a uniform grid over [0, duration].

    mm_min_ll_ev / mm_max_ll_ev: min / max LL fundamental magnitude in the event window (pu);
    mm_min_ll / mm_max_ll: same over the whole record; mm_dv1: max | |V1|/U_pre - 1 | after onset
    (whole record for normal); mm_ang_deg: max |angle(V1)| step relative to pre-event (deg);
    mm_df_end: f(end) - f_pre (Hz); mm_osc_end: depth * E(end) (oscillation, else nan).
    """
    t = np.arange(0.0, duration + 0.5 / grid_hz, 1.0 / grid_hz)
    V1, V2 = fundamental(pre, ev, t)
    ll = ll_fundamental_mags(V1, V2)
    lo, hi = event_window(ev, duration)
    w = (t >= lo) & (t < hi) if hi > lo else (t >= lo)
    if not w.any():
        w = t >= lo
    llmin = ll.min(axis=0)
    llmax = ll.max(axis=0)
    after = t >= (ev.get("t0", 0.0))
    v1rel = np.abs(V1) / pre.u_pre - 1.0
    ang = np.angle(V1 / pre.v1_pre)
    f, _ = freq_theta(pre, ev, np.array([duration]))
    m = {
        "mm_min_ll_ev": float(llmin[w].min()),
        "mm_max_ll_ev": float(llmax[w].max()),
        "mm_min_ll": float(llmin.min()),
        "mm_max_ll": float(llmax.max()),
        "mm_dv1": float(np.abs(v1rel[after]).max()),
        "mm_ang_deg": float(np.degrees(np.abs(ang[after]).max())),
        "mm_df_end": float(f[0] - pre.f_pre),
        "mm_osc_end": float(ev["depth"] * osc_envelope(ev, np.array([duration]))[0]) if ev["label"] == "oscillation" else float("nan"),
    }
    return m


def margins_ok(label: str, pre: PreEvent, ev: Event, m: dict[str, float], mc: dict[str, Any]) -> bool:
    """Label margins of brief B5 (+ D-034 oscillation band).

    oscillation: depth >= 2 % and min/max over the whole record of the analytic LL fundamental
    magnitudes (|V1 r1_k + V2 r2_k| incl. modulation, growth envelope and background unbalance; pu)
    within [0.92, 1.08] (mm_min_ll / mm_max_ll). A rejected draw redraws the oscillation event only.
    """
    eps = 1e-12
    if label in ("sag_bal", "sag_unbal"):
        return m["mm_min_ll_ev"] <= mc["sag_min_ll"]
    if label == "swell":
        return m["mm_max_ll_ev"] >= mc["swell_max_ll"]
    if label == "normal":
        lo, hi = mc["normal_ll"]
        return (m["mm_min_ll"] >= lo and m["mm_max_ll"] <= hi
                and m["mm_ang_deg"] <= mc["normal_max_ang_deg"] + eps
                and abs(pre.drift) <= mc["normal_max_drift"] + eps)
    if label == "phase_jump":
        return abs(np.degrees(ev["dphi"])) >= mc["jump_min_deg"] - eps and m["mm_dv1"] <= mc["jump_max_dv1"] + eps
    if label == "freq_event":
        return abs(m["mm_df_end"]) >= mc["freq_min_df"]
    if label == "oscillation":  # D-034: whole-record LL fundamental within [0.92, 1.08] (no dip/swell overlap)
        lo, hi = mc["osc_ll"]
        return (ev["depth"] >= mc["osc_min_depth"] - eps
                and m["mm_min_ll"] >= lo and m["mm_max_ll"] <= hi)
    raise ValueError(label)


def osc_band_excess(m: dict[str, float], mc: dict[str, Any]) -> float:
    """D-034 band violation (pu): max(0, lo - mm_min_ll) + max(0, mm_max_ll - hi); 0 inside the band."""
    lo, hi = mc["osc_ll"]
    return max(0.0, lo - m["mm_min_ll"]) + max(0.0, m["mm_max_ll"] - hi)


def event_to_record(ev: Event) -> dict[str, Any]:
    """Flat manifest columns for an event (angles in degrees, prefixed ev_/hn_)."""
    out: dict[str, Any] = {}
    deg_keys = {"psi", "ang", "v2x_ang", "dphi", "ang_amp", "phi_m", "phi_rel", "hn_ang"}
    for k, v in ev.items():
        if k == "label":
            continue
        name = k if k.startswith("hn_") else f"ev_{k}"
        if k in deg_keys:
            name += "_deg"
            v = float(np.degrees(v))
        if isinstance(v, bool):
            v = int(v)
        out[name] = v
    return out
