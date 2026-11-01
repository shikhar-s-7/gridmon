"""Continuous-stream option of the generator (M7 port-layer integration test). NEW module: the record
generator (synth.py, dataset.py) and every existing split are untouched (bit-identical, tested).

A stream is a long three-phase signal at the control-ISR rate f_in (default 50 kHz), e.g. 120 s with
14 events (2 per class, parameters from the iid event distributions with the B5 label margins), separated
by >= 4 s of normal operation that contains hard negatives. Model (phasor domain, absolute time t):

  frequency  f(t) = f_pre + A_w sin(2 pi t / T_w + phi_w) + sum_i df_i(t)     (slow wander, |slope| < 0.005 Hz/s)
             freq events: ramp at ROCOF r to df_target, hold 1 s, return to 0 at 0.2 Hz/s (secondary
             control [ASSUMED]); theta(t) = theta0 + 2 pi integral f (exact, piecewise)
  V1(t)      = u_pre W(t) B_i rel_i(t - S_i) inside episode i's window [S_i, E_i), u_pre W(t) B_i between;
             W = 1 + 0.003 sin(2 pi t / 25 s + phi) (magnitude wander), rel_i = the record model's V1/V1_pre of
             the episode's event (gridmon.generator.events.fundamental with a clean pre-event state), B_i =
             product of the persistent factors of earlier episodes (phase jumps, |V1| steps); oscillations
             are ramped out (raised cosine, 0.1 s) at the end of their 1 s virtual record (D-034 band there)
  V2(t)      = V2_bg s_i(t - S_i) + u_pre W B_i dV2_i(t - S_i)   (background unbalance; dip/swell V2)
  harmonics  h = 5, 7, 11, 13 at the pre-event levels (+ persistent hard-negative harmonic steps)
  sensor     per-channel gain/offset error (iid ranges), white noise sigma = snr_to_sigma(snr) at f_in,
             12-bit quantisation over +/- 2 pu at f_in (the control-loop ADC); output = counts * LSB (pu).
Episodes: the 14 class events and one hard-negative episode in the middle of every gap (normal-class
draw with at least one hard negative). Every episode is a "virtual record" [S_i, S_i + 1 s) for its label
margins (B5/D-034), evaluated with the stream state at S_i; a failing draw redraws the event only.
Seeding: SeedSequence(seed).spawn(6) = (pre, order, events, gaps, noise, wander); no global RNG.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from .. import CLASSES
from ..units import A_OP, COUNT_MAX, COUNT_MIN, LSB_PU, SQRT3, snr_to_sigma
from .events import Event, draw_event, fundamental, margin_metrics, margins_ok, osc_envelope
from .nuisance import HARMONICS, PreEvent, clean_pre_event, draw_pre_event

FREQ_HOLD_S = 1.0
FREQ_RECOVERY_HZ_S = 0.2
OSC_RAMP_OUT_S = 0.1
WANDER_F_HZ, WANDER_F_PERIOD_S = 0.02, 40.0
WANDER_V, WANDER_V_PERIOD_S = 0.003, 25.0


@dataclass
class Episode:
    """One event (or gap hard-negative episode) on the stream timeline (absolute seconds)."""

    label: str
    kind: str                       # "event" (one of the 14) or "gap" (hard negatives between events)
    ev: Event
    S: float                        # virtual record start
    T: float                        # onset (event t0; first hard negative for gap/normal episodes)
    E: float                        # end of the episode window (all transients settled)
    B: complex = 1 + 0j             # baseline V1 factor before the episode
    n_reject: int = 0
    freq_bp: tuple[np.ndarray, np.ndarray] | None = None  # (t_k, df_k) breakpoints of a frequency event


@dataclass
class StreamSpec:
    seed: int
    fs_in: int
    duration_s: float
    pre: PreEvent
    episodes: list[Episode] = field(default_factory=list)
    wander: dict[str, float] = field(default_factory=dict)
    gap_s: float = 0.0
    lead_s: float = 0.0

    @property
    def events(self) -> list[Episode]:
        return [e for e in self.episodes if e.kind == "event"]


# ------------------------------------------------------------------------------------------- helpers
def _rel(ev: Event, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(V1 rel, background-V2 scale s, added V2 rel) of an event on local time tau (record model)."""
    a = clean_pre_event()
    b = clean_pre_event(unb=1.0)
    if ev["label"] == "oscillation":
        E = osc_envelope(ev, tau) * _ramp_out(tau)
        ph = 2 * np.pi * ev["f_m"] * (tau - ev["t0"]) + ev["phi_m"]
        v1 = (1.0 + ev["depth"] * E * np.sin(ph)) * np.exp(1j * ev["ang_amp"] * E * np.sin(ph + ev["phi_rel"]))
        return v1, np.ones_like(tau, dtype=complex), np.zeros_like(tau, dtype=complex)
    v1, v2_add = fundamental(a, ev, tau)
    _, v2_b = fundamental(b, ev, tau)
    return v1, v2_b - v2_add, v2_add


def _ramp_out(tau: np.ndarray, start: float = 1.0) -> np.ndarray:
    x = np.clip((tau - start) / OSC_RAMP_OUT_S, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def _active_len(ev: Event) -> float:
    """Episode window length from the virtual record start (s)."""
    lab = ev["label"]
    if lab in ("sag_bal", "sag_unbal"):
        return ev["t0"] + ev["dur"] + (0.0 if ev["rec_abrupt"] else 7.0 * ev["rec_tau"]) + 0.05
    if lab == "swell":
        return ev["t0"] + ev["dur"] + 0.05
    if lab == "phase_jump":
        return ev["t0"] + 0.05
    if lab == "freq_event":
        return ev["t0"] + abs(ev["df_target"] / ev["rocof"]) + FREQ_HOLD_S + ev["df_target"] / FREQ_RECOVERY_HZ_S + 0.05
    if lab == "oscillation":
        return 1.0 + OSC_RAMP_OUT_S + 0.05
    end = 1.0
    if ev.get("hn_ramp_on"):
        end = max(end, ev["hn_ramp_t"] + ev["hn_ramp_dur"] + 0.05)
    return end


def _first_hn_time(ev: Event) -> float:
    ts = [ev[f"hn_{k}_t"] for k in ("step", "ang", "harm", "ramp") if ev.get(f"hn_{k}_on")]
    return min(ts) if ts else 0.3


def _pl_value(tk: np.ndarray, fk: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.interp(t, tk, fk, left=0.0, right=0.0)


def _pl_integral(tk: np.ndarray, fk: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Integral from -inf of the piecewise-linear function (zero outside [tk[0], tk[-1]])."""
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (fk[1:] + fk[:-1]) * np.diff(tk))])
    tc = np.clip(t, tk[0], tk[-1])
    j = np.clip(np.searchsorted(tk, tc, side="right") - 1, 0, len(tk) - 2)
    dt = tc - tk[j]
    slope = (fk[j + 1] - fk[j]) / np.maximum(tk[j + 1] - tk[j], 1e-300)
    return cum[j] + fk[j] * dt + 0.5 * slope * dt * dt


# ------------------------------------------------------------------------------------------- spec
def make_stream(cfg: dict[str, Any], seed: int = 7007, duration_s: float = 120.0, fs_in: int = 50000,
                per_class: int = 2, min_gap_s: float = 4.0, lead_s: float = 3.0) -> StreamSpec:
    """Draw the stream: pre-event/sensor state, event order and parameters (with label margins), layout."""
    rng_pre, rng_order, rng_ev, rng_gap, _, rng_w = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6))
    pre = dataclasses.replace(draw_pre_event(rng_pre, cfg["nuisance"]["iid"]), drift=0.0)
    wander = {"phi_f": float(rng_w.uniform(0, 2 * np.pi)), "phi_v": float(rng_w.uniform(0, 2 * np.pi))}
    spec = StreamSpec(seed, fs_in, duration_s, pre, wander=wander, lead_s=lead_s)
    labels = [c for c in CLASSES for _ in range(per_class)]
    labels = [labels[i] for i in rng_order.permutation(len(labels))]
    onset = tuple(cfg["record"]["onset_s"])
    mc = cfg["margins"]
    # sequential draws: every margin check uses the stream state (baseline factor B) at that episode
    B = 1 + 0j
    drawn: list[tuple[str, Event, int, complex, Event, int, complex]] = []
    for lab in labels:
        ev, nrej = _draw_with_margins(lab, rng_ev, cfg, onset, mc, pre, B)
        B_ev = B
        B = B * _rel(ev, np.array([_active_len(ev)]))[0][0]
        hn, nrej_g = _draw_gap_normal(rng_gap, cfg, mc, pre, B)
        B_gap = B
        B = B * _rel(hn, np.array([_active_len(hn)]))[0][0]
        drawn.append((lab, ev, nrej, B_ev, hn, nrej_g, B_gap))
    act = [_active_len(d[1]) for d in drawn]
    gap = (duration_s - lead_s - sum(act)) / len(drawn)
    if gap < min_gap_s:
        raise ValueError(f"stream too short: gap {gap:.2f} s < {min_gap_s} s")
    spec.gap_s = gap
    t = lead_s
    for (lab, ev, nrej, B_ev, hn, nrej_g, B_gap), a in zip(drawn, act):
        ep = Episode(lab, "event", ev, S=t, T=t + (ev["t0"] if lab != "normal" else _first_hn_time(ev)), E=t + a,
                     B=B_ev, n_reject=nrej)
        if lab == "freq_event":
            ep.freq_bp = _freq_breakpoints(ev, t)
        spec.episodes.append(ep)
        s_gap = ep.E + 0.5 * gap - 0.6  # hard negatives in the middle of the gap (>= 1.4 s from both events)
        spec.episodes.append(Episode("normal", "gap", hn, S=s_gap, T=s_gap + _first_hn_time(hn),
                                     E=s_gap + _active_len(hn), B=B_gap, n_reject=nrej_g))
        t = ep.E + gap
    return spec


def _margin_pre(pre: PreEvent, B: complex) -> PreEvent:
    """Pre-event state of a virtual record at baseline factor B (|V2_bg| unchanged; sensor errors excluded)."""
    return dataclasses.replace(pre, u_pre=pre.u_pre * abs(B), unb=pre.unb / abs(B),
                               unb_ang=pre.unb_ang - float(np.angle(B)), gain=(0.0,) * 3, offset=(0.0,) * 3)


def _draw_with_margins(label: str, rng: np.random.Generator, cfg: dict[str, Any], onset: tuple[float, float],
                       mc: dict[str, Any], pre: PreEvent, B: complex) -> tuple[Event, int]:
    p = _margin_pre(pre, B)
    for attempt in range(int(cfg["max_redraws"]) + 1):
        ev = draw_event(label, rng, cfg["events"], onset)
        if margins_ok(label, p, ev, margin_metrics(p, ev, grid_hz=mc["grid_hz"]), mc):
            return ev, attempt
    raise RuntimeError(f"{label}: label margins not met")


def _draw_gap_normal(rng: np.random.Generator, cfg: dict[str, Any], mc: dict[str, Any], pre: PreEvent,
                     B: complex) -> tuple[Event, int]:
    """Normal-class hard negatives (at least one on); |V1| steps/ramps point back towards 1 pu so the baseline
    does not random-walk over the 120 s (gap episodes are not class events)."""
    p = _margin_pre(pre, B)
    sign = -1.0 if pre.u_pre * abs(B) > 1.0 else 1.0
    for attempt in range(int(cfg["max_redraws"]) + 1):
        ev = draw_event("normal", rng, cfg["events"], (0.2, 0.4))
        ev["hn_step"] = sign * abs(ev["hn_step"])
        ev["hn_ramp"] = sign * abs(ev["hn_ramp"])
        if any(ev[f"hn_{k}_on"] for k in ("step", "ang", "harm", "ramp")) and \
                margins_ok("normal", p, ev, margin_metrics(p, ev, grid_hz=mc["grid_hz"]), mc):
            return ev, attempt
    raise RuntimeError("gap hard negative: margins not met")


def _freq_breakpoints(ev: Event, S: float) -> tuple[np.ndarray, np.ndarray]:
    T0 = S + ev["t0"]
    df = np.sign(ev["rocof"]) * ev["df_target"]
    Tr = ev["df_target"] / abs(ev["rocof"])
    t1, t2 = T0 + Tr, T0 + Tr + FREQ_HOLD_S
    t3 = t2 + ev["df_target"] / FREQ_RECOVERY_HZ_S
    return np.array([T0, t1, t2, t3]), np.array([0.0, df, df, 0.0])


# ------------------------------------------------------------------------------------------- evaluation
def frequency(spec: StreamSpec, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """f(t) (Hz) and theta(t) (rad)."""
    pre, w = spec.pre, spec.wander
    om = 2 * np.pi / WANDER_F_PERIOD_S
    f = pre.f_pre + WANDER_F_HZ * np.sin(om * t + w["phi_f"])
    integ = pre.f_pre * t + WANDER_F_HZ / om * (np.cos(w["phi_f"]) - np.cos(om * t + w["phi_f"]))
    for ep in spec.episodes:
        if ep.freq_bp is not None:
            f = f + _pl_value(*ep.freq_bp, t)
            integ = integ + _pl_integral(*ep.freq_bp, t)
    return f, pre.theta0 + 2 * np.pi * integ


def sequence_phasors(spec: StreamSpec, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """V1(t), V2(t) (complex pu, relative to theta(t))."""
    pre = spec.pre
    W = 1.0 + WANDER_V * np.sin(2 * np.pi * t / WANDER_V_PERIOD_S + spec.wander["phi_v"])
    eps = spec.episodes
    starts = np.array([e.S for e in eps])
    # baseline factor: B of the next episode applies from the previous episode's end (constant in between)
    Bs = np.array([e.B for e in eps] + [eps[-1].B * _rel(eps[-1].ev, np.array([eps[-1].E - eps[-1].S]))[0][0]])
    j = np.searchsorted(starts, t, side="right")  # episodes started so far
    B = Bs[j]                                     # B after the last started episode (fixed below inside windows)
    V1 = pre.u_pre * W * B
    V2 = np.full(t.shape, pre.v2_bg, dtype=complex)
    for i, ep in enumerate(eps):
        m = (t >= ep.S) & (t < ep.E)
        if not m.any():
            continue
        v1r, s, v2a = _rel(ep.ev, t[m] - ep.S)
        V1[m] = pre.u_pre * W[m] * ep.B * v1r
        V2[m] = pre.v2_bg * s + pre.u_pre * W[m] * ep.B * v2a
    return V1, V2


def harmonic_levels_stream(spec: StreamSpec, t: np.ndarray) -> list[np.ndarray]:
    out = []
    for h, lvl in zip(HARMONICS, spec.pre.harm_level):
        a = np.full(t.shape, lvl)
        for ep in spec.episodes:
            ev = ep.ev
            if ev["label"] == "normal" and ev["hn_harm_on"] and ev["hn_harm_h"] == h:
                a = a + ev["hn_harm_add"] * (t >= ep.S + ev["hn_harm_t"])
        out.append(spec.pre.u_pre * a)
    return out


def phase_voltages(spec: StreamSpec, t: np.ndarray) -> np.ndarray:
    """Zero-sequence-free phase (LN) voltages [3, n], pu of nominal LN peak (as synth.synth_phase_voltages)."""
    V1, V2 = sequence_phasors(spec, t)
    _, theta = frequency(spec, t)
    E = np.exp(1j * theta)
    P, Q = V1 * E, V2 * E
    a = A_OP
    r1, r2 = (1.0, a * a, a), (1.0, a, a * a)
    v = np.stack([np.real(P * r1[k] + Q * r2[k]) for k in range(3)])
    for h, A_h, ph in zip(HARMONICS, harmonic_levels_stream(spec, t), spec.pre.harm_phase):
        Eh = A_h * np.exp(1j * (h * theta + ph))
        for k in range(3):
            v[k] += np.real(Eh * np.exp(-1j * h * 2 * np.pi * k / 3))
    return v


def synth_stream(spec: StreamSpec, chunk_s: float = 1.0, ln: bool = False) -> Iterator[np.ndarray]:
    """Yield ADC samples (pu, float32 [n, 3] interleaved) chunk by chunk: LL (ln=False) or LN channels, with
    per-channel gain/offset error, white noise at f_in and 12-bit quantisation over +/- 2 pu."""
    noise = np.random.default_rng(np.random.SeedSequence(spec.seed).spawn(6)[4])
    sigma = snr_to_sigma(spec.pre.snr_db)
    g = np.asarray(spec.pre.gain)[:, None]
    o = np.asarray(spec.pre.offset)[:, None]
    n_total = int(round(spec.duration_s * spec.fs_in))
    step = int(round(chunk_s * spec.fs_in))
    for k0 in range(0, n_total, step):
        t = np.arange(k0, min(k0 + step, n_total)) / spec.fs_in
        v = phase_voltages(spec, t)
        if not ln:
            v = np.stack([(v[0] - v[1]) / SQRT3, (v[1] - v[2]) / SQRT3, (v[2] - v[0]) / SQRT3])
        v = (1.0 + g) * v + o + sigma * noise.standard_normal(v.shape)
        c = np.clip(np.floor(v / LSB_PU + 0.5), COUNT_MIN, COUNT_MAX)
        yield np.ascontiguousarray((c * LSB_PU).T.astype(np.float32))


def calibration(spec: StreamSpec) -> tuple[list[float], list[float]]:
    """Port trim (D-052) that undoes the stream's sensor error: v = gain (x - offset)."""
    return [1.0 / (1.0 + g) for g in spec.pre.gain], list(spec.pre.offset)
