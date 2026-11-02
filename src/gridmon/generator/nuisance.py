"""Pre-event and nuisance conditions (brief B6.2, B6.6).

One code path for every class: `draw_pre_event` is called once per record, before any
class-specific branch, from the record's dedicated 'pre' random stream. Label-margin rejection
never calls it again (rule: rejection redraws only event / hard-negative parameters).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

HARMONICS: tuple[int, ...] = (5, 7, 11, 13)
CHANNELS: tuple[str, ...] = ("ab", "bc", "ca")


@dataclass(frozen=True)
class PreEvent:
    """Pre-event network state and measurement nuisances of one record.

    u_pre: |V1| pre-event (pu); unb, unb_ang: background |V2|/|V1| and angle (rad);
    harm_level[h]: amplitude as a fraction of the fundamental (h = 5, 7, 11, 13), harm_phase[h] rad;
    f_pre: frequency at t = 0 (Hz); drift: df/dt (Hz/s); gain[c]: relative gain error;
    offset[c]: offset (pu of nominal peak); snr_db: white-noise SNR vs nominal LL RMS;
    theta0: reference phase at t = 0 (rad);
    tr_*: optional damped oscillatory transient (shift only): on/off, time (s), freq (Hz),
    initial amplitude (pu peak), decay time constant (s), phase (rad).
    ih_*: optional interharmonic (S2E1 'novel' nuisance, D-080): ih_on is None when the nuisance
    configuration has no interharmonic (Stage 1 configurations: no draw, no manifest columns), else
    on/off; frequency (Hz, fixed, not tracking f), amplitude (fraction of u_pre), phase (rad). It is a
    balanced positive-sequence component over the whole record (synth.synth_phase_voltages).
    """

    u_pre: float
    unb: float
    unb_ang: float
    harm_level: tuple[float, ...]
    harm_phase: tuple[float, ...]
    f_pre: float
    drift: float
    gain: tuple[float, float, float]
    offset: tuple[float, float, float]
    snr_db: float
    theta0: float
    tr_on: bool = False
    tr_t: float = float("nan")
    tr_f: float = float("nan")
    tr_amp: float = float("nan")
    tr_tau: float = float("nan")
    tr_phase: float = float("nan")
    ih_on: bool | None = None
    ih_f: float = float("nan")
    ih_amp: float = float("nan")
    ih_phase: float = float("nan")

    @property
    def v1_pre(self) -> complex:
        """Pre-event positive-sequence phasor (angle 0: theta carries the absolute phase)."""
        return complex(self.u_pre)

    @property
    def v2_bg(self) -> complex:
        """Background negative-sequence phasor: unb * u_pre * exp(j unb_ang)."""
        return complex(self.unb * self.u_pre * np.exp(1j * self.unb_ang))

    def to_record(self) -> dict[str, Any]:
        """Flat dict for the manifest (angles in degrees, harmonics in percent)."""
        d = asdict(self)
        out: dict[str, Any] = {
            "u_pre": d["u_pre"], "unb": d["unb"], "unb_ang_deg": float(np.degrees(d["unb_ang"])),
            "f_pre": d["f_pre"], "drift": d["drift"], "snr_db": d["snr_db"],
            "theta0_deg": float(np.degrees(d["theta0"])),
        }
        for h, lvl, ph in zip(HARMONICS, self.harm_level, self.harm_phase):
            out[f"h{h}_pct"] = 100.0 * lvl
            out[f"h{h}_ph_deg"] = float(np.degrees(ph))
        for c, g, o in zip(CHANNELS, self.gain, self.offset):
            out[f"gain_{c}"] = g
            out[f"offset_{c}"] = o
        out.update({"tr_on": int(self.tr_on), "tr_t": self.tr_t, "tr_f": self.tr_f,
                    "tr_amp": self.tr_amp, "tr_tau": self.tr_tau,
                    "tr_phase_deg": float(np.degrees(self.tr_phase))})
        if self.ih_on is not None:  # S2E1 interharmonic option only (Stage 1 manifests unchanged)
            out.update({"ih_on": int(self.ih_on), "ih_f": self.ih_f, "ih_amp_pct": 100.0 * self.ih_amp,
                        "ih_phase_deg": float(np.degrees(self.ih_phase)),
                        "ih_beat_hz": abs(self.ih_f - 50.0) if self.ih_on else float("nan")})
        return out


def _u(rng: np.random.Generator, lohi) -> float:
    lo, hi = lohi
    return float(rng.uniform(lo, hi))


def interharmonic_intervals(f_lo: float, f_hi: float, f_nom: float = 50.0,
                            exclude_hz: float = 10.0) -> list[tuple[float, float]]:
    """Allowed interharmonic frequency intervals: [f_lo, f_hi] minus (k f_nom - exclude, k f_nom + exclude), k >= 1."""
    out: list[tuple[float, float]] = []
    lo = f_lo
    k = 1
    while lo < f_hi:
        a, b = k * f_nom - exclude_hz, k * f_nom + exclude_hz
        if b <= lo:
            k += 1
            continue
        if a > lo:
            out.append((lo, min(a, f_hi)))
        lo = max(lo, b)
        k += 1
    return [(a, b) for a, b in out if b > a]


def draw_interharmonic_freq(rng: np.random.Generator, icfg: dict[str, Any]) -> float:
    """Uniform frequency over the allowed intervals (one uniform draw, mapped; fixed draw count)."""
    iv = interharmonic_intervals(icfg["freq_hz"][0], icfg["freq_hz"][1], 50.0, icfg["exclude_harmonic_hz"])
    lengths = np.array([b - a for a, b in iv])
    u = float(rng.uniform(0.0, lengths.sum()))
    edges = np.concatenate([[0.0], np.cumsum(lengths)])
    i = int(min(np.searchsorted(edges, u, side="right") - 1, len(iv) - 1))
    return float(iv[i][0] + (u - edges[i]))


def draw_pre_event(rng: np.random.Generator, ncfg: dict[str, Any]) -> PreEvent:
    """Draw pre-event conditions and nuisances (same code path for all classes; fixed draw order)."""
    u_pre = _u(rng, ncfg["u_pre"])
    unb = _u(rng, ncfg["unbalance"])
    unb_ang = _u(rng, (0.0, 2 * np.pi))
    hp = ncfg["harmonics_pct"]
    harm_level = tuple(_u(rng, hp[h]) / 100.0 for h in HARMONICS)
    harm_phase = tuple(_u(rng, (0.0, 2 * np.pi)) for _ in HARMONICS)
    f_pre = _u(rng, ncfg["f_pre_hz"])
    drift = _u(rng, ncfg["drift_hz_per_s"])
    gain = tuple(_u(rng, ncfg["gain"]) for _ in CHANNELS)
    offset = tuple(_u(rng, ncfg["offset_pu"]) for _ in CHANNELS)
    snr_db = _u(rng, ncfg["snr_db"])
    theta0 = _u(rng, (0.0, 2 * np.pi))
    tr = ncfg.get("transient", {"p": 0.0})
    kw: dict[str, Any] = {}
    if tr.get("p", 0.0) > 0.0:
        on = bool(rng.random() < tr["p"])
        kw = dict(tr_on=on, tr_t=_u(rng, tr["time_s"]), tr_f=_u(rng, tr["freq_hz"]),
                  tr_amp=_u(rng, tr["amp_pu"]), tr_tau=_u(rng, tr["tau_s"]),
                  tr_phase=_u(rng, (0.0, 2 * np.pi)))
        if not on:
            kw = dict(tr_on=False)
    ih = ncfg.get("interharmonic")
    if ih is not None:  # S2E1 option, drawn after every Stage 1 draw (Stage 1 streams unchanged)
        on = bool(rng.random() < ih["p"])
        f_ih = draw_interharmonic_freq(rng, ih)
        amp = _u(rng, ih["amp_pct"]) / 100.0
        ph = _u(rng, (0.0, 2 * np.pi))
        kw.update(dict(ih_on=on, ih_f=f_ih, ih_amp=amp, ih_phase=ph) if on else dict(ih_on=False))
    return PreEvent(u_pre=u_pre, unb=unb, unb_ang=unb_ang, harm_level=harm_level,
                    harm_phase=harm_phase, f_pre=f_pre, drift=drift, gain=gain,  # type: ignore[arg-type]
                    offset=offset, snr_db=snr_db, theta0=theta0, **kw)  # type: ignore[arg-type]


def clean_pre_event(**overrides: Any) -> PreEvent:
    """Nuisance-free pre-event state (nominal, balanced, no harmonics, 50 Hz, ideal sensor, no noise).

    snr_db = inf means no noise. Used for tests, golden unit signals and examples.
    """
    base: dict[str, Any] = dict(u_pre=1.0, unb=0.0, unb_ang=0.0, harm_level=(0.0,) * 4,
                                harm_phase=(0.0,) * 4, f_pre=50.0, drift=0.0, gain=(0.0,) * 3,
                                offset=(0.0,) * 3, snr_db=float("inf"), theta0=0.0)
    base.update(overrides)
    return PreEvent(**base)
