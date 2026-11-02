"""Waveform synthesis at 20 kHz and the measurement chain (brief B6.1, B6.4).

Chain: synthesise LL voltages at 20 kHz -> per-channel gain/offset -> causal linear-phase FIR
anti-alias filter -> decimate x4 to 5 kHz -> add white noise -> 12-bit quantisation (+/- 2 pu)
-> int16 counts.

Record timeline (DECISIONS D-012): the FIR group delay gd (samples at 20 kHz) is compensated by
synthesising the analog signal on t_k = (k - gd)/fs_synth, so stored sample n corresponds to
analog time n/fs exactly and the manifest onset t0 is on the record timeline. A real-time
firmware decimator using the same coefficients adds gd/fs_synth of delay.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal as sps
from numpy.lib.stride_tricks import sliding_window_view

from ..io import ROOT
from ..units import A_OP, COUNT_MAX, COUNT_MIN, DECIM, F_NOM, FS, FS_SYNTH, LSB_PU, N_RECORD, SQRT3, snr_to_sigma
from .events import Event, freq_theta, fundamental, harmonic_levels
from .nuisance import HARMONICS, PreEvent

AA_FIR_PATH = ROOT / "configs" / "aa_fir.json"


# ----------------------------------------------------------------------------------------------
# Anti-alias FIR
# ----------------------------------------------------------------------------------------------

def design_aa_fir(fcfg: dict[str, Any], fs_in: float = FS_SYNTH) -> dict[str, Any]:
    """Kaiser-window linear-phase lowpass, normalised to gain exactly 1 at 50 Hz.

    h = firwin(numtaps, cutoff, kaiser(beta)); h /= |H(e^{j 2 pi 50 / fs_in})|. Odd numtaps
    (type I) -> integer group delay gd = (numtaps - 1)/2 samples. Checks the brief's spec.
    """
    L = int(fcfg["numtaps"])
    if L % 2 != 1:
        raise ValueError("numtaps must be odd (integer group delay)")
    h = sps.firwin(L, fcfg["cutoff_hz"], window=("kaiser", fcfg["kaiser_beta"]), fs=fs_in)
    n = np.arange(L)
    h = h / np.abs(np.sum(h * np.exp(-2j * np.pi * F_NOM * n / fs_in)))
    w, H = sps.freqz(h, worN=8192, fs=fs_in)
    A = np.abs(H)
    pb = 20 * np.log10(A[w <= fcfg["passband_hz"]])
    sb = 20 * np.log10(A[w >= fcfg["stopband_hz"]].max())
    if -sb < fcfg["min_atten_db"]:
        raise ValueError(f"stopband attenuation {-sb:.1f} dB < {fcfg['min_atten_db']} dB")
    return {
        "description": "Anti-alias FIR for fs_in -> fs_out decimation (gridmon v0.2). y[n] = sum_i h[i] x[D*n + L-1 - i]; "
                       "gain exactly 1 at 50 Hz; linear phase; group delay (L-1)/2 input samples.",
        "fs_in_hz": fs_in, "decimation": DECIM, "fs_out_hz": fs_in / DECIM,
        "numtaps": L, "cutoff_hz": fcfg["cutoff_hz"], "window": ["kaiser", fcfg["kaiser_beta"]],
        "group_delay_samples_in": (L - 1) // 2, "group_delay_s": (L - 1) / 2 / fs_in,
        "passband_ripple_db": [float(pb.min()), float(pb.max())], "passband_hz": fcfg["passband_hz"],
        "stopband_atten_db": float(-sb), "stopband_hz": fcfg["stopband_hz"],
        "gain_at_50hz": 1.0,
        "coefficients": [float(x) for x in h],
    }


def save_aa_fir(fir: dict[str, Any], path: Path = AA_FIR_PATH) -> None:
    path.write_text(json.dumps(fir, indent=1), encoding="utf-8")


def save_aa_fir_if_changed(fir: dict[str, Any], path: Path = AA_FIR_PATH, atol: float = 1e-12) -> bool:
    """Write the design only if it differs from the stored file (same length, max |dh| > atol).

    A redesign on another platform differs from the committed coefficients in the last bit (libm),
    which would change the config hash and, rarely, an int16 count. Keeping the committed file makes
    the coefficient bits platform-independent (docs/ENVIRONMENT.md). Returns True if written.
    """
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        a, b = np.asarray(old.get("coefficients", []), float), np.asarray(fir["coefficients"], float)
        if a.shape == b.shape and float(np.max(np.abs(a - b))) <= atol:
            return False
    save_aa_fir(fir, path)
    return True


def load_aa_fir(path: Path = AA_FIR_PATH) -> dict[str, Any]:
    fir = json.loads(path.read_text(encoding="utf-8"))
    fir["h"] = np.asarray(fir["coefficients"], dtype=float)
    return fir


# ----------------------------------------------------------------------------------------------
# Analog synthesis
# ----------------------------------------------------------------------------------------------

def synth_time(numtaps: int) -> np.ndarray:
    """20 kHz analog time grid t_k = (k - gd)/fs_synth, k = 0..K-1, K = D*(N-1) + L (gd-compensated)."""
    gd = (numtaps - 1) // 2
    K = DECIM * (N_RECORD - 1) + numtaps
    return (np.arange(K) - gd) / FS_SYNTH


def synth_phase_voltages(pre: PreEvent, ev: Event, t: np.ndarray) -> np.ndarray:
    """Zero-sequence-free phase voltages [3, K] (pu), brief B6.1:

      v_k(t) = Re{(V1(t) r1_k + V2(t) r2_k) e^{j theta(t)}} + sum_h A_h(t) cos(h (theta(t) - 2 pi k/3) + phi_h)
               + transient_k(t),
    r1 = [1, a^2, a], r2 = [1, a, a^2]. Harmonic frequencies track f(t) through theta(t).
    Transient (shift nuisance): balanced damped oscillation in phase voltages,
      A e^{-(t-ts)/tau} cos(2 pi f_tr (t - ts) + phi - 2 pi k/3) for t >= ts.
    Interharmonic (S2E1 novel nuisance, D-080): balanced positive-sequence component over the whole record,
      a_ih u_pre cos(2 pi f_ih t + phi_ih - 2 pi k/3)  (fixed f_ih; LL amplitude = a_ih u_pre as well).
    """
    V1, V2 = fundamental(pre, ev, t)
    _, theta = freq_theta(pre, ev, t)
    E = np.exp(1j * theta)
    P, Q = V1 * E, V2 * E
    a = A_OP
    r1 = (1.0, a * a, a)
    r2 = (1.0, a, a * a)
    v = np.stack([np.real(P * r1[k] + Q * r2[k]) for k in range(3)])
    for h, A_h, ph in zip(HARMONICS, harmonic_levels(pre, ev, t), pre.harm_phase):
        if np.all(A_h == 0.0):
            continue
        Eh = A_h * np.exp(1j * (h * theta + ph))
        for k in range(3):
            v[k] += np.real(Eh * np.exp(-1j * h * 2 * np.pi * k / 3))
    if pre.tr_on:
        tau = t - pre.tr_t
        env = np.where(tau >= 0.0, pre.tr_amp * np.exp(-np.clip(tau, 0.0, None) / pre.tr_tau), 0.0)
        for k in range(3):
            v[k] += env * np.cos(2 * np.pi * pre.tr_f * tau + pre.tr_phase - 2 * np.pi * k / 3)
    if pre.ih_on:
        A_ih = pre.ih_amp * pre.u_pre
        for k in range(3):
            v[k] += A_ih * np.cos(2 * np.pi * pre.ih_f * t + pre.ih_phase - 2 * np.pi * k / 3)
    return v


def phase_to_ll(v: np.ndarray) -> np.ndarray:
    """LL voltages: v_ab = (v_a - v_b)/sqrt3, v_bc = (v_b - v_c)/sqrt3, v_ca = (v_c - v_a)/sqrt3."""
    return np.stack([(v[0] - v[1]) / SQRT3, (v[1] - v[2]) / SQRT3, (v[2] - v[0]) / SQRT3])


def synth_analog_ll(pre: PreEvent, ev: Event, t: np.ndarray) -> np.ndarray:
    """LL voltages at 20 kHz [3, K] after per-channel gain/offset: (1 + g_c) v_c + o_c."""
    ll = phase_to_ll(synth_phase_voltages(pre, ev, t))
    g = np.asarray(pre.gain)[:, None]
    o = np.asarray(pre.offset)[:, None]
    return (1.0 + g) * ll + o


# ----------------------------------------------------------------------------------------------
# Measurement chain (vectorised over a chunk of records)
# ----------------------------------------------------------------------------------------------

def fir_decimate(s: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Polyphase-equivalent decimating FIR: y[n] = sum_i h[i] s[D n + L - 1 - i], n = 0..N-1 (last axis)."""
    L = h.shape[0]
    win = sliding_window_view(s, L, axis=-1)[..., ::DECIM, :]
    return win @ h[::-1]


def quantise(v: np.ndarray, lsb: float = LSB_PU) -> tuple[np.ndarray, np.ndarray]:
    """12-bit ADC: count = clip(floor(v/lsb + 0.5), -2048, 2047) as int16. Returns (counts, n_saturated per record)."""
    c = np.floor(v / lsb + 0.5)
    sat = (c < COUNT_MIN) | (c > COUNT_MAX)
    c = np.clip(c, COUNT_MIN, COUNT_MAX).astype(np.int16)
    return c, sat.reshape(sat.shape[0], -1).sum(axis=1)


def measure(analog: np.ndarray, snr_db: np.ndarray, noise_rngs: list[np.random.Generator],
            h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """analog [R, 3, K] (pu, 20 kHz) -> int16 counts [R, 3, 5000] and saturated-sample counts [R].

    Noise: white Gaussian, sigma = (1/sqrt2) 10^(-SNR/20) pu, drawn per record from its noise stream
    (shape [3, 5000]); SNR = inf gives no noise.
    """
    y = fir_decimate(analog, h)
    for i, (snr, rng) in enumerate(zip(snr_db, noise_rngs)):
        if np.isfinite(snr):
            y[i] += snr_to_sigma(snr) * rng.standard_normal((3, N_RECORD))
    return quantise(y)


def synth_record(pre: PreEvent, ev: Event, noise_rng: np.random.Generator | None = None,
                 fir: dict[str, Any] | None = None) -> np.ndarray:
    """Convenience: one record -> int16 [3, 5000]."""
    fir = fir or load_aa_fir()
    t = synth_time(len(fir["h"]))
    analog = synth_analog_ll(pre, ev, t)[None]
    rng = noise_rng or np.random.default_rng(0)
    x, _ = measure(analog, np.array([pre.snr_db]), [rng], fir["h"])
    return x[0]


def record_time() -> np.ndarray:
    """Record time axis (s) at the monitor rate: n / fs."""
    return np.arange(N_RECORD) / FS
