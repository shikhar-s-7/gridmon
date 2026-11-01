"""Fundamental envelope at 100 Hz and its low-frequency spectra (brief B7.1 step 6).

Indices: phasor index m (n = 5m + 4, valid m >= 19); envelope index p = 0..99, block p uses
phasor samples m = 10p .. 10p+9, valid for p >= 2 (p = 0, 1 contain invalid phasors -> NaN).
  e_mag(p) = mean_{m in block} |V1(m)| / |V1_ref|                       (fraction of reference)
  e_ang(p) = mean_{m in block} [ angU(m) - 2 pi (f_ref - 50) t_m ]     (rad), t_m = (5m+4)/fs,
  angU = angle of V1 unwrapped from m = 19: angU(m) = angU(m-1) + wrap(ang(m) - ang(m-1)).
A 10-sample block spans one 100 Hz period, so the 2f ripple caused by unbalance averages out.
Detrending: r(p) = e(p) - median(e(q), q in [max(2, p-10), min(99, p+10)]) (centred 21-sample
running median, truncated at the edges; even counts take the mean of the two middle values).
Monotone trends (ramps, steps) are removed by the median; oscillations remain.
Spectra: windows of L = 50 envelope samples (0.5 s), hop 10, starting at p = 2 + 10q, q = 0..4
(last window covers p = 42..91; DECISIONS D-024). Symmetric Hann w[k] = 0.5 - 0.5 cos(2 pi k/49).
  S_q(f) = (2 / sum(w)) * | sum_{k=0}^{49} w[k] r(s_q + k) exp(-j 2 pi f k / 100) |, f = 3..30 Hz
(twiddles from a 100-entry table indexed by (f k) mod 100). A sinusoid of amplitude A at an
integer frequency peaks at ~A. Per window: peak = max_f S, f_peak = argmax, prominence = peak /
median_f S (28 bins). The record value is taken from the window with the largest peak.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..units import F_NOM, FS, wrap
from .phasor import M_FIRST, STEP

BLOCK: int = 10
P_FIRST: int = 2
MED_HALF: int = 10
WIN: int = 50
HOP: int = 10
N_WIN: int = 5
FREQS: np.ndarray = np.arange(3, 31)
FS_ENV: int = 100


def unwrap_from(ang: np.ndarray, m0: int = M_FIRST) -> np.ndarray:
    """Incremental unwrap along the last axis starting at m0 (NaN before)."""
    out = np.full_like(ang, np.nan)
    d = wrap(np.diff(ang[..., m0:], axis=-1))
    out[..., m0] = ang[..., m0]
    out[..., m0 + 1:] = ang[..., m0:m0 + 1] + np.cumsum(d, axis=-1)
    return out


def envelopes(v1: np.ndarray, v1_ref: np.ndarray, f_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(e_mag [R, P], e_ang [R, P]) with P = M // 10; NaN for p < 2."""
    dt = v1.real.dtype
    M = v1.shape[-1]
    P = M // BLOCK
    t_m = ((STEP * np.arange(M) + STEP - 1) / FS).astype(dt)
    mag = np.abs(v1) / v1_ref[:, None]
    angu = unwrap_from(np.angle(v1))
    ramp = dt.type(2 * np.pi) * (f_ref - dt.type(F_NOM))[:, None] * t_m[None, :]
    angd = angu - ramp
    e_mag = mag[:, : P * BLOCK].reshape(-1, P, BLOCK).mean(axis=-1)
    e_ang = angd[:, : P * BLOCK].reshape(-1, P, BLOCK).mean(axis=-1)
    return e_mag, e_ang


def detrend_median(e: np.ndarray, half: int = MED_HALF, p0: int = P_FIRST) -> np.ndarray:
    """e - centred running median over valid samples p >= p0 (window truncated at the edges)."""
    R, P = e.shape
    valid = e[:, p0:]
    pad = np.full((R, valid.shape[1] + 2 * half), np.nan, dtype=e.dtype)
    pad[:, half:half + valid.shape[1]] = valid
    med = np.nanmedian(sliding_window_view(pad, 2 * half + 1, axis=-1), axis=-1)
    out = np.full_like(e, np.nan)
    out[:, p0:] = valid - med.astype(e.dtype)
    return out


def hann(n: int = WIN, dtype=np.float64) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))).astype(dtype)


def spectra(r: np.ndarray) -> np.ndarray:
    """Envelope spectra [R, N_WIN, len(FREQS)] (same units as r)."""
    dt = r.dtype
    w = hann(WIN, dt)
    scale = dt.type(2.0) / w.sum(dtype=dt)
    tbl = np.exp(-2j * np.pi * np.arange(FS_ENV) / FS_ENV)
    k = np.arange(WIN)
    E = tbl[(FREQS[None, :] * k[:, None]) % FS_ENV].astype(np.complex64 if dt == np.float32 else np.complex128)
    out = np.empty((r.shape[0], N_WIN, len(FREQS)), dtype=dt)
    for q in range(N_WIN):
        s = P_FIRST + HOP * q
        seg = r[:, s:s + WIN] * w
        out[:, q, :] = np.abs(seg @ E) * scale
    return out


def peak_stats(S: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(peak, f_peak, prominence, gap) per record from spectra [R, Q, F], at the window with the largest peak.

    gap = (S1 - S2) / S1 over all (window, frequency) cells, S1 >= S2 the two largest values: a
    near-tie (gap < 1e-3) means f_peak / prominence may legitimately differ in another float
    implementation (the C port compares them only when gap >= 1e-3).
    """
    pk = S.max(axis=-1)
    q = np.argmax(pk, axis=-1)
    rows = np.arange(S.shape[0])
    Sq = S[rows, q, :]
    peak = Sq.max(axis=-1)
    fpk = FREQS[np.argmax(Sq, axis=-1)].astype(S.dtype)
    prom = peak / np.maximum(np.median(Sq, axis=-1), S.dtype.type(1e-12))
    flat = np.sort(S.reshape(S.shape[0], -1), axis=-1)
    gap = (flat[:, -1] - flat[:, -2]) / np.maximum(flat[:, -1], S.dtype.type(1e-12))
    return peak, fpk, prom, gap
