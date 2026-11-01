"""DFT-based frequency, ROCOF and the phase-jump metric at the 1 kHz phasor rate (brief B7.1 steps 3-4).

All arrays are indexed by phasor index m (n = 5m + 4); NaN marks samples whose inputs are not yet valid.
  df_raw(m) = wrap(angV1(m) - angV1(m-20)) / (2 pi * 0.02)          [Hz]  valid m >= 39
  f_dft(m)  = 50 + mean(df_raw(m-19 .. m))   (one-cycle moving average) valid m >= 58
  ROCOF(m)  = least-squares slope of f_dft(m-99 .. m)              [Hz/s] valid m >= 157
            = sum_{j=0}^{99} c_j * (f_dft(m-99+j) - 50),  c_j = (j - 49.5) / (sum_j (j-49.5)^2 * dt)
  J(m)      = wrap(angV1(m) - 2 angV1(m-20) + angV1(m-40))           [rad]  valid m >= 59
Computing on the deviation f - 50 keeps float32 resolution. ROCOF is evaluated as a direct
100-tap weighted sum (non-recursive) rather than running sums: identical in exact arithmetic,
no float32 drift (DECISIONS D-016).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..units import F_NOM, wrap

DT_PHASOR: float = 0.001  # s
LAG: int = 20  # one nominal cycle at 1 kHz
MA: int = 20
ROCOF_WIN: int = 100


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    """x(m - k) with NaN for m < k."""
    out = np.full_like(x, np.nan)
    out[..., k:] = x[..., :-k]
    return out


def angle(v1: np.ndarray) -> np.ndarray:
    """angV1 = atan2(Im, Re) (float32 for complex64 input)."""
    return np.angle(v1)


def df_dft(ang: np.ndarray) -> np.ndarray:
    """Frequency deviation f_dft - 50 (Hz) after the one-cycle moving average."""
    k = ang.dtype.type(1.0 / (2.0 * np.pi * LAG * DT_PHASOR))
    raw = wrap(ang - _lag(ang, LAG)) * k
    out = np.full_like(raw, np.nan)
    out[..., MA - 1:] = sliding_window_view(raw, MA, axis=-1).mean(axis=-1)
    return out


def f_dft(ang: np.ndarray) -> np.ndarray:
    return ang.dtype.type(F_NOM) + df_dft(ang)


def rocof_kernel(dtype=np.float64, win: int = ROCOF_WIN, dt: float = DT_PHASOR) -> np.ndarray:
    j = np.arange(win) - (win - 1) / 2.0
    return (j / (np.sum(j * j) * dt)).astype(dtype)


def rocof(dfd: np.ndarray) -> np.ndarray:
    """Least-squares slope (Hz/s) of f_dft over the last 100 phasor samples."""
    c = rocof_kernel(dfd.dtype)
    out = np.full_like(dfd, np.nan)
    out[..., ROCOF_WIN - 1:] = sliding_window_view(dfd, ROCOF_WIN, axis=-1) @ c
    return out


def jump_metric(ang: np.ndarray) -> np.ndarray:
    """J(m) = wrap(ang(m) - 2 ang(m-20) + ang(m-40)) (rad)."""
    two = ang.dtype.type(2.0)
    return wrap(ang - two * _lag(ang, LAG) + _lag(ang, 2 * LAG))
