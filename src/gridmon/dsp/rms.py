"""RMS measurements.

urms_half: IEC 61000-4-30-style Urms(1/2) on a fixed grid (brief B7.1 step 1):
  value k uses samples [50k, 50k+99] (window N = 100, refresh N/2 = 50; first value ends at n = 99),
  Urms_k = sqrt(2 * mean(v^2)) in pu of nominal RMS. 5000 samples -> 99 values per channel.
  The IEC method aligns windows to zero crossings; the fixed grid is a logged simplification.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..units import N_CYCLE


def urms_half(v: np.ndarray, n: int = N_CYCLE) -> np.ndarray:
    """Urms(1/2) [..., K], K = (len - n)//(n/2) + 1, in pu of nominal RMS; dtype follows v.

    C note: for each k, acc = sum_{i=0}^{n-1} v[50k+i]^2; Urms = sqrtf(2*acc/n). No state between windows.
    """
    two = v.dtype.type(2.0)
    sq = v * v
    win = sliding_window_view(sq, n, axis=-1)[..., :: n // 2, :]
    return np.sqrt(two * win.mean(axis=-1))


def urms_end_samples(n_values: int, n: int = N_CYCLE) -> np.ndarray:
    """Sample index of the last sample of each Urms(1/2) window: 50k + 99."""
    return np.arange(n_values) * (n // 2) + n - 1


def rms_one_cycle_sliding(v: np.ndarray, n: int = N_CYCLE) -> np.ndarray:
    """Per-sample one-cycle sliding RMS (pu RMS), defined for samples n-1 .. end (NaN before).

    Used for M1 label-margin waveform checks and figures (not a baseline feature).
    """
    v = np.asarray(v, dtype=float)
    c = np.cumsum(np.concatenate([np.zeros(v.shape[:-1] + (1,)), v * v], axis=-1), axis=-1)
    ms = (c[..., n:] - c[..., :-n]) / n
    out = np.full(v.shape, np.nan)
    out[..., n - 1:] = np.sqrt(2.0 * np.clip(ms, 0.0, None))
    return out
