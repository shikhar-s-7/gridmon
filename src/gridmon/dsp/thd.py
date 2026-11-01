"""Windowed THD by direct DFT at harmonic bins (brief B7.1 step 7).

Windows w = 0..9 of 500 samples (5 nominal cycles, rectangular), hop 500. Per channel:
  X_h = sum_{j=0}^{499} v[500 w + j] * T[(5 h j) mod 500],  T[i] = exp(-j 2 pi i / 500),  h = 1..15
(bin 5h = 50 h Hz; window starts are multiples of 500, so j = n mod 500).
  THD_c = sqrt(sum_{h=2}^{15} |X_h|^2) / |X_1|;  THD(w) = mean over the three channels.
"""

from __future__ import annotations

import numpy as np

WIN: int = 500
H_MAX: int = 15


def thd_windows(v: np.ndarray) -> np.ndarray:
    """THD per window [..., W] (fraction), W = len // 500. dtype follows v."""
    n = v.shape[-1]
    W = n // WIN
    cd = np.complex64 if v.dtype == np.float32 else np.complex128
    tbl = np.exp(-2j * np.pi * np.arange(WIN) / WIN)
    j = np.arange(WIN)
    h = np.arange(1, H_MAX + 1)
    E = tbl[(5 * h[None, :] * j[:, None]) % WIN].astype(cd)
    seg = v[..., : W * WIN].reshape(v.shape[:-1] + (W, WIN))
    X = seg @ E  # [..., 3, W, 15]
    p = (X.real * X.real + X.imag * X.imag)
    thd_c = np.sqrt(p[..., 1:].sum(axis=-1)) / np.sqrt(p[..., 0])
    return thd_c.mean(axis=-2)
