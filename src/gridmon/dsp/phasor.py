"""One-cycle DFT phasors at 1 kHz and symmetrical components (brief B7.1 step 2).

Phasor index m: sample n = 5m + 4 (m = 0..999 for 5000 samples); valid for n >= 99, i.e. m >= 19.
  X_c(m) = (2/N) * sum_{i=0}^{N-1} v_c[n-i] * W[(n-i) mod N],  W[k] = exp(-j 2 pi k / N), N = 100.
Twiddles are indexed by (sample index mod N), so a 50 Hz sinusoid A cos(2 pi 50 t + phi) gives the
stationary phasor A e^{j phi}; off-nominal frequency f rotates it at (f - 50) Hz.
  V1 = (X_ab + a X_bc + a^2 X_ca)/3,  V2 = (X_ab + a^2 X_bc + a X_ca)/3.
For LL inputs V1 = V1_phase e^{j pi/6} and V2 = V2_phase e^{-j pi/6} (magnitudes preserved).
Invalid entries (m < 19) are NaN.

C note: non-recursive. For each m, three 100-term complex MACs against a const twiddle table
(no sliding/recursive DFT, which drifts in float32). Python sums 5-sample blocks then 20 blocks:
the same sum in a different association order (float32 differences are within tolerance).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..units import A_OP, N_CYCLE

STEP: int = 5  # phasor decimation (5 kHz -> 1 kHz)
M_FIRST: int = (N_CYCLE - 1 - (STEP - 1)) // STEP  # 19: first valid phasor index


def complex_dtype(dtype) -> type:
    return np.complex64 if np.dtype(dtype) == np.float32 else np.complex128


def twiddle(n: int, dtype=np.float64) -> np.ndarray:
    """W[k] = exp(-j 2 pi k / n), computed in float64 then cast (C: const table)."""
    return np.exp(-2j * np.pi * np.arange(n) / n).astype(complex_dtype(dtype))


def phasor_sample_index(m_count: int) -> np.ndarray:
    """Sample index n = 5m + 4 of phasor m."""
    return STEP * np.arange(m_count) + STEP - 1


def dft_phasors(v: np.ndarray, n_cycle: int = N_CYCLE) -> np.ndarray:
    """One-cycle DFT phasors [..., 3, M] (M = len//5), NaN for m < 19. dtype follows v (float32 -> complex64)."""
    n = v.shape[-1]
    W = twiddle(n_cycle, v.dtype)
    z = v * W[np.arange(n) % n_cycle]
    M = n // STEP
    blocks = z[..., : M * STEP].reshape(z.shape[:-1] + (M, STEP)).sum(axis=-1)
    nb = n_cycle // STEP
    s = sliding_window_view(blocks, nb, axis=-1).sum(axis=-1)
    X = np.full(z.shape[:-1] + (M,), np.nan, dtype=z.dtype)
    X[..., nb - 1:] = (2.0 / n_cycle) * s
    return X


def sequence(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(V1, V2) from LL phasors X [..., 3, M] (channels ab, bc, ca)."""
    a = X.dtype.type(A_OP)
    a2 = X.dtype.type(A_OP * A_OP)
    third = X.real.dtype.type(1.0 / 3.0)
    xab, xbc, xca = X[..., 0, :], X[..., 1, :], X[..., 2, :]
    v1 = (xab + a * xbc + a2 * xca) * third
    v2 = (xab + a2 * xbc + a * xca) * third
    return v1, v2
