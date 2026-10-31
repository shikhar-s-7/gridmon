"""Units, constants and small helpers shared by the generator and the DSP chain.

Conventions (PROJECT_BRIEF B4, rule A1.2):
- instantaneous voltages in pu of nominal line-to-line (LL) PEAK (nominal amplitude 1.0);
- RMS-type quantities in pu of nominal RMS: RMS_pu = sqrt(2 * mean(v^2)) (1.0 at nominal);
- angles in radians in code (degrees only in reports), frequency in Hz, time in s.
"""

from __future__ import annotations

import numpy as np

F_NOM: float = 50.0  # Hz, nominal grid frequency
FS: int = 5000  # Hz, monitor sampling rate
FS_SYNTH: int = 20000  # Hz, synthesis rate (stand-in for a PCS control-loop ADC rate)
DECIM: int = FS_SYNTH // FS  # 4
N_CYCLE: int = FS // int(F_NOM)  # 100 samples per nominal cycle
N_RECORD: int = FS  # 5000 samples = 1.0 s
TS: float = 1.0 / FS

ADC_BITS: int = 12
ADC_FULL_SCALE_PU: float = 2.0  # +/- 2 pu
LSB_PU: float = 2.0 * ADC_FULL_SCALE_PU / 2**ADC_BITS  # 4/4096 pu per count
COUNT_MIN: int = -(2 ** (ADC_BITS - 1))  # -2048
COUNT_MAX: int = 2 ** (ADC_BITS - 1) - 1  # 2047

A_OP: complex = complex(np.exp(2j * np.pi / 3))  # Fortescue operator a = exp(j*2*pi/3)
SQRT3: float = float(np.sqrt(3.0))

PRE_EVENT_SAMPLES: int = 750  # first 0.15 s of every record is pre-event (B4)


def deg(x):
    """Radians -> degrees."""
    return np.degrees(x)


def rad(x):
    """Degrees -> radians."""
    return np.radians(x)


def wrap(x):
    """Wrap angle(s) to [-pi, pi): wrap(x) = x - 2*pi*floor((x + pi) / (2*pi)). Works for float32."""
    x = np.asarray(x)
    two_pi = x.dtype.type(2.0 * np.pi) if x.dtype.kind == "f" else 2.0 * np.pi
    pi = x.dtype.type(np.pi) if x.dtype.kind == "f" else np.pi
    return x - two_pi * np.floor((x + pi) / two_pi)


def rms_pu(v, axis: int = -1):
    """RMS in pu of nominal RMS for a waveform in pu of nominal peak: sqrt(2 * mean(v^2))."""
    v = np.asarray(v)
    return np.sqrt(2.0 * np.mean(v * v, axis=axis))


def snr_to_sigma(snr_db):
    """Noise standard deviation (pu of peak) for an SNR relative to nominal LL RMS (1/sqrt(2) pu peak)."""
    return (1.0 / np.sqrt(2.0)) * 10.0 ** (-np.asarray(snr_db) / 20.0)


def ll_fundamental_mags(v1, v2):
    """Magnitudes of the three LL fundamental phasors from sequence phasors (complex pu).

    With v_ab = (v_a - v_b)/sqrt(3) etc. and zero-sequence-free phase voltages
    v_k = Re{(V1 r1_k + V2 r2_k) e^{j theta}}, the LL phasors are
      X_ab = V1 e^{j pi/6} + V2 e^{-j pi/6},
      X_bc = a^2 V1 e^{j pi/6} + a V2 e^{-j pi/6},
      X_ca = a V1 e^{j pi/6} + a^2 V2 e^{-j pi/6}.
    Returns an array with a leading axis of length 3 (ab, bc, ca).
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    p = v1 * np.exp(1j * np.pi / 6)
    q = v2 * np.exp(-1j * np.pi / 6)
    a = A_OP
    return np.abs(np.stack([p + q, a * a * p + a * q, a * p + a * a * q]))
