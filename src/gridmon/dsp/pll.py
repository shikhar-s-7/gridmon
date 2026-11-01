"""DSOGI-PLL at 5 kHz (brief B7.1 step 5) [LIT: Rodriguez et al., PESC 2006].

alpha-beta from LL voltages: v_alpha = (v_ab - v_ca)/sqrt3, v_beta = v_bc
(balanced positive sequence: v_alpha = cos(theta), v_beta = sin(theta)).

SOGI-QSG on alpha and beta, fixed w0 = 2 pi 50 (no FLL), k = sqrt2:
  D(s) = k w s / (s^2 + k w s + w^2),  Q(s) = k w^2 / (s^2 + k w s + w^2)
Tustin s = c (z-1)/(z+1), c = 2/Ts, with w prewarped, w = c tan(w0 Ts/2), so the discrete
resonance is exactly 50 Hz. With a0 = c^2 + k w c + w^2:
  D: b = [k w c, 0, -k w c]/a0;      Q: b = [k w^2, 2 k w^2, k w^2]/a0
  a1 = 2 (w^2 - c^2)/a0,  a2 = (c^2 - k w c + w^2)/a0
realised as direct form II transposed biquads (2 states each, 4 filters).
Positive-sequence calculator: va+ = (va' - qvb')/2, vb+ = (qva' + vb')/2.
SRF-PLL: e = (-va+ sin th + vb+ cos th) / max(|v+|, 0.05)  (~ sin(theta - th));
PI (Tustin): u[n] = u[n-1] + b0 e[n] + b1 e[n-1], b0 = Kp + Ki Ts/2, b1 = -Kp + Ki Ts/2,
  Kp = 2 zeta wn, Ki = wn^2 (wn = 2 pi 20 rad/s, zeta = 0.707);
w_hat[n] = 2 pi 50 + u[n];  f_pll[n] = w_hat[n]/(2 pi);  th[n+1] = wrap(th[n] + w_hat[n] Ts).
Initial conditions: SOGI and PI states 0, e[-1] = 0, th[0] = atan2(v_beta[0], v_alpha[0]).
"""

from __future__ import annotations

import numpy as np

from ..units import F_NOM, SQRT3, TS


def sogi_coeffs(f0: float = F_NOM, k: float = np.sqrt(2.0), ts: float = TS) -> dict[str, np.ndarray]:
    """Tustin (prewarped at f0) coefficients of the SOGI D and Q filters (float64)."""
    c = 2.0 / ts
    w = c * np.tan(2 * np.pi * f0 * ts / 2.0)
    a0 = c * c + k * w * c + w * w
    a = np.array([1.0, 2.0 * (w * w - c * c) / a0, (c * c - k * w * c + w * w) / a0])
    bd = np.array([k * w * c, 0.0, -k * w * c]) / a0
    bq = np.array([k * w * w, 2.0 * k * w * w, k * w * w]) / a0
    return {"a": a, "bd": bd, "bq": bq, "w_prewarped": np.array(w)}


def pi_coeffs(wn: float = 2 * np.pi * 20.0, zeta: float = 0.707, ts: float = TS) -> dict[str, float]:
    """PI gains for the linearised loop th/theta = (Kp s + Ki)/(s^2 + Kp s + Ki), Tustin-discretised."""
    kp = 2.0 * zeta * wn
    ki = wn * wn
    return {"kp": kp, "ki": ki, "b0": kp + ki * ts / 2.0, "b1": -kp + ki * ts / 2.0}


def dsogi_pll(v: np.ndarray, f0: float = F_NOM, ts: float = TS, wn: float = 2 * np.pi * 20.0,
              zeta: float = 0.707, vmin: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Run the DSOGI-PLL over records v [R, 3, n] (pu). Returns f_pll [R, n] (Hz) and theta_hat [R, n].

    Loops over time; all records are processed as arrays. Arithmetic in v.dtype (float32 for the
    C-port robustness check).
    """
    dt = v.dtype.type
    sc = sogi_coeffs(f0, ts=ts)
    pc = pi_coeffs(wn, zeta, ts)
    a1, a2 = dt(sc["a"][1]), dt(sc["a"][2])
    bd0, bd2 = dt(sc["bd"][0]), dt(sc["bd"][2])
    bq0, bq1, bq2 = (dt(x) for x in sc["bq"])
    b0, b1 = dt(pc["b0"]), dt(pc["b1"])
    w0, tsd = dt(2 * np.pi * f0), dt(ts)
    two_pi, pi, half, vmn = dt(2 * np.pi), dt(np.pi), dt(0.5), dt(vmin)
    inv_sqrt3 = dt(1.0 / SQRT3)

    R, _, n = v.shape
    va_all = (v[:, 0, :] - v[:, 2, :]) * inv_sqrt3
    vb_all = v[:, 1, :]
    z = np.zeros(R, dtype=v.dtype)
    sda1, sda2, sqa1, sqa2 = z.copy(), z.copy(), z.copy(), z.copy()
    sdb1, sdb2, sqb1, sqb2 = z.copy(), z.copy(), z.copy(), z.copy()
    u, e_prev = z.copy(), z.copy()
    th = np.arctan2(vb_all[:, 0], va_all[:, 0]).astype(v.dtype)
    f_out = np.empty((R, n), dtype=v.dtype)
    th_out = np.empty((R, n), dtype=v.dtype)
    inv_2pi = dt(1.0 / (2 * np.pi))
    for i in range(n):
        xa = va_all[:, i]
        xb = vb_all[:, i]
        # SOGI on alpha (DF-II transposed; D has b1 = 0)
        da = bd0 * xa + sda1
        sda1 = -a1 * da + sda2
        sda2 = bd2 * xa - a2 * da
        qa = bq0 * xa + sqa1
        sqa1 = bq1 * xa - a1 * qa + sqa2
        sqa2 = bq2 * xa - a2 * qa
        # SOGI on beta
        db = bd0 * xb + sdb1
        sdb1 = -a1 * db + sdb2
        sdb2 = bd2 * xb - a2 * db
        qb = bq0 * xb + sqb1
        sqb1 = bq1 * xb - a1 * qb + sqb2
        sqb2 = bq2 * xb - a2 * qb
        # positive sequence
        vap = (da - qb) * half
        vbp = (qa + db) * half
        mag = np.sqrt(vap * vap + vbp * vbp)
        e = (-vap * np.sin(th) + vbp * np.cos(th)) / np.maximum(mag, vmn)
        u = u + b0 * e + b1 * e_prev
        e_prev = e
        w = w0 + u
        f_out[:, i] = w * inv_2pi
        th_out[:, i] = th
        th = th + w * tsd
        th = th - two_pi * np.floor((th + pi) / two_pi)
    return f_out, th_out
