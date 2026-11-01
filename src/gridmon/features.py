"""Record-level features (brief B7.2) from the DSP traces. Input: int16 waveforms only (rule A1.5).

Windows on the phasor grid (m = 0..M-1, n_m = 5m + 4, valid m >= 19):
  REF = {m : n_m <= 749}   (first 0.15 s, assumed pre-event)
  OBS = {m : n_m >= 750}   (rest of the record)
  END = last 100 phasor samples (0.1 s)
For the pre-event-only control the same function runs on the first 750 samples with OBS = REF.
Every feature is phase-permutation invariant (per-channel statistics are sorted or averaged).
Definitions, units and float32 tolerances: docs/FEATURES.md; DSP: docs/DSP_SPEC.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .dsp import envelope as env
from .dsp.freq import df_dft, jump_metric, rocof
from .dsp.phasor import M_FIRST, STEP, dft_phasors, sequence
from .dsp.pll import dsogi_pll
from .dsp.rms import urms_end_samples, urms_half
from .dsp.thd import thd_windows
from .units import F_NOM, FS, LSB_PU

# name, unit, float32 abs tolerance (C port), short definition
FEATURE_SPEC: list[tuple[str, str, float, str]] = [
    ("urms_min_1", "pu", 1e-5, "smallest per-channel min of Urms(1/2)"),
    ("urms_min_2", "pu", 1e-5, "middle per-channel min of Urms(1/2)"),
    ("urms_min_3", "pu", 1e-5, "largest per-channel min of Urms(1/2)"),
    ("urms_max_1", "pu", 1e-5, "smallest per-channel max of Urms(1/2)"),
    ("urms_max_2", "pu", 1e-5, "middle per-channel max of Urms(1/2)"),
    ("urms_max_3", "pu", 1e-5, "largest per-channel max of Urms(1/2)"),
    ("t_below_090", "s", 0.0101, "time with any channel Urms(1/2) < 0.90 (0.01 s per value)"),
    ("t_above_110", "s", 0.0101, "time with any channel Urms(1/2) > 1.10"),
    ("v1_ref", "pu", 1e-5, "median |V1| over REF"),
    ("v1_min_rel", "1", 1e-5, "min |V1| over OBS / v1_ref"),
    ("v1_max_rel", "1", 1e-5, "max |V1| over OBS / v1_ref"),
    ("v1_end_rel", "1", 1e-5, "mean |V1| over END / v1_ref"),
    ("u2_ref", "1", 1e-5, "median |V2|/max(|V1|,0.05) over REF"),
    ("u2_max", "1", 1e-5, "max |V2|/max(|V1|,0.05) over OBS"),
    ("u2_end", "1", 1e-5, "mean |V2|/max(|V1|,0.05) over END"),
    ("rho", "1", 5e-4, "Bollen ratio |V2d(m*)-V2d_ref|/max(v1_ref-|V1(m*)|,0.01), m* = argmin_OBS |V1|"),
    ("j_max_deg", "deg", 1e-3, "max |J| over OBS"),
    ("j_sign", "1", 0.0, "sign of J at the first OBS sample with |J| >= 0.5 max|J| (first lobe = jump sign); compared when j_max_deg >= 1"),
    ("f_ref", "Hz", 1e-4, "median f_dft over REF"),
    ("df_end", "Hz", 1e-4, "mean f_dft over END (jump-masked) - f_ref"),
    ("df_max", "Hz", 1e-4, "max |f_dft - f_ref| over OBS (jump-masked)"),
    ("rocof_max", "Hz/s", 1e-3, "max |ROCOF| over OBS (windows touching the jump mask excluded)"),
    ("t_rocof_02", "s", 0.0021, "time with |ROCOF| > 0.2 Hz/s over OBS (0.001 s per sample)"),
    ("rocof_end", "Hz/s", 1e-3, "mean ROCOF over END (masked)"),
    ("dfpll_max", "Hz", 3e-3, "max |f_pll - f_ref| over OBS"),
    ("t_fpll_05", "s", 0.0021, "time with |f_pll - f_ref| > 0.5 Hz over OBS"),
    ("env_mag_amp_pct", "%", 1e-3, "envelope |V1| spectral peak amplitude (3-30 Hz)"),
    ("env_mag_freq", "Hz", 0.0, "frequency of that peak; compared when aux_env_mag_gap >= 1e-3 (no near-tie)"),
    ("env_mag_prom", "1", 0.05, "peak / band median of that spectrum; compared when aux_env_mag_gap >= 1e-3"),
    ("env_ang_amp_deg", "deg", 1e-3, "envelope angle spectral peak amplitude"),
    ("env_ang_freq", "Hz", 0.0, "frequency of that peak; compared when aux_env_ang_gap >= 1e-3"),
    ("env_ang_prom", "1", 0.05, "peak / band median; compared when aux_env_ang_gap >= 1e-3"),
    ("env_mag_rms_pct", "%", 1e-3, "RMS of the detrended |V1| envelope"),
    ("env_ang_rms_deg", "deg", 1e-3, "RMS of the detrended angle envelope"),
    ("thd_first", "1", 1e-6, "THD of the first 500-sample window (mean over channels)"),
    ("thd_last", "1", 1e-6, "THD of the last window"),
    ("thd_max", "1", 1e-6, "max THD over windows"),
]
FEATURE_NAMES: list[str] = [f[0] for f in FEATURE_SPEC]
AUX_NAMES: list[str] = ["aux_t_dip", "aux_t_swell", "aux_t_jump", "aux_t_freq", "aux_t_osc_mag", "aux_t_osc_ang",
                         "aux_env_mag_gap", "aux_env_ang_gap"]
# Float32 / C-port comparison rules for argmax-type features. A feature is compared only on records
# where every gate (reference column >= value) holds: no near-tie between spectral cells, and a peak
# well above float32 rounding of the envelope (0.01 % / 0.01 deg, ~100x below any rule threshold).
TOL_GATES: dict[str, list[tuple[str, float]]] = {
    "j_sign": [("j_max_deg", 1.0)],
    "env_mag_freq": [("aux_env_mag_gap", 1e-3), ("env_mag_amp_pct", 0.01)],
    "env_mag_prom": [("aux_env_mag_gap", 1e-3), ("env_mag_amp_pct", 0.01)],
    "env_ang_freq": [("aux_env_ang_gap", 1e-3), ("env_ang_amp_deg", 0.01)],
    "env_ang_prom": [("aux_env_ang_gap", 1e-3), ("env_ang_amp_deg", 0.01)],
}
REL_TOL: dict[str, float] = {"env_mag_prom": 0.01, "env_ang_prom": 0.01}  # |d| <= tol + rel * |ref|


def tolerance_check(F_ref: pd.DataFrame, F_test: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compare two feature tables with the documented tolerances (FEATURE_SPEC, TOL_GATES, REL_TOL)."""
    out: dict[str, dict[str, Any]] = {}
    for name, unit, tol, _ in FEATURE_SPEC:
        a = F_ref[name].to_numpy(dtype=float)
        d = np.abs(a - F_test[name].to_numpy(dtype=float))
        gate = np.isfinite(a)
        for col, lo in TOL_GATES.get(name, []):
            gate &= F_ref[col].to_numpy(dtype=float) >= lo
        lim = tol + REL_TOL.get(name, 0.0) * np.abs(a)
        bad = gate & ~(d <= lim)
        out[name] = {"unit": unit, "tol": tol, "rel_tol": REL_TOL.get(name, 0.0), "n_compared": int(gate.sum()),
                     "max_abs_diff": float(d[gate].max()) if gate.any() else 0.0, "n_over_tol": int(bad.sum())}
    return out


def compute_traces(x: np.ndarray, lsb: float = LSB_PU, dtype=np.float64) -> dict[str, np.ndarray]:
    """DSP traces of records x (int16 [R, 3, n]) in the given arithmetic dtype."""
    dt = np.dtype(dtype)
    v = np.asarray(x).astype(dt) * dt.type(lsb)
    urms = urms_half(v)
    X = dft_phasors(v)
    v1, v2 = sequence(X)
    ang = np.angle(v1)
    dfd = df_dft(ang)
    fp, _ = dsogi_pll(v)
    return {
        "v": v, "urms": urms, "v1": v1, "v2": v2, "ang": ang, "df_dft": dfd,
        "f_dft": dt.type(F_NOM) + dfd, "rocof_raw": rocof(dfd), "J": jump_metric(ang),
        "f_pll": fp[:, STEP - 1::STEP], "thd": thd_windows(v),
    }


def _first_true_time(mask: np.ndarray, times: np.ndarray) -> np.ndarray:
    any_ = mask.any(axis=-1)
    idx = np.argmax(mask, axis=-1)
    return np.where(any_, times[idx], np.nan)


def features_from_traces(tr: dict[str, np.ndarray], fcfg: dict[str, Any],
                         ref_end: int | None = None, obs_start: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Feature vectors (dict name -> [R]) and extra traces (masks, envelopes, spectra)."""
    ref_end = fcfg["ref_end_sample"] if ref_end is None else ref_end
    obs_start = fcfg["obs_start_sample"] if obs_start is None else obs_start
    v1, v2 = tr["v1"], tr["v2"]
    dt = v1.real.dtype
    T = dt.type
    R, M = v1.shape
    m = np.arange(M)
    n_m = STEP * m + STEP - 1
    t_m = (n_m / FS).astype(dt)
    valid = m >= M_FIRST
    ref = valid & (n_m <= ref_end)
    obs = valid & (n_m >= obs_start)
    end = valid & (m >= M - fcfg["end_window_phasors"])
    F: dict[str, np.ndarray] = {}
    A: dict[str, np.ndarray] = {}

    # --- Urms(1/2)
    urms = tr["urms"]
    mins = np.sort(urms.min(axis=-1), axis=-1)
    maxs = np.sort(urms.max(axis=-1), axis=-1)
    for i in range(3):
        F[f"urms_min_{i + 1}"] = mins[:, i]
        F[f"urms_max_{i + 1}"] = maxs[:, i]
    below = urms.min(axis=1) < T(0.90)
    above = urms.max(axis=1) > T(1.10)
    F["t_below_090"] = below.sum(axis=-1) * T(0.01)
    F["t_above_110"] = above.sum(axis=-1) * T(0.01)
    t_urms = (urms_end_samples(urms.shape[-1]) - 49).astype(dt) / T(FS)  # window centre
    A["aux_t_dip"] = _first_true_time(below, t_urms)
    A["aux_t_swell"] = _first_true_time(above, t_urms)

    # --- magnitudes and unbalance
    a1, a2 = np.abs(v1), np.abs(v2)
    v1_ref = np.median(a1[:, ref], axis=-1)
    F["v1_ref"] = v1_ref
    F["v1_min_rel"] = a1[:, obs].min(axis=-1) / v1_ref
    F["v1_max_rel"] = a1[:, obs].max(axis=-1) / v1_ref
    F["v1_end_rel"] = a1[:, end].mean(axis=-1) / v1_ref
    u2 = a2 / np.maximum(a1, T(fcfg["v1_floor"]))
    F["u2_ref"] = np.median(u2[:, ref], axis=-1)
    F["u2_max"] = u2[:, obs].max(axis=-1)
    F["u2_end"] = u2[:, end].mean(axis=-1)

    # --- frequency reference
    f_dft = tr["f_dft"]
    f_ref = np.nanmedian(np.where(ref, f_dft, np.nan), axis=-1).astype(dt)
    F["f_ref"] = f_ref

    # --- Bollen ratio on V2 de-rotated by the reference frequency
    rot = np.exp((-2j * np.pi * (f_ref.astype(np.float64) - F_NOM))[:, None] * t_m[None, :].astype(np.float64)).astype(v2.dtype)
    v2d = v2 * rot
    v2r = np.median(v2d.real[:, ref], axis=-1) + 1j * np.median(v2d.imag[:, ref], axis=-1)
    obs_idx = np.where(obs)[0]
    mstar = obs_idx[np.argmin(a1[:, obs], axis=-1)]
    rows = np.arange(R)
    den = np.maximum(v1_ref - a1[rows, mstar], T(fcfg["rho_den_floor"]))
    F["rho"] = (np.abs(v2d[rows, mstar] - v2r.astype(v2.dtype)) / den).astype(dt)

    # --- jump metric and frequency mask
    J = tr["J"]
    aJ = np.where(obs[None, :] & np.isfinite(J), np.abs(J), T(-1.0))
    mJ = np.argmax(aJ, axis=-1)
    jmax = aJ[rows, mJ]
    F["j_max_deg"] = np.degrees(jmax).astype(dt)
    first_lobe = np.argmax(aJ >= T(0.5) * jmax[:, None], axis=-1)  # a step gives +dphi then -dphi
    F["j_sign"] = np.sign(J[rows, first_lobe]).astype(dt)
    detected = F["j_max_deg"] > T(fcfg["jump_detect_deg"])
    h = fcfg["jump_mask_half"]
    lo, hi = (mJ - h)[:, None], (mJ + h)[:, None]
    fmask = detected[:, None] & (m[None, :] >= lo) & (m[None, :] <= hi)
    rmask = detected[:, None] & (m[None, :] >= lo) & (m[None, :] <= hi + 99)
    A["aux_t_jump"] = np.where(detected, _first_true_time(obs[None, :] & (np.abs(np.nan_to_num(J)) > np.radians(T(fcfg["jump_detect_deg"]))), t_m), np.nan)

    dfr = np.where(fmask, np.nan, f_dft - f_ref[:, None])
    df_end = _nanmean(np.where(end[None, :], dfr, np.nan))
    fallback = ~np.isfinite(df_end)  # whole END window masked: use the unmasked mean
    if fallback.any():
        df_end[fallback] = _nanmean(np.where(end[None, :], f_dft - f_ref[:, None], np.nan))[fallback]
    F["df_end"] = df_end.astype(dt)
    adf = np.where(obs[None, :], np.abs(dfr), np.nan)
    F["df_max"] = _nanmax(adf)
    A["aux_t_freq"] = _first_true_time(np.nan_to_num(adf, nan=0.0) > T(fcfg["freq_onset_hz"]), t_m)
    rc = np.where(rmask, np.nan, tr["rocof_raw"])
    arc = np.where(obs[None, :], np.abs(rc), np.nan)
    F["rocof_max"] = _nanmax(arc)
    F["t_rocof_02"] = (np.nan_to_num(arc, nan=0.0) > T(fcfg["rocof_flag_hz_s"])).sum(axis=-1) * T(0.001)
    F["rocof_end"] = _nanmean(np.where(end[None, :], rc, np.nan))

    # --- PLL
    dfp = np.abs(tr["f_pll"] - f_ref[:, None])
    F["dfpll_max"] = np.where(obs[None, :], dfp, -np.inf).max(axis=-1) if obs.any() else np.full(R, np.nan, dtype=dt)
    F["t_fpll_05"] = (obs[None, :] & (dfp > T(fcfg["fpll_flag_hz"]))).sum(axis=-1) * T(0.001)

    # --- envelope spectra
    e_mag, e_ang = env.envelopes(v1, v1_ref, f_ref)
    r_mag = env.detrend_median(e_mag)
    r_ang = env.detrend_median(e_ang)
    P = e_mag.shape[-1]
    t_p = ((STEP * (env.BLOCK * np.arange(P) + env.BLOCK // 2) + STEP - 1) / FS).astype(dt)
    if P >= env.P_FIRST + env.HOP * (env.N_WIN - 1) + env.WIN:
        S_mag = env.spectra(r_mag * T(100.0))
        S_ang = env.spectra(np.degrees(r_ang).astype(dt))
        F["env_mag_amp_pct"], F["env_mag_freq"], F["env_mag_prom"], A["aux_env_mag_gap"] = env.peak_stats(S_mag)
        F["env_ang_amp_deg"], F["env_ang_freq"], F["env_ang_prom"], A["aux_env_ang_gap"] = env.peak_stats(S_ang)
    else:
        S_mag = S_ang = np.full((R, env.N_WIN, len(env.FREQS)), np.nan, dtype=dt)
        for k in ("env_mag_amp_pct", "env_mag_freq", "env_mag_prom", "env_ang_amp_deg", "env_ang_freq", "env_ang_prom"):
            F[k] = np.full(R, np.nan, dtype=dt)
        A["aux_env_mag_gap"] = A["aux_env_ang_gap"] = np.full(R, np.nan, dtype=dt)
    F["env_mag_rms_pct"] = (np.sqrt(np.nanmean(r_mag * r_mag, axis=-1)) * T(100.0)).astype(dt)
    F["env_ang_rms_deg"] = np.degrees(np.sqrt(np.nanmean(r_ang * r_ang, axis=-1))).astype(dt)
    with np.errstate(invalid="ignore"):
        A["aux_t_osc_mag"] = _first_true_time(np.nan_to_num(np.abs(r_mag) * T(100.0)) > T(0.5) * F["env_mag_amp_pct"][:, None], t_p)
        A["aux_t_osc_ang"] = _first_true_time(np.nan_to_num(np.degrees(np.abs(r_ang))) > T(0.5) * F["env_ang_amp_deg"][:, None], t_p)

    # --- THD
    thd = tr["thd"]
    F["thd_first"] = thd[:, 0]
    F["thd_last"] = thd[:, -1]
    F["thd_max"] = thd.max(axis=-1)

    extra = {"fmask": fmask, "rmask": rmask, "e_mag": e_mag, "e_ang": e_ang, "r_mag": r_mag, "r_ang": r_ang,
             "S_mag": S_mag, "S_ang": S_ang, "v2d_ref": v2r, "m_star": mstar, "m_jump": mJ}
    return {**{k: F[k] for k in FEATURE_NAMES}, **A}, extra


def _nanmax(a: np.ndarray) -> np.ndarray:
    ok = np.isfinite(a).any(axis=-1)
    return np.where(ok, np.where(np.isfinite(a), a, -np.inf).max(axis=-1), np.nan).astype(a.dtype)


def _nanmean(a: np.ndarray) -> np.ndarray:
    ok = np.isfinite(a).any(axis=-1)
    s = np.where(np.isfinite(a), a, 0).sum(axis=-1)
    c = np.isfinite(a).sum(axis=-1)
    return np.where(ok, s / np.maximum(c, 1), np.nan).astype(a.dtype)


def extract_features(X: np.ndarray, fcfg: dict[str, Any], lsb: float = LSB_PU, dtype=np.float64,
                     n_samples: int | None = None, ref_end: int | None = None, obs_start: int | None = None,
                     progress: bool = False) -> pd.DataFrame:
    """Features + aux columns for all records of X (int16 [N, 3, 5000]), chunked. Float64 output.

    n_samples truncates every record (pre-event-only control: n_samples = 750, obs_start = 0).
    """
    N = X.shape[0]
    chunk = int(fcfg["chunk_records"])
    parts = []
    it = range(0, N, chunk)
    if progress:
        from tqdm import tqdm
        it = tqdm(it, unit="chunk", leave=False)
    for c0 in it:
        x = np.asarray(X[c0:c0 + chunk])
        if n_samples is not None:
            x = x[..., :n_samples]
        tr = compute_traces(x, lsb, dtype)
        feats, _ = features_from_traces(tr, fcfg, ref_end, obs_start)
        parts.append(pd.DataFrame({k: np.asarray(v, dtype=np.float64) for k, v in feats.items()}))
    return pd.concat(parts, ignore_index=True)
