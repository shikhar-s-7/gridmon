"""M2 DSP / feature tests (PROMPT A4 a-g, plus sealed-loader and permutation checks)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gridmon.baseline_rules import classify
from gridmon.dsp.freq import jump_metric
from gridmon.dsp.phasor import dft_phasors, phasor_sample_index, sequence
from gridmon.dsp.pll import dsogi_pll
from gridmon.dsp.rms import urms_end_samples, urms_half
from gridmon.features import FEATURE_SPEC, extract_features, tolerance_check
from gridmon.generator.events import quiet_normal
from gridmon.generator.nuisance import clean_pre_event
from gridmon.generator.phasors import bollen_closed_form
from gridmon.generator.synth import synth_record
from gridmon.io import SealedSplitError, load_baseline_config, load_split, load_waveforms, reports_root
from gridmon.units import FS, LSB_PU


@pytest.fixture(scope="module")
def bcfg():
    return load_baseline_config()


def _sine(A=1.0, phi=0.3, f=50.0, n=5000):
    t = np.arange(n) / FS
    return A * np.cos(2 * np.pi * f * t + phi)


def _dip_event(typ, V=0.5, special="a", psi=0.0, t0=0.3013, dur=0.4):
    lab = "sag_bal" if typ == "A" else "sag_unbal"
    return {"label": lab, "t0": t0, "V": V, "psi": psi, "dur": dur, "rec_abrupt": True, "rec_level": 1.0,
            "rec_tau": 0.1, "sag_type": typ, "special": special}


# ---------------------------------------------------------------------------------------------- (a)
@pytest.mark.parametrize("A,phi", [(1.0, 0.3), (0.5, -2.0), (1.2, 3.0)])
def test_phasor_dft_clean_sine(A, phi):
    v = np.stack([_sine(A, phi), _sine(A, phi - 2 * np.pi / 3), _sine(A, phi + 2 * np.pi / 3)])[None]
    X = dft_phasors(v)[0, 0]
    ok = np.isfinite(X)
    assert ok.sum() == 1000 - 19 and not ok[:19].any()
    assert np.max(np.abs(np.abs(X[ok]) / A - 1)) < 1e-3
    assert np.max(np.abs(np.degrees(np.angle(X[ok] * np.exp(-1j * phi))))) < 0.1


# ---------------------------------------------------------------------------------------------- (b)
def test_urms_clean_and_step():
    v = _sine()[None, None]
    u = urms_half(v)
    assert u.shape[-1] == 99 and np.all(np.abs(u - 1.0) <= 0.002)
    for ts in (0.30137, 0.3051, 0.3099):
        t = np.arange(5000) / FS
        w = _sine() * np.where(t >= ts, 0.5, 1.0)
        u = urms_half(w[None, None])[0, 0]
        k = int(np.argmax(u < 0.90))
        t_det = (urms_end_samples(99)[k]) / FS
        assert u[k] < 0.90 and 0.0 <= t_det - ts <= 0.020


# ---------------------------------------------------------------------------------------------- (c)
@pytest.mark.parametrize("typ", ["A", "C", "D", "F", "G"])
@pytest.mark.parametrize("special", ["a", "c"])
def test_sequence_recovers_bollen(typ, special):
    x = synth_record(clean_pre_event(theta0=1.0), _dip_event(typ, 0.5, special))
    v = x[None].astype(float) * LSB_PU
    v1, v2 = sequence(dft_phasors(v))
    n = phasor_sample_index(1000)
    inside = (n / FS > 0.3013 + 0.03) & (n / FS < 0.3013 + 0.37)
    c1, c2 = bollen_closed_form(typ, 0.5)
    assert np.max(np.abs(np.abs(v1[0, inside]) - abs(c1))) < 0.005
    assert np.max(np.abs(np.abs(v2[0, inside]) - abs(c2))) < 0.005


# ---------------------------------------------------------------------------------------------- (d)
def test_pll_lock_and_ramp():
    recs = [synth_record(clean_pre_event(theta0=th), quiet_normal()) for th in np.linspace(0, 2 * np.pi, 9)[:-1]]
    v = np.stack(recs).astype(float) * LSB_PU
    f, _ = dsogi_pll(v)
    lock = int(0.150 * FS)
    assert np.max(np.abs(f[:, lock:] - 50.0)) <= 0.01
    ev = {"label": "freq_event", "t0": 0.0, "rocof": 1.0, "df_target": 5.0, "dv1": 0.0}
    x = synth_record(clean_pre_event(theta0=0.3), ev).astype(float) * LSB_PU
    f, _ = dsogi_pll(x[None])
    t = np.arange(5000) / FS
    err = f[0, 4000:] - (50.0 + t[4000:])
    assert abs(err.mean()) < 0.05


# ---------------------------------------------------------------------------------------------- (e)
def test_jump_metric_step_and_ramp():
    pre = clean_pre_event(theta0=0.8)
    for dphi in (20.0, -20.0):
        x = synth_record(pre, {"label": "phase_jump", "t0": 0.3013, "dphi": np.radians(dphi), "dv1": 0.0})
        v1, _ = sequence(dft_phasors(x[None].astype(float) * LSB_PU))
        J = np.degrees(jump_metric(np.angle(v1)))[0]
        assert abs(np.nanmax(np.abs(J)) - 20.0) <= 1.0
    x = synth_record(pre, {"label": "freq_event", "t0": 0.2, "rocof": 4.0, "df_target": 5.0, "dv1": 0.0})
    v1, _ = sequence(dft_phasors(x[None].astype(float) * LSB_PU))
    J = np.degrees(jump_metric(np.angle(v1)))[0]
    assert np.nanmax(np.abs(J)) < 1.0


# ---------------------------------------------------------------------------------------------- (f)
@pytest.mark.parametrize("typ,expected", [("A", 0.0), ("C", 1.0), ("D", 1.0), ("F", 0.5), ("G", 0.5)])
def test_bollen_ratio(typ, expected, bcfg):
    xs = [synth_record(clean_pre_event(theta0=0.5 + i), _dip_event(typ, V, sp))
          for i, (V, sp) in enumerate([(0.2, "a"), (0.5, "b"), (0.8, "c")])]
    F = extract_features(np.stack(xs), bcfg["features"])
    assert np.all(np.abs(F["rho"].to_numpy() - expected) <= 0.05), F["rho"].to_numpy()


# ---------------------------------------------------------------------------------------------- (g)
def test_float32_robustness_val(bcfg, fast_flag):
    X, meta = load_waveforms("val", fast=fast_flag)
    F64 = extract_features(X, bcfg["features"], lsb=meta["lsb_pu"], dtype=np.float64)
    F32 = extract_features(X, bcfg["features"], lsb=meta["lsb_pu"], dtype=np.float32)
    ths = {"default": bcfg["rules"]}
    rep_path = reports_root(fast_flag) / "m2_baseline_val.json"
    if rep_path.exists():
        ths["tuned"] = json.loads(rep_path.read_text())["thresholds"]["tuned"]
    for name, th in ths.items():
        a = classify(F64, th)["rule_label"].to_numpy()
        b = classify(F32, th)["rule_label"].to_numpy()
        assert np.array_equal(a, b), f"{name}: {(a != b).sum()} labels changed"
    chk = tolerance_check(F64, F32)
    bad = {k: v for k, v in chk.items() if v["n_over_tol"]}
    assert not bad, bad
    assert all(v["n_compared"] > 0 for v in chk.values())


# ---------------------------------------------------------------------------------------------- extra
def test_sealed_loader_guard():
    for s in ("test_iid", "test_shift"):
        with pytest.raises(SealedSplitError):
            load_split(s)
        with pytest.raises(SealedSplitError):
            load_waveforms(s)


def test_cyclic_channel_rotation_invariance(bcfg):
    pre = clean_pre_event(theta0=0.2, unb=0.01, unb_ang=1.0, harm_level=(0.02, 0.01, 0.0, 0.0), snr_db=50.0)
    xs = [synth_record(pre, _dip_event("D", 0.4, "b")),
          synth_record(pre, {"label": "phase_jump", "t0": 0.31, "dphi": np.radians(-15.0), "dv1": 0.01})]
    x = np.stack(xs)
    F = extract_features(x, bcfg["features"])
    Fr = extract_features(np.roll(x, 1, axis=1), bcfg["features"])
    for fname, _, tol, _ in FEATURE_SPEC:
        assert np.allclose(F[fname], Fr[fname], atol=max(10 * tol, 1e-3), equal_nan=True), fname
