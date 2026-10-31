"""M1 generator tests (PROMPT A3 a-i)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from gridmon import CLASSES
from gridmon.dsp.rms import rms_one_cycle_sliding
from gridmon.generator import dataset as ds
from gridmon.generator.events import freq_theta, margins_ok, quiet_normal
from gridmon.generator.nuisance import clean_pre_event
from gridmon.generator.phasors import (A, A2, bollen_closed_form, bollen_phase_voltages, dip_sequence,
                                       rotate_special_phase, sequence_components)
from gridmon.generator.synth import (load_aa_fir, measure, synth_analog_ll, synth_record, synth_time)
from gridmon.io import load_manifest, load_waveforms
from gridmon.units import FS, LSB_PU, N_CYCLE

from conftest import ALL_SPLITS

TOL = 1e-9


# ---------------------------------------------------------------------------------------------- (a)
@pytest.mark.parametrize("typ", ["A", "C", "D", "F", "G"])
@pytest.mark.parametrize("V", [0.0, 0.5, 0.85])
def test_bollen_sequence_closed_forms(typ, V):
    v0, v1, v2 = sequence_components(bollen_phase_voltages(typ, V))
    c1, c2 = bollen_closed_form(typ, V)
    assert abs(v0) < TOL
    assert abs(v1 - c1) < TOL and abs(v2 - c2) < TOL


@pytest.mark.parametrize("V", [0.0, 0.5, 0.85])
def test_bollen_b_e_without_zero_sequence(V):
    # B minus zero sequence == D with characteristic voltage (1+2V)/3
    v0, v1, v2 = sequence_components(bollen_phase_voltages("B", V))
    d1, d2 = bollen_closed_form("D", (1 + 2 * V) / 3)
    assert abs(v0 - (V - 1) / 3) < TOL
    assert abs(v1 - d1) < TOL and abs(v2 - d2) < TOL
    # E minus zero sequence == G
    v0, v1, v2 = sequence_components(bollen_phase_voltages("E", V))
    g1, g2 = bollen_closed_form("G", V)
    assert abs(v0 - (1 - V) / 3) < TOL
    assert abs(v1 - g1) < TOL and abs(v2 - g2) < TOL


# ---------------------------------------------------------------------------------------------- (b)
@pytest.mark.parametrize("typ", ["A", "C", "D", "F", "G"])
def test_special_phase_rotation(typ):
    V = 0.4 * np.exp(1j * np.radians(-15))
    u = bollen_phase_voltages(typ, V)
    _, v1a, v2a = sequence_components(u)
    _, v1b, v2b = sequence_components(rotate_special_phase(u, "b"))
    _, v1c, v2c = sequence_components(rotate_special_phase(u, "c"))
    assert abs(v1b - v1a) < TOL and abs(v2b - A * v2a) < TOL
    assert abs(v1c - v1a) < TOL and abs(v2c - A2 * v2a) < TOL
    # generator closed form uses the same convention
    r1, r2 = dip_sequence(typ, 0.4, np.radians(-15), "b")
    assert abs(r1 - v1b) < TOL and abs(r2 - v2b) < TOL


# ---------------------------------------------------------------------------------------------- (c)
def _dft50(v, n0=0, n=N_CYCLE):
    k = np.arange(n0, n0 + n)
    return (2.0 / n) * np.sum(v[..., n0:n0 + n] * np.exp(-2j * np.pi * k / N_CYCLE), axis=-1)


def test_clean_nominal_record():
    x = synth_record(clean_pre_event(theta0=0.7), quiet_normal())
    v = x.astype(float) * LSB_PU
    rms = np.sqrt(2.0 * np.mean(v * v, axis=-1))
    assert np.all(np.abs(rms - 1.0) <= 0.002), rms
    X = _dft50(v, 1000)
    lead = np.degrees(np.angle(X[0] / X[1]))
    assert abs(lead - 120.0) < 0.5  # v_ab leads v_bc by 120 deg (abc sequence)
    assert abs(np.degrees(np.angle(X[1] / X[2])) - 120.0) < 0.5


def test_fir_unity_gain_at_50hz():
    fir = load_aa_fir()
    h = fir["h"]
    n = np.arange(len(h))
    H50 = np.sum(h * np.exp(-2j * np.pi * 50.0 * n / fir["fs_in_hz"]))
    assert abs(abs(H50) - 1.0) < 1e-12
    assert fir["stopband_atten_db"] >= 40.0 and fir["passband_hz"] >= 1000.0


# ---------------------------------------------------------------------------------------------- (d)
def test_frequency_ramp_theta():
    pre = clean_pre_event(f_pre=49.95, drift=0.02, theta0=1.1)
    ev = {"label": "freq_event", "t0": 0.3137, "rocof": -2.0, "df_target": 0.8, "dv1": 0.0}
    t = np.arange(-0.01, 1.01, 1.0 / 200000)
    f, theta = freq_theta(pre, ev, t)
    finst = np.gradient(theta, t) / (2 * np.pi)
    err = np.abs(finst - f)[5:-5]
    assert err.max() < 1e-3
    # f(t) is the specified trajectory: ramp -2 Hz/s from t0 until |df| = 0.8 Hz, then hold
    assert abs(f[np.searchsorted(t, 0.99)] - (49.95 + 0.02 * 0.99 - 0.8)) < 1e-6
    # the synthesised waveform carries theta: v_ab = cos(theta + pi/6) for a clean record
    ts = synth_time(len(load_aa_fir()["h"]))
    ll = synth_analog_ll(pre, ev, ts)
    _, th = freq_theta(pre, ev, ts)
    assert np.max(np.abs(ll[0] - np.cos(th + np.pi / 6))) < 1e-12


# ---------------------------------------------------------------------------------------------- (e)
@pytest.mark.parametrize("dphi_deg", [6.0, -12.0, 20.0, -30.0])
def test_phase_jump_angle(dphi_deg):
    t0 = 0.30137
    pre = clean_pre_event(theta0=2.3)
    ev = {"label": "phase_jump", "t0": t0, "dphi": np.radians(dphi_deg), "dv1": 0.0}
    v = synth_record(pre, ev).astype(float) * LSB_PU
    n0 = int(round(t0 * FS))

    def v1_angle(start):
        X = _dft50(v, start)
        return np.angle((X[0] + A * X[1] + A2 * X[2]) / 3)

    before = v1_angle(n0 - 2 * N_CYCLE - N_CYCLE)  # window ends >= 2 cycles before onset
    after = v1_angle(n0 + 2 * N_CYCLE)  # window starts >= 2 cycles after onset
    step = np.degrees(np.angle(np.exp(1j * (after - before))))
    assert abs(step - dphi_deg) <= 0.3


# ---------------------------------------------------------------------------------------------- (f)
def test_determinism(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c["splits"] = {"tiny": {"seed": 777, "per_class": 2, "nuisance": "shift"}}
    c["chunk_records"] = 5
    ds.generate_split("tiny", c, tmp_path / "a", progress=False)
    ds.generate_split("tiny", c, tmp_path / "b", progress=False)
    xa = np.load(tmp_path / "a" / "X.npy")
    xb = np.load(tmp_path / "b" / "X.npy")
    assert np.array_equal(xa, xb)
    assert (tmp_path / "a" / "manifest.csv").read_bytes() == (tmp_path / "b" / "manifest.csv").read_bytes()
    # a record synthesised alone equals the same record synthesised inside a chunk
    seeds, _ = ds.split_seeds(777, 14)
    import pandas as pd
    man = pd.read_csv(tmp_path / "a" / "manifest.csv")
    i = 9
    spec = ds.make_record_spec(i, man.loc[i, "label"], int(man.loc[i, "seed"]), c, "shift")
    fir = load_aa_fir()
    alone, _ = measure(synth_analog_ll(spec.pre, spec.ev, synth_time(len(fir["h"])))[None],
                       np.array([spec.pre.snr_db]), [ds.record_streams(spec.seed)[2]], fir["h"])
    assert int(man.loc[i, "seed"]) == seeds[i]
    assert np.array_equal(alone[0], xa[i])


def test_seed_disjointness(fast_flag):
    seen: set[int] = set()
    total = 0
    for s in ALL_SPLITS:
        m = load_manifest(s, fast=fast_flag, allow_test_manifest=True)
        seen |= set(m["seed"].astype("int64").tolist())
        total += len(m)
    assert len(seen) == total


def test_split_counts(cfg, fast_flag):
    for s in ALL_SPLITS:
        m = load_manifest(s, fast=fast_flag, allow_test_manifest=True)
        expected = ds.split_labels(cfg["splits"][s], tuple(cfg["classes"]), cfg["fast_fraction"] if fast_flag else None)
        assert m["label"].value_counts().to_dict() == {k: expected.count(k) for k in set(expected)}
        assert m["record_id"].tolist() == list(range(len(m)))


# ---------------------------------------------------------------------------------------------- (g)
def test_label_margins_analytic(cfg, fast_flag):
    """Regenerate every record's spec from (seed, label) and re-check the margins from phasors."""
    for s in ALL_SPLITS:
        m = load_manifest(s, fast=fast_flag, allow_test_manifest=True)
        nuis = cfg["splits"][s]["nuisance"]
        for rid, lab, seed, mm in zip(m["record_id"], m["label"], m["seed"], m["mm_min_ll_ev"]):
            spec = ds.make_record_spec(int(rid), lab, int(seed), cfg, nuis)
            assert margins_ok(lab, spec.pre, spec.ev, spec.metrics, cfg["margins"]), (s, rid, lab)
            assert abs(spec.metrics["mm_min_ll_ev"] - mm) < 1e-8


@pytest.mark.parametrize("split", ["train", "val"])
def test_label_margins_waveform(split, fast_flag):
    """One-cycle RMS re-check from the stored waveforms (train/val only: test sets stay sealed)."""
    X, _ = load_waveforms(split, fast=fast_flag)
    m = load_manifest(split, fast=fast_flag)
    for lab in ("sag_bal", "sag_unbal", "swell", "normal"):
        sub = m[m["label"] == lab]
        v = np.asarray(X[sub.index.to_numpy()], dtype=float) * LSB_PU
        rms = rms_one_cycle_sliding(v)
        for k, (_, r) in enumerate(sub.iterrows()):
            if lab == "normal":
                x = rms[k][:, N_CYCLE - 1:]
                assert x.min() > 0.90 and x.max() < 1.10, r["record_id"]
                continue
            a = int(round(r["ev_t0"] * FS)) + N_CYCLE
            b = min(int(round((r["ev_t0"] + r["ev_dur"]) * FS)), FS)
            w = rms[k][:, a:b]
            if lab.startswith("sag"):
                assert w.min() < 0.90, (r["record_id"], w.min())
            else:
                assert w.max() > 1.10, (r["record_id"], w.max())


# ---------------------------------------------------------------------------------------------- (h)
def test_pre_event_common_path(cfg):
    seeds, _ = ds.split_seeds(123, 5)
    for seed in seeds:
        pres = [ds.make_record_spec(0, lab, seed, cfg, "iid").pre for lab in CLASSES]
        assert all(p == pres[0] for p in pres)
        pres = [ds.make_record_spec(0, lab, seed, cfg, "shift").pre for lab in CLASSES]
        assert all(p == pres[0] for p in pres)


def test_rejection_redraws_only_event(cfg, monkeypatch):
    seed = ds.split_seeds(321, 1)[0][0]
    calls = {"pre": 0}
    real_draw_pre = ds.draw_pre_event

    def counting_draw_pre(*a, **k):
        calls["pre"] += 1
        return real_draw_pre(*a, **k)

    monkeypatch.setattr(ds, "draw_pre_event", counting_draw_pre)
    for lab in CLASSES:
        base = ds.make_record_spec(0, lab, seed, cfg, "iid")
        state = {"n": 0}

        def reject_first_three(label, pre, ev, m, mc):
            state["n"] += 1
            return state["n"] > 3 and margins_ok(label, pre, ev, m, mc)

        calls["pre"] = 0
        spec = ds.make_record_spec(0, lab, seed, cfg, "iid", margins_ok=reject_first_three)
        assert calls["pre"] == 1
        assert spec.pre == base.pre
        assert spec.n_reject >= 3
        assert spec.ev != base.ev  # event parameters were redrawn


# ---------------------------------------------------------------------------------------------- (i)
def test_storage_no_clipping(fast_flag):
    worst = 0
    for s in ALL_SPLITS:
        m = load_manifest(s, fast=fast_flag, allow_test_manifest=True)
        assert int(m["n_sat"].sum()) == 0, s
        worst = max(worst, int(m["max_abs_count"].max()))
    for s in ("train", "val"):
        X, meta = load_waveforms(s, fast=fast_flag)
        assert int(np.abs(np.asarray(X, dtype=np.int32)).max()) == meta["max_abs_count"]
        assert X.dtype == np.int16
    assert worst < 2048
    print(f"max |int16 count| over all splits = {worst} (12-bit limit 2047)")
