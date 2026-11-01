"""M7 port layer (firmware/c2000/gm_port.c) on the host: stream-generator option, FIR/decimator tables,
calibration trim, LN->LL, ring-buffer overrun handling and the 120 s continuous-stream integration test
(seed 7007; not an accuracy result). Skipped when no C compiler is available."""

from __future__ import annotations

import numpy as np
import pytest

from gridmon.c_build import BuildError, find_compiler
from gridmon.io import load_dataset_config


@pytest.fixture(scope="module")
def cc():
    try:
        return find_compiler(install_zig=False)
    except BuildError as e:
        pytest.skip(str(e))


def _three_phase(f_in: int, seconds: float, amp=1.0, f=50.0, dip=None) -> np.ndarray:
    """Clean LL voltages [n, 3] (pu); dip = (t0, t1, depth) scales all channels inside [t0, t1)."""
    t = np.arange(int(seconds * f_in)) / f_in
    a = np.full(t.shape, amp)
    if dip:
        a = np.where((t >= dip[0]) & (t < dip[1]), amp * dip[2], a)
    return np.stack([a * np.cos(2 * np.pi * f * t - k * 2 * np.pi / 3) for k in range(3)], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------------------------- generator option
def test_stream_option_leaves_record_generator_unchanged(fast_flag):
    """The stream module is an option only: regenerated VAL records and the golden unit signals stay bit-identical."""
    from gridmon.generator.dataset import make_record_spec, record_streams, split_seeds
    from gridmon.generator.synth import load_aa_fir, measure, synth_analog_ll, synth_time
    from gridmon.golden import unit_signals
    from gridmon.io import load_split
    cfg = load_dataset_config()
    X, man, _ = load_split("val", fast=fast_flag)
    fir = load_aa_fir()
    t = synth_time(len(fir["h"]))
    for i in range(5):
        seed = int(man.loc[i, "seed"])
        spec = make_record_spec(i, man.loc[i, "label"], seed, cfg, "iid")
        x, _ = measure(synth_analog_ll(spec.pre, spec.ev, t)[None], np.array([spec.pre.snr_db]),
                       [record_streams(seed)[2]], fir["h"])
        assert np.array_equal(x[0], X[i])
    from gridmon.io import ROOT
    for (name, _, x), p in zip(unit_signals(), sorted((ROOT / "golden").glob("2[2-7]_u*.npz"))):
        assert np.array_equal(x, np.load(p)["x"]), name


def test_stream_spec_reproducible_and_valid():
    from gridmon import CLASSES
    from gridmon.generator.stream import make_stream, synth_stream
    cfg = load_dataset_config()
    a, b = make_stream(cfg, seed=7007), make_stream(cfg, seed=7007)
    assert [(e.label, e.S, e.T) for e in a.episodes] == [(e.label, e.S, e.T) for e in b.episodes]
    assert sorted(e.label for e in a.events) == sorted(c for c in CLASSES for _ in range(2))
    ev = a.events
    assert all(ev[i + 1].S - ev[i].E >= 4.0 for i in range(len(ev) - 1))           # >= 4 s of normal operation
    assert sum(e.kind == "gap" for e in a.episodes) == len(ev)                       # hard negatives in every gap
    assert np.array_equal(next(synth_stream(a, chunk_s=0.1)), next(synth_stream(b, chunk_s=0.1)))


# ---------------------------------------------------------------------------------------------- FIR / decimator
def test_fir_tables_meet_spec(cc):
    from gridmon.port_stream import fir_table, isr_cost
    rates = {p["f_in_hz"] for p in fir_table()}
    assert rates == {20000, 40000, 50000, 100000}
    for c in isr_cost():
        assert c["D"] * 5000 == c["f_in_hz"]
        assert abs(c["gain_50hz_float32_taps"] - 1.0) < 1e-6          # gain 1 at 50 Hz
        assert c["stopband_atten_db"] >= 40.0                        # >= 40 dB at >= 2.5 kHz
        assert abs(c["gain_1khz_db"]) < 0.1                          # passband >= 1 kHz
        assert c["total_macs_per_input_sample"] <= 45


@pytest.mark.parametrize("f_in", [20000, 40000, 50000, 100000])
def test_decimator_matches_float64_reference(cc, f_in):
    from gridmon.port_stream import decimator_check
    r = decimator_check(f_in, seconds=0.2)
    assert r["n_outputs"] > 900 and r["max_abs_diff_lsb"] <= 1.0


def test_calibration_trim_and_ln_input(cc):
    """Trim undoes a channel gain/offset error (D-052); LN inputs give the same LL samples."""
    from gridmon.port_sim import PortSim
    ll = _three_phase(50000, 0.2)
    ref = PortSim(50000).decimate(ll).astype(int)
    bad = ll.copy()
    bad[:, 0] = 1.02 * bad[:, 0] + 0.01
    trimmed = PortSim(50000, gain=(1 / 1.02, 1, 1), offset=(0.01, 0, 0)).decimate(bad).astype(int)
    assert np.abs(trimmed - ref).max() <= 2
    assert np.abs(PortSim(50000).decimate(bad).astype(int) - ref).max() > 100
    t = np.arange(ll.shape[0]) / 50000
    # LN phase voltages (pu of nominal LN peak) whose differences / sqrt3 are the LL voltages above:
    # (cos(x - 30 deg) - cos(x - 150 deg)) / sqrt3 = cos(x)
    ln = np.stack([np.cos(2 * np.pi * 50 * t - np.pi / 6 - k * 2 * np.pi / 3) for k in range(3)], axis=1).astype(np.float32)
    y_ln = PortSim(50000, input_ln=True).decimate(ln).astype(int)
    assert np.abs(y_ln - ref).max() <= 2


def test_ring_overrun_never_blocks_and_resyncs(cc):
    from gridmon.port_sim import PortSim
    x = _three_phase(50000, 2.0)
    sim = PortSim(50000, bg_every=10 ** 9)                  # background never runs
    sim.feed(x)
    n_out = (x.shape[0] + 7) // 10 - 12                      # emitted outputs (pad 7, K - 1 = 12 pre-roll discarded)
    assert sim.head == 8000 and sim.overruns == n_out - 8000   # ring holds 1.6 s; later samples dropped, counted
    sim2 = PortSim(50000, bg_every=50, proc_delay=3 * 50000)   # background blocked 3 s after the first record
    tel = sim2.feed(_three_phase(50000, 8.0, dip=(2.0, 2.2, 0.5)))
    tel += sim2.feed(_three_phase(50000, 3.0, dip=(1.0, 1.2, 0.5)))
    assert sim2.overruns > 0
    assert tel[0]["rule_label_name"] == "sag_bal" and tel[-1]["resyncs"] >= 1


# ---------------------------------------------------------------------------------------------- integration
@pytest.fixture(scope="module")
def stream_result(cc):
    from gridmon.port_stream import run_stream
    return run_stream(seed=7007, duration_s=120.0)


def test_stream_all_events_captured_without_false_triggers(stream_result):
    s = stream_result["summary"]
    assert s["events"] == 12 and s["events_triggered"] == 12
    assert s["normal_events_triggered"] == 0 and s["false_triggers_outside_events"] == 0
    assert s["overruns"] == 0 and s["resyncs"] == 0 and s["clipped"] == 0


def test_stream_record_placement(stream_result):
    """The true onset lands where the classifier expects it (trained with onsets in 0.20-0.40 s, first 0.15 s pre-event)."""
    for r in stream_result["per_event"]:
        if r["triggered"]:
            assert 0.15 <= r["onset_in_record_s"] <= 0.40, r


def test_stream_labels(stream_result):
    """Integration check (n = 12, not an accuracy claim): the captured records are classified like records."""
    s = stream_result["summary"]
    assert s["rule_label_correct"] >= s["events"] - 1
    assert all(r["gbm_label"] is not None for r in stream_result["per_event"] if r["triggered"])
