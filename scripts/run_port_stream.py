"""M7: host simulation of the C2000 port layer on a continuous 50 kHz stream (integration test).

Stream: 120 s, 14 events (2 per class, iid parameters with label margins, seed 7007), >= 4 s gaps with
hard negatives, iid sensor errors + noise at 50 kHz + 12-bit ADC (src/gridmon/generator/stream.py). Fed
sample by sample through gm_port_push (calibration trim, FIR, decimation); background every 1 ms,
blocked 50 ms after each classified record. Also: LN-input run, decimator vs float64 reference for
every f_in, ISR MACs per input sample. Writes reports[/fast]/m7_port_stream.json and
reports[/fast]/figures/m7_port_stream.png. Not an accuracy result (n = 14).

Usage: python scripts/run_port_stream.py [--fast] [--seed 7007]
"""

from __future__ import annotations

import argparse
import sys
import time

from gridmon.c_build import build_all
from gridmon.io import ROOT, reports_root, write_json
from gridmon.port_stream import decimator_check, figure, fir_table, isr_cost, public, run_stream


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="shorter stream (60 s, 1 event per class), reports/fast")
    ap.add_argument("--seed", type=int, default=7007)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    build_all(sanitize=False)
    dur = 60.0 if args.fast else 120.0
    res = run_stream(seed=args.seed, duration_s=dur) if not args.fast else _fast(args.seed)
    s = res["summary"]
    print(f"stream LL: {s['events_triggered']}/{s['events']} events captured, rules {s['rule_label_correct']}/{s['events']}, "
          f"GBM {s['gbm_agrees_with_truth']}/{s['events']}, normal episodes triggered {s['normal_events_triggered']}/"
          f"{s['normal_events']}, false {s['false_triggers_outside_events']}, secondary {s['secondary_captures']}, "
          f"overruns {s['overruns']}; onset in record {s['onset_in_record_s_min']:.3f}..{s['onset_in_record_s_max']:.3f} s")
    ln = run_stream(seed=args.seed, duration_s=dur, ln=True, spec=res["_spec"])
    same = [a.get("rule_label") == b.get("rule_label") and a["triggered"] == b["triggered"]
            for a, b in zip(res["per_event"], ln["per_event"])]
    print(f"stream LN input: {ln['summary']['events_triggered']}/{ln['summary']['events']} captured, "
          f"same outcome as LL for {sum(same)}/{len(same)} events")
    dec = [decimator_check(p["f_in_hz"]) for p in fir_table()]
    cost = isr_cost()
    for c, d in zip(cost, dec):
        print(f"f_in {c['f_in_hz'] // 1000:3d} kHz: L {c['L']:3d}, D {c['D']:2d}, K {c['K']}: {c['total_macs_per_input_sample']} MAC/input "
              f"sample ({c['macs_per_s'] / 1e6:.2f} MMAC/s), stopband {c['stopband_atten_db']:.1f} dB; decimator vs float64 "
              f"max {d['max_abs_diff_lsb']:.2f} LSB")
    rep = {"main_LL_calibrated": public(res),
           "LN_input": {"summary": ln["summary"], "same_outcome_as_LL": int(sum(same)), "n": len(same)},
           "isr_cost": cost, "decimator_check": dec, "runtime_s": round(time.perf_counter() - t0, 1)}
    out = reports_root(args.fast) / "m7_port_stream.json"
    write_json(out, rep)
    fig = figure(res, reports_root(args.fast) / "figures" / "m7_port_stream.png")
    print(f"-> {out.relative_to(ROOT)}, {fig.relative_to(ROOT)} ({rep['runtime_s']} s)")
    return 0


def _fast(seed: int):
    from gridmon.generator.stream import make_stream
    from gridmon.io import load_dataset_config
    spec = make_stream(load_dataset_config(), seed=seed, duration_s=60.0, per_class=1)
    return run_stream(spec=spec, duration_s=60.0)


if __name__ == "__main__":
    sys.exit(main())
