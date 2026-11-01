"""M1: design the anti-alias FIR and generate the dataset splits.

Usage: python scripts/make_dataset.py [--fast] [--splits train val ...] [--no-figures]
Writes data[/fast]/<split>/{X.npy, manifest.csv, meta.json}, configs/aa_fir.json, and M1 figures.
"""

from __future__ import annotations

import argparse
import sys
import time

from gridmon.generator.dataset import generate_split
from gridmon.generator.synth import AA_FIR_PATH, design_aa_fir, save_aa_fir_if_changed
from gridmon.io import data_root, load_dataset_config, reports_root, write_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="~10%% of the records (data/fast, reports/fast)")
    ap.add_argument("--splits", nargs="*", default=None, help="subset of splits (default: all)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_dataset_config()
    fir = design_aa_fir(cfg["aa_fir"], fs_in=cfg["record"]["fs_synth_hz"])
    written = save_aa_fir_if_changed(fir, AA_FIR_PATH)  # keep committed bits if equal to 1e-12
    print(f"AA FIR: {fir['numtaps']} taps, stopband {fir['stopband_atten_db']:.1f} dB @ >= {fir['stopband_hz']:.0f} Hz, "
          f"passband ripple {fir['passband_ripple_db'][0]:+.3f}/{fir['passband_ripple_db'][1]:+.3f} dB, "
          f"group delay {fir['group_delay_samples_in']} samples @20 kHz"
          f" ({'written' if written else 'unchanged, committed coefficients kept'})")

    splits = args.splits or list(cfg["splits"])
    root = data_root(args.fast)
    summary = {}
    t0 = time.perf_counter()
    for name in splits:
        meta = generate_split(name, cfg, root / name, fast=args.fast)
        rr = ", ".join(f"{k}={v:.2f}" for k, v in meta["margin_rejection_rate"].items())
        print(f"{name:>13s}: N={meta['n_records']:5d}  {meta['runtime_s']:6.1f} s  max|count|={meta['max_abs_count']}"
              f"  saturated={meta['n_records_saturated']}  reject[{rr}]")
        summary[name] = {k: meta[k] for k in ("n_records", "counts", "runtime_s", "margin_rejection_rate",
                                              "max_abs_count", "n_records_saturated", "config_hash")}
    total = time.perf_counter() - t0
    summary["total_runtime_s"] = round(total, 1)
    rep = reports_root(args.fast)
    write_json(rep / "m1_dataset_summary.json", summary)
    print(f"dataset total: {total:.1f} s")

    if not args.no_figures:
        from gridmon.plots import m1_figures
        m1_figures(fast=args.fast)
        print(f"M1 figures -> {rep / 'figures'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
