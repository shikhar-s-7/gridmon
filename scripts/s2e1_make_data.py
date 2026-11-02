"""S2E1 data: generate the pre-registered splits (configs/s2e1.yaml) with the Stage 1 generator.

Usage: python scripts/s2e1_make_data.py [--fast] [--splits train_wide val_wide ...] [--no-features]
Writes data[/fast]/<split>/{X.npy, manifest.csv, meta.json} for train_wide, val_wide, test2_shift,
test2_novel, preview_wide, preview_novel; re-checks the label margins (incl. D-034) and the interharmonic
draws from the manifests (generator-side checks: no waveform, no method); computes features.csv for the
NON-sealed splits only (sealed features are computed once, by scripts/s2e1_evaluate.py after the freeze).
Summary: reports[/fast]/s2e1_data_summary.json. Must run after the "S2E1-prereg: ..." commit.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from gridmon.features import extract_features
from gridmon.generator.dataset import generate_split
from gridmon.io import data_root, load_baseline_config, load_waveforms, reports_root, write_json
from gridmon.s2e1 import NONSEALED, dataset_config, verify_interharmonic, verify_margins


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="~10%% of the records (data/fast, reports/fast)")
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--no-features", action="store_true")
    args = ap.parse_args(argv)
    cfg = dataset_config()
    splits = args.splits or list(cfg["splits"])
    root = data_root(args.fast)
    summary: dict = {"fast": args.fast, "splits": {}}
    t0 = time.perf_counter()
    import pandas as pd
    for name in splits:
        meta = generate_split(name, cfg, root / name, fast=args.fast, progress=False)
        man = pd.read_csv(root / name / "manifest.csv")  # generator-side check of what was just written
        ncfg = cfg["nuisance"][cfg["splits"][name]["nuisance"]]
        mg = verify_margins(man, cfg["margins"], int(cfg["max_redraws"]))
        ih = verify_interharmonic(man, ncfg.get("interharmonic"))
        snr = man["snr_db"].to_numpy()
        summary["splits"][name] = {
            **{k: meta[k] for k in ("n_records", "counts", "split_seed", "nuisance_config", "sealed", "config_hash",
                                    "git_commit", "runtime_s", "margin_rejection_rate", "max_abs_count",
                                    "n_records_saturated", "osc_band_fallback")},
            "snr_db_range": [float(snr.min()), float(snr.max())],
            "n_transient": int(man["tr_on"].sum()) if "tr_on" in man else 0,
            "margins": mg, "interharmonic": ih}
        print(f"{name:>13s}: N={meta['n_records']:5d} {meta['runtime_s']:6.1f} s  margins hold (except D-039 fallback): "
              f"{mg['all_hold_except_d039_fallback']}  osc fallback {meta['osc_band_fallback']}  transient "
              f"{summary['splits'][name]['n_transient']}" + (f"  ih on {ih['n_on']} (beat<25 Hz: {ih['n_beat_lt_25hz']}, "
                                                             f"exclusion {ih['harmonic_exclusion_holds']})"
                                                             if ih.get("configured") else ""))
    if not args.no_features:
        fcfg = load_baseline_config()["features"]
        for name in [s for s in splits if s in NONSEALED]:
            Xw, meta = load_waveforms(name, fast=args.fast)
            t1 = time.perf_counter()
            F = extract_features(Xw, fcfg, lsb=meta["lsb_pu"])  # waveforms only (rule A1.5)
            F.insert(0, "record_id", np.arange(len(F)))
            F.to_csv(root / name / "features.csv", index=False, float_format="%.9g")
            print(f"{name:>13s}: features {len(F)} records, {time.perf_counter() - t1:.1f} s")
    summary["total_runtime_s"] = round(time.perf_counter() - t0, 1)
    out = reports_root(args.fast) / "s2e1_data_summary.json"
    if out.exists() and args.splits:  # partial regeneration: keep the other splits' entries
        import json
        old = json.loads(out.read_text(encoding="utf-8"))
        old["splits"].update(summary["splits"])
        summary["splits"] = old["splits"]
    write_json(out, summary)
    print(f"S2E1 data done in {summary['total_runtime_s']} s -> {out}")
    ok = all(v["margins"]["all_hold_except_d039_fallback"] for v in summary["splits"].values())
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
