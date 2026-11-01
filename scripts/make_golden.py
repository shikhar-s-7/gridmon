"""M2: golden vectors for the C port (brief B7.5): 21 val records (3 per class) + 6 unit signals.

Usage: python scripts/make_golden.py [--fast]
Writes golden/*.npz, golden/index.json and golden/README.md (fast: data/fast/golden, not committed).
Uses val WAVEFORMS only (labels are stored as metadata for reference); never the sealed test sets.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from gridmon import CLASSES
from gridmon.features import AUX_NAMES, FEATURE_NAMES, FEATURE_SPEC, REL_TOL, TOL_GATES, tolerance_check
from gridmon.golden import TRACE_KEYS, float32_errors, reference_outputs, unit_signals, write_npz
from gridmon.io import ROOT, data_root, load_baseline_config, load_labels, load_waveforms, write_json

ARRAYS = [  # name, shape, dtype, unit, description, tolerance floor
    ("x", "[3, 5000]", "int16", "count", "input ADC counts, channels ab, bc, ca (v = 4/4096 * x pu)", None),
    ("urms", "[3, 99]", "f8", "pu", "Urms(1/2), value k over samples [50k, 50k+99]", 1e-5),
    ("v1", "[1000]", "c16", "pu", "positive-sequence phasor at n = 5m+4 (NaN for m < 19)", 1e-5),
    ("v2", "[1000]", "c16", "pu", "negative-sequence phasor (NaN for m < 19)", 1e-5),
    ("f_dft", "[1000]", "f8", "Hz", "DFT frequency after one-cycle MA (NaN for m < 58)", 1e-4),
    ("rocof", "[1000]", "f8", "Hz/s", "LS slope over 100 phasor samples, unmasked (NaN for m < 157)", 1e-3),
    ("J", "[1000]", "f8", "rad", "jump metric (NaN for m < 59); compare wrapped", 1e-4),
    ("f_pll", "[1000]", "f8", "Hz", "DSOGI-PLL frequency at n = 5m+4 (all m)", 3e-3),
    ("fmask", "[1000]", "bool", "-", "f_dft samples excluded around a detected jump (exact unless j_max_deg is within 1e-3 deg of the 3 deg detection level)", None),
    ("rmask", "[1000]", "bool", "-", "ROCOF samples excluded (window touches fmask)", None),
    ("e_mag", "[100]", "f8", "1", "abs(V1) block means / v1_ref (NaN for p < 2)", 1e-5),
    ("e_ang", "[100]", "f8", "rad", "de-rotated unwrapped angle block means (NaN for p < 2)", 1e-4),
    ("r_mag", "[100]", "f8", "1", "e_mag minus centred running median", 1e-5),
    ("r_ang", "[100]", "f8", "rad", "e_ang minus centred running median", 1e-4),
    ("S_mag", "[5, 28]", "f8", "%", "envelope spectra of 100*r_mag, windows q x f = 3..30 Hz", 1e-3),
    ("S_ang", "[5, 28]", "f8", "deg", "envelope spectra of degrees(r_ang)", 1e-3),
    ("thd", "[10]", "f8", "1", "THD per 500-sample window (mean over channels)", 1e-6),
    ("features", f"[{len(FEATURE_NAMES)}]", "f8", "mixed", "feature vector in feature_names order (docs/FEATURES.md)", None),
    ("aux", f"[{len(AUX_NAMES)}]", "f8", "s / 1", "auxiliary outputs in aux_names order (onset times, spectral gaps)", None),
    ("rule_label", "[]", "str", "-", "rule label with DEFAULT thresholds", None),
    ("rule_reason", "[]", "int", "-", "reason code 0..7 (baseline_rules.REASONS)", None),
    ("rule_reason_name", "[]", "str", "-", "reason name", None),
    ("rule_onset", "[]", "f8", "s", "onset estimate (first detection time; NaN for normal)", None),
    ("feature_names", "[37]", "str", "-", "names of the feature vector", None),
    ("aux_names", f"[{len(AUX_NAMES)}]", "str", "-", "names of the aux vector", None),
    ("meta_json", "[]", "str", "-", "source, true label (val records), description, thresholds", None),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args(argv)
    bcfg = load_baseline_config()
    fcfg, th = bcfg["features"], dict(bcfg["rules"])
    out_dir = (data_root(True) / "golden") if args.fast else (ROOT / "golden")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.npz"):
        old.unlink()

    X, meta = load_waveforms("val", fast=args.fast)
    y = load_labels("val", fast=args.fast)
    ids = [int(i) for c in CLASSES for i in np.where(y == c)[0][:3]]
    items = [(f"val_{y[i]}_{i:04d}", {"source": f"val record {i}", "true_label": str(y[i])}, np.asarray(X[i])) for i in ids]
    for name, desc, x in unit_signals():
        items.append((name, {"source": "unit signal (noise-free, nuisances off, quantised)", "description": desc}, x))
    xs = np.stack([it[2] for it in items])
    ref = reference_outputs(xs, fcfg, th, np.float64)
    f32 = reference_outputs(xs, fcfg, th, np.float32)

    index = []
    for k, ((name, info, _), d) in enumerate(zip(items, ref)):
        fname = f"{k + 1:02d}_{name}.npz"
        write_npz(out_dir / fname, d, {**info, "thresholds": th, "lsb_pu": meta["lsb_pu"], "fs_hz": 5000})
        index.append({"file": fname, **info, "rule_label": d["rule_label"], "rule_reason": d["rule_reason"]})
    write_json(out_dir / "index.json", index)

    errs = float32_errors(ref, f32)
    Fr = pd.DataFrame([dict(zip(FEATURE_NAMES + AUX_NAMES, np.r_[d["features"], d["aux"]])) for d in ref])
    Ft = pd.DataFrame([dict(zip(FEATURE_NAMES + AUX_NAMES, np.r_[d["features"], d["aux"]])) for d in f32])
    fchk = tolerance_check(Fr, Ft)
    same_rules = sum(a["rule_label"] == b["rule_label"] for a, b in zip(ref, f32))
    size = sum(p.stat().st_size for p in out_dir.glob("*"))
    tol_rows = []
    for name, shape, dt, unit, desc, floor in ARRAYS:
        if floor is None:
            tol = "exact" if name in ("x", "fmask", "rmask", "rule_label", "rule_reason", "rule_reason_name") else "-"
            if name == "features":
                tol = "per feature: docs/FEATURES.md"
            obs = ""
        else:
            obs = f"{errs[name]:.2e}"
            tol = f"{max(10 * errs[name], floor):.1e}"
        tol_rows.append(f"| `{name}` | {shape} | {dt} | {unit} | {desc} | {tol} | {obs} |")
    readme = f"""# Golden vectors (C-port reference, brief B7.5)

Generated by `scripts/make_golden.py` from the float64 Python reference (gridmon v0.2).
{len(index)} files: 21 val records (first 3 per class in val order) + 6 noise-free unit signals.
Total size {size / 1e6:.2f} MB. Rule outputs use the DEFAULT thresholds of configs/baseline.yaml
(stored in each file's meta_json). Simulation-only data.

## Files
| file | source | rule label (reason) |
|---|---|---|
""" + "\n".join(f"| {e['file']} | {e['source']}{(' — ' + e['true_label']) if 'true_label' in e else (' — ' + e.get('description', ''))} | {e['rule_label']} ({e['rule_reason']}) |" for e in index) + f"""

## Arrays in each .npz (`numpy.load(path)`; complex stored as complex128)
Indices: phasor index m = 0..999 at sample n = 5m+4 (valid m >= 19); envelope index p = 0..99
(block m = 10p..10p+9, valid p >= 2); spectra windows q = 0..4 start at p = 2 + 10q; THD windows w = 0..9.
NaN marks samples the C core does not compute (compare only finite entries).

Tolerance = allowed |C float32 - golden| (complex: modulus of the difference; J: wrapped angle
difference), set to max(10 x the float32-numpy error observed on these files, a floor).

| array | shape | dtype | unit | content | tolerance | observed float32 error |
|---|---|---|---|---|---|---|
""" + "\n".join(tol_rows) + f"""

Features: tolerances and comparison gates per feature are listed in docs/FEATURES.md
(argmax-type features j_sign, env_*_freq, env_*_prom are compared only when not a near-tie and
above the envelope noise floor). On these files the float32-numpy reference keeps
{sum(v['n_over_tol'] == 0 for v in fchk.values())}/{len(fchk)} features within tolerance and
{same_rules}/{len(ref)} rule labels identical.

## C-equivalence criteria (M6)
- traces: within the tolerances above on every finite entry;
- features: within docs/FEATURES.md tolerances (with gates);
- rule label and reason code: identical on all golden files (and >= 99.5 % on full splits).
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"golden: {len(index)} files, {size / 1e6:.2f} MB -> {out_dir}; float32 rule agreement {same_rules}/{len(ref)}; "
          f"features over tol: {[k for k, v in fchk.items() if v['n_over_tol']]}")
    return 0 if size <= 10e6 else 1


if __name__ == "__main__":
    sys.exit(main())
