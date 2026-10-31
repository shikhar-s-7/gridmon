# gridmon — grid-event monitor for BESS PCS voltage records

> **Simulation-only; not validated on measured data; not for protection or control.**

gridmon detects and classifies grid events (balanced/unbalanced dips, swells, phase jumps,
frequency events, low-frequency oscillations, or normal) in triggered 1 s three-phase line-to-line
voltage records at the AC terminals of a battery energy storage system's power conversion system,
and compares a transparent, standards-based signal-processing baseline with gradient boosting on
the same physics features. The Python package is the reference for a later portable C99 float32
core (TI C2000 monitor-only module and a WebAssembly website). Full brief: docs/PROJECT_BRIEF.md.

## Quickstart
```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows (use .venv/bin/python elsewhere)
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python scripts/run_all.py --fast              # ~10 % of the data, minutes
.venv/Scripts/python scripts/run_all.py                     # full data
.venv/Scripts/python -m pytest
```
Outputs: data/ (git-ignored int16 records + manifests), reports/ (json + figures), golden/ (C-port
test vectors). `--fast` writes to data/fast and reports/fast.

## Status
See docs/PROGRESS.md and docs/HANDOFF.md. Milestones M1 (generator, datasets) and M2 (baseline,
DSP spec, golden vectors) are the current scope; ML, sealed evaluation, the C core, the C2000 port
and the website follow.

## Data
Records: 3 x 5000 int16 ADC counts (12-bit, +/- 2 pu, LSB = 4/4096 pu of nominal LL peak) at 5 kHz.
The test splits (test_iid, test_shift) are sealed: loaders refuse them without `allow_test=True`.

## Docs
- docs/PROJECT_BRIEF.md — specification (v0.2)
- docs/DECISIONS.md — every deviation and interpretation
- docs/DSP_SPEC.md, docs/FEATURES.md — port-exact DSP and feature definitions
- docs/VALIDATION.md — validation ladder L0-L6
- docs/ENVIRONMENT.md — Python and package versions
