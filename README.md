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
Stages: make_dataset -> run_baseline -> make_golden -> run_gbm -> evaluate_test (sealed test sets; refuses to
run unless the methods match configs/frozen_methods.yaml).
Outputs: data/ (git-ignored int16 records + manifests), reports/ (json, RESULTS.md, figures), golden/ (C-port
test vectors), models/ (exported GBM models), c/generated/ (C model tables). `--fast` writes to data/fast and reports/fast.

## Status
Stage 1 milestones M1-M5 are done: generator and datasets (M1, with the oscillation label margin of R1),
standards-based rules baseline, DSP spec and golden vectors (M2), gradient boosting with a 64 KB embedded
budget and a C-ready export (M3), and the one-shot sealed evaluation (M4/M5, reports/RESULTS.md).
Headline [SIM]: on iid test data the rules (macro-F1 0.998) and the 14 KB GBM (1.000) are not significantly
different; under distribution shift the rules (0.990) are significantly better than the GBM (0.865).
Next: portable C core and SIL (M6), C2000 port (M7), website (M8). See docs/PROGRESS.md and docs/HANDOFF.md.

## Data
Records: 3 x 5000 int16 ADC counts (12-bit, +/- 2 pu, LSB = 4/4096 pu of nominal LL peak) at 5 kHz.
The test splits (test_iid, test_shift) are sealed: loaders refuse them without `allow_test=True`.

## Docs
- docs/PROJECT_BRIEF.md — specification (v0.2)
- docs/DECISIONS.md — every deviation and interpretation
- docs/DSP_SPEC.md, docs/FEATURES.md — port-exact DSP and feature definitions
- docs/MODEL_FORMAT.md — exported GBM format, size accounting, float32 walker
- reports/RESULTS.md — sealed test results, findings and limitations
- docs/VALIDATION.md — validation ladder L0-L6
- docs/ENVIRONMENT.md — Python and package versions
