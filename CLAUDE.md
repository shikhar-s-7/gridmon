# gridmon — notes for Claude Code

Grid-event monitor for BESS PCS voltage records: synthetic generator (M1), conventional DSP
baseline + rules (M2), later GBM (M3), sealed evaluation (M4), C99 core (M6), C2000 port (M7),
website (M8). Python is the REFERENCE for a float32 C port. Simulation-only.

## Commands (Windows paths; use .venv/bin/python on macOS/Linux)
- setup: `py -3.12 -m venv .venv`; `.venv/Scripts/python -m pip install -r requirements.txt`; `... -m pip install -e .`
- tests: `.venv/Scripts/python -m pytest` (uses data/ if present, else data/fast; GRIDMON_TEST_FAST=1 forces fast)
- everything: `.venv/Scripts/python scripts/run_all.py [--fast]`
- stages: `scripts/make_dataset.py [--fast] [--splits ...]`, `scripts/run_baseline.py [--fast]`, `scripts/make_golden.py`

## Rules (PROMPT A1)
1. Spec = docs/PROJECT_BRIEF.md. Every deviation/interpretation -> docs/DECISIONS.md (ID, date, what, why, alternatives).
2. Units: instantaneous pu of nominal LL peak; RMS pu = sqrt(2*mean(v^2)); radians in code, degrees in reports.
3. Reproducible: numpy Generators from SeedSequence only; no global RNG; config hash + version in meta.json.
4. Sealed test sets: never load/plot/evaluate test_iid/test_shift waveforms before M4 (loader raises unless allow_test=True; only M4 and M6 may pass it). Tests may read sealed manifests.
5. No leakage: features/rules/models see only the waveform array, never manifest fields.
6. Vectorise over records; recursions (PLL) loop over time with records as arrays; chunk ~250-500 records; every script has --fast.
7. Cross-platform Python scripts, pathlib.
8. Small typed modules; docstrings state equation + units; pytest tests.
9. Commit at the end of each milestone; never commit data/ or .venv/; golden/ is committed.
10. Keep docs/PROGRESS.md and this file current.
11. Keep tool output short (summaries, no array dumps).
12. Port-exact DSP: every step specified in docs/DSP_SPEC.md; float32, no malloc, <= 32 KB working state, no look-ahead beyond the record; direct DFTs at needed bins; twiddles indexed by (n mod N).

## Layout
- src/gridmon/generator: phasors (Bollen), nuisance (pre-event draw), events (models, margins), synth (20 kHz + chain), dataset (splits)
- src/gridmon/dsp: rms, phasor, freq, pll, envelope, thd; features.py, baseline_rules.py, metrics.py, golden.py, plots.py
- configs/: dataset.yaml, baseline.yaml, aa_fir.json (generated)
- docs/: PROJECT_BRIEF, DECISIONS, PROGRESS, DSP_SPEC, FEATURES, VALIDATION, ENVIRONMENT, HANDOFF (+ handoffs/)
