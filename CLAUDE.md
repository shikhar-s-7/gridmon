# gridmon — notes for Claude Code

Grid-event monitor for BESS PCS voltage records: synthetic generator (M1), conventional DSP
baseline + rules (M2), GBM with 64 KB C export (M3), sealed evaluation (M4, done: reports/RESULTS.md),
C99 core (M6), C2000 port (M7), website (M8) — all done. Python is the REFERENCE for a float32 C port. Simulation-only.
Status (2026-11-02, session 5): Stage 1 COMPLETE (M1-M8; L0-L2 done, D-076). Stage 2: S2E1 done (docs/S2E1_PREREG.md,
reports/S2E1_RESULTS.md; STAGE2_PLAN item 3). Next: docs/STAGE2_PLAN.md.
Methods FROZEN (configs/frozen_methods.yaml, M4-freeze commit b3afdec; older docs cite 6240f60, D-083); any new method needs NEW sealed splits.

## Commands (Windows paths; use .venv/bin/python on macOS/Linux)
- setup: `py -3.12 -m venv .venv`; `.venv/Scripts/python -m pip install -r requirements.txt`; `... -m pip install -e .`
- tests: `.venv/Scripts/python -m pytest` (uses data/ if present, else data/fast; GRIDMON_TEST_FAST=1 forces fast)
- everything: `.venv/Scripts/python scripts/run_all.py [--fast]`
- stages: `scripts/make_dataset.py [--fast] [--splits ...]`, `scripts/run_baseline.py [--fast]`, `scripts/make_golden.py`,
  `scripts/run_gbm.py [--fast]` (full mode keeps frozen models unless --force), `scripts/freeze_methods.py`,
  `scripts/evaluate_test.py [--fast] [--dry-run] [--render-only]` (only script with allow_test; refuses unless frozen.verify passes)
- C core (M6): `scripts/gen_c_tables.py [--check]` (D-054 tables), `scripts/build_c.py [--force]` (build/: SIL lib, gm_cli,
  gm_cli_asan; cc/gcc/clang or `python -m ziglang cc`), `scripts/run_sil.py [--fast]` (golden + splits -> reports/m6_sil.json;
  the only other allow_test caller: equivalence only, no labels)
- C2000 port (M7): `scripts/run_port_stream.py [--fast]` (120 s 50 kHz stream through firmware/c2000/gm_port.c ->
  reports/m7_port_stream.json + figure), `scripts/c2000_compile_check.py --cgt-root <TI CGT>` (L2, local; `--dry-run`),
  `scripts/make_pil_records.py [--check]` (LaunchPad PIL template records)
- Website (M8): `scripts/build_site.py [--check] [--skip-export]` (export_web -> web/src/data, zig wasm32-wasi -> web/dist,
  pages with {{path|fmt}} tokens; --check = WASM vs native C in Node), preview `python -m http.server -d web/dist 8000`,
  headless check `node web/tools/check_site.mjs http://localhost:8000 reports/m8_site_check.json reports/figures`;
  `scripts/update_readme.py [--check]` (README results block); deploy: docs/DEPLOY.md
- Stage 2 S2E1: `scripts/s2e1_make_data.py [--fast] [--splits ...]` (configs/s2e1.yaml splits; margins/interharmonic checks ->
  reports/s2e1_data_summary.json; features for non-sealed splits), `scripts/s2e1_train.py [--fast]` (GW/GP on train_wide/val_wide;
  refuses once frozen unless --force), `scripts/s2e1_freeze.py` (configs/s2e1_frozen.yaml), `scripts/s2e1_evaluate.py [--fast]
  [--dry-run] [--render-only]` (one-shot on test2_*; refuses unless gridmon.s2e1.verify passes)
- Linux cloud VM: `python3.12 -m venv .venv` then `.venv/bin/python ...` (all pins install unchanged); optional `pip install ziglang` (-m32/msp430 checks)

## Rules (PROMPT A1)
1. Spec = docs/PROJECT_BRIEF.md. Every deviation/interpretation -> docs/DECISIONS.md (ID, date, what, why, alternatives).
2. Units: instantaneous pu of nominal LL peak; RMS pu = sqrt(2*mean(v^2)); radians in code, degrees in reports.
3. Reproducible: numpy Generators from SeedSequence only; no global RNG; config hash + version in meta.json.
4. Sealed test sets (test_iid, test_shift, test2_shift, test2_novel): loader raises unless allow_test=True; only scripts/evaluate_test.py (M4), the M6 SIL (scripts/run_sil.py, equivalence), the M8 export (scripts/export_web.py, display only, D-077) and scripts/s2e1_evaluate.py (test2_* only, after the S2E1 freeze) may pass it (tested). M4 and S2E1 have each been run once: test_shift, test2_shift and test2_novel are no longer blind, so any new method needs NEW sealed splits (new seeds; used seeds: configs/dataset.yaml, configs/s2e1.yaml). Tests may read sealed manifests.
4b. Frozen methods: changing a method file (dsp/*, features, rules, units, ml_gbm, export_gbm, configs/baseline.yaml, configs/ml.yaml, c/generated/*) or a model breaks the freeze; report frozen results first, label post-fix results.
5. No leakage: features/rules/models see only the waveform array, never manifest fields.
6. Vectorise over records; recursions (PLL) loop over time with records as arrays; chunk ~250-500 records; every script has --fast.
7. Cross-platform Python scripts, pathlib.
8. Small typed modules; docstrings state equation + units; pytest tests.
9. Commit at the end of each milestone (commit code BEFORE generating data/models so meta records a clean hash); never commit data/ or .venv/; golden/, models/, c/generated/, reports/ are committed.
10. Keep docs/PROGRESS.md and this file current.
11. Keep tool output short (summaries, no array dumps).
12. Port-exact DSP: every step specified in docs/DSP_SPEC.md; float32, no malloc, <= 32 KB working state, no look-ahead beyond the record; direct DFTs at needed bins; twiddles indexed by (n mod N).

## Layout
- src/gridmon/generator: phasors (Bollen), nuisance (pre-event draw), events (models, margins), synth (20 kHz + chain), dataset (splits)
- src/gridmon/dsp: rms, phasor, freq, pll, envelope, thd; features.py, baseline_rules.py, metrics.py, golden.py, plots.py
- src/gridmon (M3/M4): ml_gbm (train/select; NaN -> 0 input, D-040), export_gbm (JSON + C tables + float32 walker),
  plots_ml, frozen (freeze verify), evaluation, stats (bootstrap, McNemar), results_report
- configs/: dataset.yaml, baseline.yaml, ml.yaml, frozen_methods.yaml, aa_fir.json (generated; kept if equal, D-038)
- models/: gbm_embedded.json (C model, seed 0) + seed1/2 JSON, sklearn pickles; c/generated/gm_model_gbm.{c,h}
- c/: include/gridmon.h (API), src/gm_*.c (core), tables/ (GENERATED), generated/ (frozen GBM tables), host/ (CLI, ctypes helpers);
  src/gridmon c_build, c_sil (ctypes), c_portability (static checks)
- firmware/c2000/: gm_port.{h,c} (portable port layer: trim, LN->LL, polyphase FIR, ring, trigger, capture, telemetry),
  gm_port_fir.{h,c} (GENERATED taps 20/40/50/100 kHz), gm_port_c2000.c (thin adapter), host/ (sim driver), pil_launchpad/ (UNTESTED);
  src/gridmon port_sim (ctypes), port_stream (evaluation, figure), generator/stream.py (continuous-stream option)
- Stage 2 S2E1: configs/s2e1.yaml (prereg: splits, arms, physics subset, plan), configs/s2e1_frozen.yaml, src/gridmon s2e1
  (config merge, data checks, arms/hybrids, freeze verify, stats), s2e1_plots, s2e1_report; generator option: interharmonic
  (nuisance.py/synth.py, D-080); models/s2e1/ (GW, GP JSON), c/generated_s2e1/{gw,gp}/ (C tables, not used by the core build)
- web/: src/ (pages/ templates, css, js, data/ GENERATED by export_web, examples/), wasm/gm_wasm.c (glue), tools/ (node checks),
  dist/ (built site, committed; served by GitHub Pages)
- docs/: PROJECT_BRIEF, DECISIONS (D-001..D-08x), S2E1_PREREG, C_CORE, C2000_INTEGRATION, C2000_BUILD, DEPLOY, STAGE2_PLAN, PROJECT_SUMMARY, REPRODUCIBILITY, PROGRESS, DSP_SPEC, FEATURES, MODEL_FORMAT, VALIDATION, ENVIRONMENT,
  HANDOFF (+ handoffs/); reports/RESULTS.md + results.json (+ results_findings.md, hand-written); reports/S2E1_RESULTS.md +
  s2e1_results.json (+ s2e1_findings.md, hand-written)

- Git: do not add Co-Authored-By, Claude-Session or "Generated with Claude Code" lines to commits or PR descriptions.