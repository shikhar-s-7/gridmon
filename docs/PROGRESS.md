# Progress

## Milestones (Stage 1)
- [x] Setup: venv, pinned requirements, src layout, docs skeleton, git
- [x] M1 — event generator and datasets (B4-B6)
- [x] M2 — conventional baseline on train/val, DSP spec, golden vectors (B7)
- [x] M3 — gradient boosting, embedded size budget, C-ready export (B8; CNN not done, optional)
- [x] M4 — sealed evaluation on test_iid and test_shift (B9)
- [x] M5 — results report (RESULTS.md, results.json, findings and limitations; done with M4)
- [x] M6 — portable C core + SIL equivalence (B13.1)
- [x] M7 — C2000 port layer, integration guide, C28x compile/size (B13.2) — compile/size step is a script for a local run (D-067)
- [x] M8 — static website with in-browser classification (B13.4)

**Stage 1 complete (2026-11-02).** Next: docs/STAGE2_PLAN.md.

## Log
- 2026-11-01 Setup (commit `b6a7ed7` with M1): Python 3.12.9 venv, numpy 2.5 / scipy 1.18 / sklearn 1.9 pinned; repo initialised.
- 2026-11-01 M1 (`b6a7ed7`): generator (Bollen phasors, 7 event models, nuisances, 20 kHz synthesis, 49-tap AA FIR,
  x4 decimation, noise, 12-bit int16 storage), splits generated (train 5600, val 1750, test_iid 1750,
  test_shift 1750, previews 2x200) in 87 s; 39 generator tests pass; M1 figures written.
- 2026-11-01 M2 (`f3b97ba`, handoff `6e87145`): DSP chain (Urms(1/2), 1 kHz DFT phasors, f_dft/ROCOF/J, DSOGI-PLL, envelope spectra, THD),
  37 features, priority rules, train-only threshold tuning, pre-event-only control (val macro-F1 0.146),
  naive-PLL diagnostic, float32 robustness, 27 golden vectors (2.0 MB), DSP_SPEC.md, FEATURES.md,
  embedded budget (1.52 M ops, ~21 KB RAM). Rules on val: macro-F1 0.933 (default and tuned);
  main error = growing oscillations reaching dip/swell levels (open question in HANDOFF).
- 2026-11-01 Session 2 (PROMPT_02), Linux cloud VM: environment re-created (all pins install unchanged);
  M1/M2 reproduce exactly on Linux (val macro-F1 0.933, same confusion; golden int16 inputs bit-identical).
- 2026-11-01 R1 (`d713ccc` code): oscillation label margin D-034 (LL fundamental within [0.92, 1.08] over the
  record; rejection 0.73-0.78), D-039 infeasible-band fallback (fast subset only), platform-stable FIR file
  (D-038). Non-oscillation records bit-identical before/after. All splits regenerated (145 s); M2 re-run:
  rules val macro-F1 0.997 (default = tuned; oscillation 249/250), pre-event control 0.140, golden regenerated.
- 2026-11-01 M3 (`fd105b4` code, `e11cc1d` results): GBM-embedded grid (60 configs x 3 seeds, 36 fits via stages, 261 s), selected
  20 stages x 4 leaves (lr 0.1, subsample 1.0) = 13,972 B, val macro-F1 1.000 (rules 0.997, HGB reference 0.9998);
  export JSON + C tables; float32 walker == sklearn on val (max |dp| 2e-7, labels 100 %, 0 branch mismatches);
  pre-event-only control 0.124; docs/MODEL_FORMAT.md; D-040..D-046.
- 2026-11-01 M4 (`da44023`): freeze `6240f60` (configs/frozen_methods.yaml; evaluation code dry-run on val first), one run on
  the sealed sets: test_iid rules 0.9983 / GBM-embedded 1.0000 / GBM-unconstrained 1.0000 (no significant
  difference); test_shift rules 0.9903 / GBM-embedded 0.8653 +/- 0.0014 / GBM-unconstrained 0.8484 +/- 0.0066
  (rules significantly better: -0.125, CI [-0.141, -0.110]). reports/RESULTS.md, results.json, m4 figures; D-047, D-048.
- 2026-11-01 Wrap-up (session 2): 81 tests pass (new: oscillation margin, GBM export equivalence, frozen-config
  hashes, evaluate_test refusal, statistics); run_all --fast 88 s through all five stages; docs updated; new
  HANDOFF (session-1 handoff archived in docs/handoffs/). Next: Prompt 03 (C core, SIL).
- 2026-11-01 Session 3 (PROMPT_03), Linux cloud VM: venv re-created, all splits regenerated (123 s). D-049..D-054 logged.
- 2026-11-01 M6: portable C99 float32 core c/ (gridmon-core 0.6.0; one file per DSP_SPEC block + features, rules, GBM
  walker, API), constants generated as float32 hex literals (scripts/gen_c_tables.py, D-054), host build/ctypes SIL
  (scripts/build_c.py, src/gridmon/c_sil.py, scripts/run_sil.py -> reports/m6_sil.json): golden 27/27; rule labels
  100 % on val/test_iid/test_shift (1750 each) and 560 train records; GBM bit-exact on the same C features, 100 %
  end to end; features within tolerance except 2 rho values (float32 argmin tie, D-060); ASan/UBSan clean; static
  C28x checks, ILP32 and 16-bit-int compiles pass; sizeof(gm_ctx_t) 15,816 B; ~1.3 ms/record on the host. D-055..D-061.
- 2026-11-01 M7: port layer firmware/c2000 (calibration trim D-052, LN->LL, polyphase FIR taps for 20/40/50/100 kHz generated
  per D-054, 45 MACs per ISR sample, SPSC ring 1.6 s, trigger Urms/J/frequency/envelope, onset back-extrapolation,
  record placement at 0.28 s, telemetry), continuous-stream generator option (record generator bit-identical), host
  simulation (scripts/run_port_stream.py): 120 s at 50 kHz, 12/12 events captured, onset 0.225-0.288 s in the record,
  rules and GBM 12/12, 0 false triggers, 0 overruns; LN input identical outcome. C28x compile check script +
  docs/C2000_BUILD.md (ti.com download not possible in the cloud session), LaunchPad PIL template (UNTESTED),
  docs/C2000_INTEGRATION.md. D-062..D-068.
- 2026-11-02 Session 4 (PROMPT_04): user's local L2 run integrated (cl2000 25.11.0.LTS, all files compile, GBM on/off; D-076);
  decisions D-069..D-074 (rho SIL gate: 1 remaining test_iid exceedance at urms_min_1 0.942, non-dip).
- 2026-11-02 M8: website web/ (overview, results, explorer, try it, firmware, about), data exported from the committed reports,
  C core in WebAssembly (zig wasm32-wasi reactor, 45 KB, no imports; equal to native C on 97 records), headless browser check
  (no console errors, phone width, explorer re-classification 5/5, try-it 7/7 classes + example CSV), web/dist 5.8 MB,
  GitHub Pages workflow + docs/DEPLOY.md. D-075, D-077, D-078.
- 2026-11-02 Docs: README landing page (generated results block), docs/STAGE2_PLAN.md, docs/PROJECT_SUMMARY.md.
