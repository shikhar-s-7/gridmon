# Progress

## Milestones (Stage 1)
- [x] Setup: venv, pinned requirements, src layout, docs skeleton, git
- [x] M1 — event generator and datasets (B4-B6)
- [x] M2 — conventional baseline on train/val, DSP spec, golden vectors (B7)
- [x] M3 — gradient boosting, embedded size budget, C-ready export (B8; CNN not done, optional)
- [ ] M4 — sealed evaluation on test_iid and test_shift (B9)
- [ ] M5 — results report (RESULTS.md, results.json, findings and limitations)
- [ ] M6 — portable C core + SIL equivalence (B13.1)
- [ ] M7 — C2000 port layer, integration guide, C28x compile/size (B13.2)
- [ ] M8 — static website with in-browser classification (B13.4)

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
- 2026-11-01 M3 (`fd105b4` code): GBM-embedded grid (60 configs x 3 seeds, 36 fits via stages, 261 s), selected
  20 stages x 4 leaves (lr 0.1, subsample 1.0) = 13,972 B, val macro-F1 1.000 (rules 0.997, HGB reference 0.9998);
  export JSON + C tables; float32 walker == sklearn on val (max |dp| 2e-7, labels 100 %, 0 branch mismatches);
  pre-event-only control 0.124; docs/MODEL_FORMAT.md; D-040..D-046.
