# Progress

## Milestones (Stage 1)
- [x] Setup: venv, pinned requirements, src layout, docs skeleton, git
- [x] M1 — event generator and datasets (B4-B6)
- [x] M2 — conventional baseline on train/val, DSP spec, golden vectors (B7)
- [ ] M3 — gradient boosting (+ optional CNN), embedded size budget (B8)
- [ ] M4 — sealed evaluation on test_iid and test_shift (B9)
- [ ] M5 — results report (RESULTS.md, results.json, findings and limitations)
- [ ] M6 — portable C core + SIL equivalence (B13.1)
- [ ] M7 — C2000 port layer, integration guide, C28x compile/size (B13.2)
- [ ] M8 — static website with in-browser classification (B13.4)

## Log
- 2026-11-01 Setup: Python 3.12.9 venv, numpy 2.5 / scipy 1.18 / sklearn 1.9 pinned; repo initialised.
- 2026-11-01 M1: generator (Bollen phasors, 7 event models, nuisances, 20 kHz synthesis, 49-tap AA FIR,
  x4 decimation, noise, 12-bit int16 storage), splits generated (train 5600, val 1750, test_iid 1750,
  test_shift 1750, previews 2x200) in 87 s; 39 generator tests pass; M1 figures written.
- 2026-11-01 M2: DSP chain (Urms(1/2), 1 kHz DFT phasors, f_dft/ROCOF/J, DSOGI-PLL, envelope spectra, THD),
  37 features, priority rules, train-only threshold tuning, pre-event-only control (val macro-F1 0.146),
  naive-PLL diagnostic, float32 robustness, 27 golden vectors (2.0 MB), DSP_SPEC.md, FEATURES.md,
  embedded budget (1.52 M ops, ~21 KB RAM). Rules on val: macro-F1 0.933 (default and tuned);
  main error = growing oscillations reaching dip/swell levels (open question in HANDOFF).
