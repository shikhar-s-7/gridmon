# Progress

## Milestones (Stage 1)
- [x] Setup: venv, pinned requirements, src layout, docs skeleton, git
- [x] M1 — event generator and datasets (B4-B6)
- [ ] M2 — conventional baseline on train/val, DSP spec, golden vectors (B7)
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
