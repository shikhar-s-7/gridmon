# HANDOFF — gridmon session 1 (PROMPT_01 v0.2), 2026-11-01

## 1. Status
- Completed: setup, **M1** (generator + datasets), **M2** (baseline on train/val, DSP spec, golden vectors).
- Commits: M1 `ae6a09a`, M2 `703f398` (this handoff is the next commit). Data meta.json records
  generator commit `ae6a09a+dirty` (generator code unchanged since M1; generation is deterministic).
- Runtimes (Windows 11 laptop CPU, Python 3.12.9): `run_all.py --fast` 24.3 s; full `run_all.py` 116.1 s
  (dataset 76.9 s, baseline 36.3 s, golden 0.7 s); `pytest` 18 s.
- No v0.1 work existed, so no migration (D-021).

## 2. What exists
- `generator/phasors.py` Bollen A-G phase sets, Fortescue, closed forms, special-phase rotation.
- `generator/nuisance.py` pre-event/nuisance draw (one code path for all classes; iid and shift configs).
- `generator/events.py` 7 event models, hard negatives, analytic label margins and realised margin metrics.
- `generator/synth.py` 20 kHz synthesis, 49-tap AA FIR (configs/aa_fir.json), x4 decimation, noise, 12-bit int16.
- `generator/dataset.py` SeedSequence seeding, rejection (event stream only), chunked split writer.
- `dsp/` rms (Urms(1/2)), phasor (1 kHz DFT + sequences), freq (f_dft, ROCOF, J), pll (DSOGI), envelope, thd.
- `features.py` 37 features + aux outputs + float32 tolerance checker; `baseline_rules.py` priority rules + grid tuning.
- `metrics.py`, `budget.py` (B7.6 estimate), `golden.py`, `plots.py`, `io.py` (sealed-split guard), `units.py`.
- `scripts/` make_dataset.py, run_baseline.py, make_golden.py, run_all.py (all accept `--fast`).
- `tests/` test_generator.py (39), test_dsp.py (24).
- docs: PROJECT_BRIEF, DECISIONS (D-001..D-033), PROGRESS, DSP_SPEC, FEATURES, VALIDATION, ENVIRONMENT.

## 3. Verification
- pytest: **63 passed, 0 failed, 0 skipped** (on full data).
- Bollen closed forms exact (1e-9); B/E minus zero sequence = D((1+2V)/3)/G; special-phase rotation V2 x a.
- Clean record LL RMS 1.000 +/- 0.002; abc sequence; FIR |H(50 Hz)| = 1 (1e-12); 53.9 dB at >= 2.5 kHz.
- theta(t) reproduces f(t) < 1 mHz; one-cycle-DFT jump angle within 0.3 deg (6..30 deg, both signs).
- Determinism (bit-identical re-generation; record alone == in chunk); 10,850 record seeds unique.
- Margins re-derived analytically from (seed, label) for every record of every split; waveform
  one-cycle-RMS re-check on train/val (sags < 0.90, swells > 1.10, normal within [0.90, 1.10]).
- DSP: phasor error < 0.1 % / 0.1 deg; Urms step detected within one cycle; sequences within 0.5 %
  for A/C/D/F/G; PLL lock 68 ms (|f-50| <= 0.01 Hz), 1 Hz/s ramp mean error -0.004 Hz;
  J = 20 +/- 1 deg for a 20 deg step, < 1 deg for 4 Hz/s; rho 0 / 0.5 / 1 within 0.05.

## 4. Dataset
| split | normal | sag_bal | sag_unbal | swell | phase_jump | freq_event | oscillation | total |
|---|---|---|---|---|---|---|---|---|
| train (1001, iid) | 800 | 800 | 800 | 800 | 800 | 800 | 800 | 5600 |
| val (2002, iid) | 250 | 250 | 250 | 250 | 250 | 250 | 250 | 1750 |
| test_iid (3003, sealed) | 250 | 250 | 250 | 250 | 250 | 250 | 250 | 1750 |
| test_shift (4004, sealed) | 250 | 250 | 250 | 250 | 250 | 250 | 250 | 1750 |
| preview_iid / _shift (9009/9010) | 29 | 29 | 29 | 29 | 28 | 28 | 28 | 200 each |

- Margin-rejection rate (train/val/test_iid/test_shift): swell 0.15/0.17/0.20/0.16; sag_unbal
  0.001/0/0.008/0; sag_bal 0/0.004/0/0; all other classes 0.
- Realised (iid | shift): sag_bal V 0.10-0.85, min LL 0.10-0.868 | 0.10-0.857; sag_unbal V 0.001-0.85,
  min LL 0.005-0.864 (C 211, D 190, F 207, G 192 in train); swell S 1.10-1.30, max LL 1.13-1.36;
  jump +/-6..30 deg, |dV1| <= 3 %; freq ROCOF +/-0.5..4 Hz/s, |f_end - f_pre| 0.275-1.52 Hz;
  oscillation depth 2-8 %, f_m 5-25 Hz, sigma -1..2 /s, depth x envelope at end 1-34 %;
  normal LL range 0.925-1.06 pu. Shift transient in 32.7 % of test_shift records.
- Max |int16 count|: 1420 iid (1.39 pu), **1879 test_shift** (1.83 pu); no 12-bit saturation anywhere.

## 5. Baseline on VAL (rules; thresholds tuned on TRAIN only)
| | macro-F1 | normal | sag_bal | sag_unbal | swell | phase_jump | freq_event | oscillation |
|---|---|---|---|---|---|---|---|---|
| default | **0.933** | 1.000 | 0.874 | 0.992 | 0.938 | 0.990 | 1.000 | 0.737 |
| tuned (rho_th 0.35, rest = default) | **0.933** | 1.000 | 0.873 | 0.994 | 0.938 | 0.990 | 1.000 | 0.737 |

Confusion, default (rows true, cols predicted; tuned differs only in oscillation: 73 / 0 / 31):
```
true \ pred     normal  sag_bal sag_unbal  swell  phase_jump freq_event oscillation
normal             250        0        0       0          0          0          0
sag_bal              0      250        0       0          0          0          0
sag_unbal            0        0      250       0          0          0          0
swell                0        0        0     250          0          0          0
phase_jump           0        0        3       2        245          0          0
freq_event           0        0        0       0          0        250          0
oscillation          0       72        1      31          0          0        146
```
- False-alarm rate (normal -> any event) **0.0 %**; miss rate (event -> normal) **0.0 %**.
- Critical confusions: phase_jump->freq_event 0, freq_event->phase_jump 0, sag_*->normal 0.
- Train macro-F1 0.9412 (default) / 0.9413 (tuned); tuning finds no real gain (21,000 combos).
- Onset error (correct detections, median / p90 |err|): dips -0.9 / 6 ms, swell 3 / 9 ms,
  jump 4 / 8 ms, freq 72 / 133 ms, oscillation 51 / 68 ms (indicative only).
- Runtime: features 2.5 ms/record (float64 numpy), rules < 0.001 ms/record.

## 6. Diagnostics
- **Pre-event-only control** (HGB on first-0.15 s features): val macro-F1 **0.140** (chance 0.143) -> PASS,
  no generator leakage. (Fast mode gives 0.22: 560 train records, noisy; quote the full value.)
- **Naive PLL rule** (|f_pll - f_ref| > 1 Hz, val): normal 0/250 (0 %), sag_bal 249/250 (99.6 %),
  sag_unbal 250/250 (100 %), phase_jump 250/250 (100 %); also swell 34 %, oscillation 42 %
  (freq_event 104/250 are true frequency excursions). By jump size: 6-12 deg 75/75, 12-20 87/87,
  20-30 88/88 (100 % each). The DFT jump metric + masked f_dft gives 0 jump<->freq confusions.

## 7. Embedded readiness
- DSP_SPEC.md and FEATURES.md complete: **yes** (index grids, validity, coefficients, init, edges).
- Golden vectors: **27 files, 2.05 MB** (21 val + 6 unit signals) with golden/README.md tolerances.
- float32 test: all 1750 val rule labels unchanged (default and tuned); all 37 features within
  tolerance (argmax-type features compared under documented gates, D-027).
- Budget (reports/m2_embedded_budget.json): **1.52 M ops/record** (1.38 M MAC, 24 k transcendental),
  **21.2 KB** working RAM + 6.1 KB const tables (besides the 30 KB record); no flags.
- Awkward on C2000: per-sample sin/cos/sqrt/div in the PLL (TMU recommended); 16-bit char sizes;
  a second pass over the 4 KB f_dft buffer (jump mask before ROCOF); running median sorts;
  THD (450 k MAC) and the phasor DFT (589 k MAC) dominate; AA FIR is designed for 20 kHz input
  (TIDA-01606 at 50 kHz needs a /10 decimator to the same spec; `design_aa_fir` is parametric).

## 8. Deviations, known issues, open questions
- Deviations/interpretations: D-012 (group-delay-compensated timeline), D-015 (generator details B6
  leaves open), D-016 (ROCOF as a direct 100-tap LS kernel, not running sums), D-022 (prewarped
  Tustin, PLL init), D-023 (rho on f_ref-de-rotated V2), D-024 (envelope windows p 2..91),
  D-025 (j_sign = first lobe), D-026 (windows/masking), D-027 (float32 gates), D-017 (sealed
  manifests read by tests for counts/seeds/margins only).
- **Open Q1 (main error source):** the oscillation class is not margin-separated from dips/swells.
  With sigma up to +2 /s and depth up to 8 %, the |V1| modulation reaches 34 % by record end;
  124/250 val oscillation records have an analytic LL fundamental outside [0.90, 1.10], and the
  rules (dip/swell first) call 104 of them sag_bal/swell (oscillation recall 0.58). Options:
  (a) keep (a standards-based monitor would also report the dip/swell), (b) add an oscillation
  margin (envelope within [0.92, 1.08]) or cap sigma, (c) add an oscillation check inside the
  dip/swell branch. Not changed: B6.3 is explicit. Decide before M3/M4 (it affects all methods).
- Q2: 5/250 phase jumps cross Urms 0.90/1.10 transiently (one-cycle window across the jump with
  U_pre near the band edge) -> sag_unbal/swell. Physical behaviour of Urms(1/2); accept?
- Q3: threshold tuning has nothing to gain on train; keep tuned = default + rho_th 0.35 for M4?
- Known: DSOGI with fixed w0 shows +/-0.1 Hz 100/300 Hz ripple from unbalance/harmonics (no FLL,
  no harmonic decoupling), visible in f_pll; it does not affect the rules (f_dft-based).

## 9. Figures (reports/figures/)
- m1_examples_per_class.png, m1_param_distributions.png, m1_shift_preview.png
- m2_signal_chain_examples.png, m2_baseline_val_confusion.png, m2_naive_pll_false_alarms.png

**Bring back to the project-lead chat:** docs/HANDOFF.md, reports/figures/m1_examples_per_class.png,
reports/figures/m2_signal_chain_examples.png, reports/figures/m2_baseline_val_confusion.png
