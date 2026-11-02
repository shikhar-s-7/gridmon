# S2E1 pre-registration — does ML catch up with representative training data? [SIM]

Stage 2, experiment 1 (docs/STAGE2_PLAN.md item 3). Written 2026-11-02 and committed as "S2E1-prereg: ..." together
with configs/s2e1.yaml (the machine-readable version of everything below) **before any S2E1 split was generated**.
Before this commit the interharmonic generator option was only checked on single synthetic clean records
(amplitude, sequence) and by regenerating two Stage 1 preview splits bit-identically; no train, val, test or preview
split of S2E1 existed. Simulation-only; every result will be [SIM].

## 1. Question and context
Stage 1 is complete and frozen (configs/frozen_methods.yaml, M4-freeze commit 6240f60). Its main finding: on test_shift
the rules (macro-F1 0.990) beat the GBM-embedded (0.865) because the GBM, trained only at 35-60 dB SNR and +/-0.5 %
calibration error, learned noise-dependent shortcuts (reports/RESULTS.md findings 2, 3, 7).

**Question:** when the GBM is trained on the full nuisance ranges it will meet, does it match the rules on fresh shifted
data, and do either of them survive a nuisance neither has seen? Every Stage 1 method and file stays untouched; new
work goes in new files. "Rules still win" or "no significant difference" are valid outcomes. The Stage 1 sealed sets
(test_iid, test_shift) are not used in S2E1 at all.

## 2. Design (project lead, PROMPT_05 A3, verbatim)
> ### New data (generator option configs only; Stage 1 configs unchanged)
> - Nuisance config "wide": SNR U[25, 60] dB; f_pre U[49.5, 50.5] Hz; harmonics h5 U[0,5], h7 U[0,4],
>   h11 U[0,2], h13 U[0,2] %; background unbalance U[0, 2.5] %; gain U[-2, 2] %; offset U[-2, 2] %.
>   NO transients of any kind (the Stage 1 capacitor-switching transient stays unseen in training).
> - Splits (250 records/class unless stated; new seeds never used before):
>   train_wide 800/class seed 1101 (wide); val_wide seed 2202 (wide; selection only);
>   test2_shift seed 5005 (sealed; identical config to Stage 1 test_shift incl. the transient in 30 %);
>   test2_novel seed 6006 (sealed; wide nuisances plus a SECOND unseen nuisance in 50 % of records:
>   one interharmonic voltage component, frequency U[70, 950] Hz excluding +/-10 Hz around every
>   harmonic of 50 Hz, amplitude U[0.5, 2] % of fundamental, present over the whole record, label
>   unchanged; context: IEEE 1159 lists interharmonics at 0-2 % [STD]). Note in the prereg that an
>   interharmonic at f_ih beats with the fundamental in a one-cycle DFT magnitude at |f_ih - 50| Hz
>   and multiples, so it can mimic the oscillation class for BOTH methods — that is intended.
>   Label margins as in Stage 1 (incl. D-034); verify they hold.
> ### Arms
> - R: Stage 1 rules, frozen, unchanged (no retraining).
> - G0: Stage 1 GBM-embedded (frozen models, seed 0-2), unchanged.
> - GW: GBM-embedded recipe (same 37 features, same grid, 64 KB budget, 3 seeds) trained on
>   train_wide, selected on val_wide macro-F1 (ties within 0.002 -> smaller model).
> - GP: as GW but restricted to a physics subset of features fixed in the prereg by physical
>   reasoning only: the magnitude, sequence, Bollen ratio, jump-metric, f_end - f_ref and envelope
>   amplitude features; EXCLUDE rocof_max/rocof_* time-above features, envelope prominence features
>   and THD features. Disclose that this choice was motivated by the Stage 1 test_shift diagnosis
>   (which is why it is evaluated only on new sealed data).
> - H0 / HW: hybrids — G0 (resp. GW) label if its max probability >= 0.5, otherwise the rules label.
>   Threshold 0.5 fixed in the prereg (from Stage 1 finding 7; no tuning).
> ### Evaluation (once, after S2E1-freeze)
> - On test2_shift and test2_novel: macro-F1 per arm (ML mean +/- sd over seeds; hybrids per seed),
>   per-class F1, false-alarm and miss rates, critical confusions (phase_jump <-> freq_event,
>   sag -> normal, sag_bal <-> sag_unbal), breakdowns by SNR bin and by presence of the transient /
>   interharmonic (and interharmonic |f_ih - 50| bin: < 25 Hz vs >= 25 Hz).
> - Primary comparisons (paired bootstrap 95 % CI, 2,000 resamples stratified by class; exact
>   McNemar): GW - R and GP - R on both sets; secondary: HW - R, H0 - R, GW - G0.
> - Pre-registered decision rule: "ML matches the rules" only if the CI of (arm - R) includes 0 or is
>   above 0 on BOTH sealed sets; "ML better" only if the CI lies above 0.
> - Pre-event-only control for GW and GP (expect ~0.14).

## 3. Data (implementation)
| split | records | seed | nuisance | role |
|---|---|---|---|---|
| train_wide | 800/class (5,600) | 1101 | wide | training of GW, GP |
| val_wide | 250/class (1,750) | 2202 | wide | selection of GW, GP only |
| test2_shift | 250/class (1,750) | 5005 | Stage 1 `shift` (by reference, incl. transient p = 0.3) | sealed |
| test2_novel | 250/class (1,750) | 6006 | wide + interharmonic (p = 0.5) | sealed |
| preview_wide / preview_novel | 70 / 140 (round-robin) | 9101 / 9102 | wide / novel | development, dry run, figures |

- Generator: the Stage 1 generator and measurement chain unchanged; configs/dataset.yaml supplies record, ADC, AA FIR,
  event models, margins and chunking; configs/s2e1.yaml adds the nuisances `wide` and `novel` and the splits
  (gridmon.s2e1.dataset_config). u_pre [0.96, 1.04] and drift +/-0.03 Hz/s stay as in Stage 1 (not named in the design).
- **Interharmonic model (D-080):** one balanced positive-sequence component over the whole record,
  `a_ih u_pre cos(2 pi f_ih t + phi_ih - 2 pi k/3)` added to the phase voltages (so every LL channel carries a_ih of its
  fundamental); f_ih fixed (does not track the grid frequency), uniform over [70, 950] Hz minus (50k - 10, 50k + 10) Hz
  (allowed length 530 Hz, one uniform draw mapped onto the 18 intervals), a_ih U[0.5, 2] %, phi_ih U[0, 2 pi).
  Drawn from the record's pre-event stream after every Stage 1 draw, so configurations without the option draw exactly as
  before (Stage 1 data are bit-identical; tested). Alternatives considered: an unbalanced/single-phase interharmonic (more
  parameters, no stated source); frequency tracking f (not how interharmonic sources behave).
- **Beat mechanism (intended):** in the one-cycle DFT, a component at f_ih leaks into the 50 Hz phasor with gain
  ~ |sinc((f_ih - 50) x 20 ms)| and rotates against it at |f_ih - 50| Hz, so |V1| and angle(V1) ripple at that frequency.
  The 100 Hz envelope (10 ms block means) attenuates it by ~ |sinc(|f_ih - 50| x 10 ms)| and aliases higher beats to
  |f_ih - 50| mod 100 Hz ("and multiples"). Expected envelope amplitude ~ a_ih x 0.71 / 0.57 / 0.43 for beats of 20 / 25 / 30 Hz,
  so the rules' oscillation test (> 1.0 % with prominence >= 4) can only fire for beats below ~27 Hz (f_ih ~70-77 Hz) at the
  upper amplitudes. The phase ripple also reaches the frequency-derivative features (ROCOF, df_max, PLL) for every f_ih.
  These are expectations from the design, not hypotheses; the < 25 Hz bin will be small (about 5/530 of the ~875
  interharmonic records, ~8), hence the additional pre-registered secondary bins (section 6).
- **Label margins:** Stage 1 margins (brief B5) incl. D-034 (oscillation LL band) and D-039 (least-violating fallback when the
  band is infeasible). The interharmonic is a measurement nuisance like the harmonics; margins are defined on the
  fundamental phasors and are unaffected (label unchanged by design). Verified per split from the manifests after
  generation (reports/s2e1_data_summary.json) and by tests.

## 4. Arms
- **R:** Stage 1 rules with the frozen TRAIN-tuned thresholds (D-036). **G0:** Stage 1 GBM-embedded, frozen models seeds 0-2,
  float32 walker. Both verified unchanged by gridmon.frozen.verify (part of the S2E1 freeze check).
- **GW:** the Stage 1 GBM-embedded recipe unchanged (gridmon.ml_gbm, configs/ml.yaml: grid n_estimators {20, 30, 40, 60, 100}
  x max_leaf_nodes {4, 8, 16} x learning_rate {0.1, 0.2} x subsample {0.8, 1.0}, seeds 0-2, budget 65,536 B, D-040 input
  mapping) on the 37 features of train_wide.
- **GP:** as GW on 23 features. **Physical reasoning (D-081):** the classes are defined by the fundamental magnitude (dip,
  swell), its sequence content (balanced vs unbalanced), a phase step, a sustained frequency change and a magnitude/angle
  modulation; the subset keeps the features that measure these quantities directly with record-level averaging or IEC
  thresholds: magnitude (urms_min_1..3, urms_max_1..3, t_below_090, t_above_110, v1_ref, v1_min_rel, v1_max_rel,
  v1_end_rel), sequence (u2_ref, u2_max, u2_end), Bollen ratio (rho), jump metric (j_max_deg, j_sign), f_end - f_ref
  (df_end) and envelope amplitude (env_mag_amp_pct, env_ang_amp_deg, env_mag_rms_pct, env_ang_rms_deg). Excluded (14):
  f_ref (pre-event frequency, a nuisance), df_max, rocof_max, t_rocof_02, rocof_end, dfpll_max, t_fpll_05 (extremes and
  time-above counts of differentiated frequency signals, which amplify noise), env_mag_freq, env_ang_freq (frequencies, not
  amplitudes), env_mag_prom, env_ang_prom (ratios to the noise floor) and thd_first, thd_last, thd_max (harmonic
  background is a nuisance in every class). Interpretation of "magnitude" and "envelope amplitude" as whole feature
  families (incl. the IEC time-below/above counts and the envelope RMS values) is ours (D-081).
  **Disclosure:** this choice was motivated by the Stage 1 test_shift diagnosis (finding 3: rocof_max and env_mag_prom
  shortcuts), which is why GP is evaluated only on new sealed data.
- **Export:** GW and GP models are written to models/s2e1/ (GP with feature indices remapped to the full 37-feature vector,
  so the same walker and C tables apply) and the seed-0 C tables to c/generated_s2e1/{gw,gp}/ only if every seed fits 64 KB.
- **H0 / HW:** per ML seed: ML label if the walker's max softmax probability >= 0.5, otherwise the rules' label.

## 5. Selection (GW, GP)
Highest mean val_wide macro-F1 over the 3 seeds among configurations whose exported size is <= 65,536 B for every seed;
configurations within 0.002 of the best -> the smaller model (max bytes over seeds), then higher F1, fewer stages,
fewer leaves (gridmon.ml_gbm.select_embedded, unchanged). Only train_wide and val_wide are used before the freeze.
Pre-event-only control: the selected configuration refit on features of the first 750 samples (OBS = REF), train_wide
-> val_wide; must be <= 0.25 macro-F1 (expected ~0.14 = chance).

## 6. Evaluation plan (one run on test2_shift and test2_novel, after the S2E1 freeze, at the freeze commit)
- Per arm: macro-F1 (ML mean +/- sd over seeds; hybrids per seed and mean), per-class F1, false-alarm rate (normal called
  an event), miss rate (event called normal), critical confusions (phase_jump <-> freq_event, sag_* -> normal,
  sag_bal <-> sag_unbal), confusion matrices (seed 0).
- Breakdowns: SNR bins [25, 30, 35, 40, 50, 60] dB; transient yes/no (test2_shift); interharmonic yes/no (test2_novel);
  interharmonic records by |f_ih - 50| < 25 Hz vs >= 25 Hz (primary) and by the secondary edges [20, 25, 40, 100, 900] Hz
  (recall per bin, n reported); Stage 1 severity bins.
- **Comparisons:** paired bootstrap, 2,000 resamples stratified by class, one set of resamples per sealed set shared by all
  arms (seed 20261102); statistic = seed-mean macro-F1(arm) - macro-F1(reference); for GW - G0 the seed-mean of both;
  95 % percentile CI; exact two-sided McNemar per seed (arm seed k vs R; GW seed k vs G0 seed k). Primary: GW - R, GP - R on
  both sets; secondary: HW - R, H0 - R, GW - G0. No multiplicity correction; the decision needs both sets.
- **Unseen-nuisance survival (D-082)** on test2_novel, per arm: delta = macro-F1(with interharmonic) - macro-F1(without),
  seed-mean, unpaired bootstrap stratified by class within each subset (2,000). *Survives* if the 95 % CI lower bound
  >= -0.03; *fails* if the CI upper bound < 0 and delta <= -0.03; otherwise *inconclusive*. (0.03 ~ 4x the Stage 1 rules'
  iid -> shift change, ~1/4 of the Stage 1 GBM's.)

## 7. Hypotheses and decision rule
Decision rule, per ML arm, on the CI of macro-F1(arm) - macro-F1(R) on each sealed set: **ML better** if the CI lies above 0
on both sets; **ML matches** if it includes 0 or lies above 0 on both sets (and not "better"); **rules better** if it lies
below 0 on at least one set. GW and GP are primary (the verdict); HW and H0 get the same reading as secondary results.
- H1: GW matches or beats the rules. H2: GP matches or beats the rules. H3: GW - G0 > 0 on test2_shift (CI above 0).
- H4: HW matches or beats the rules. H5: R survives the interharmonic. H6: GW survives the interharmonic.
- Replication expectation (not a hypothesis): R ~0.99 and G0 ~0.87 on test2_shift, as on Stage 1 test_shift.

## 8. Procedure
1. "S2E1-prereg" commit (this file, configs/s2e1.yaml, generator option, gridmon.s2e1, data script, tests).
2. Generate all S2E1 splits at that commit (scripts/s2e1_make_data.py); margin and interharmonic checks; features for the
   non-sealed splits only.
3. Train/select GW and GP on train_wide/val_wide (scripts/s2e1_train.py; code committed before models are generated);
   pre-event control; export. Dry run of the evaluation with val_wide and preview_novel standing in for the sealed sets.
4. "S2E1-freeze" commit: configs/s2e1_frozen.yaml (models + SHA-256, S2E1 method files, Stage 1 freeze check, this plan).
   scripts/s2e1_evaluate.py refuses to load test2_* unless both freezes verify.
5. One evaluation run -> reports/s2e1_results.json, S2E1_RESULTS.md, figures. Any deviation from this plan is logged in
   docs/DECISIONS.md and labelled in the results.

## 9. Known limitations (stated in advance)
Same generator physics for training and test; test2_shift's configuration is known (only its data are new); the rules and
the novel nuisance were designed by people who know the generator; the GP subset is motivated by the Stage 1 diagnosis;
the interharmonic is designed to be able to mimic oscillation; < 25 Hz beat bin small; 4 primary tests without correction.
