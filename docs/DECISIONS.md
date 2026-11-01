# Decisions log

Format: ID — date — what — why — alternatives. D-001..D-011 are the initial decisions of the
brief (B11). Later entries record every deviation from, or interpretation of, PROJECT_BRIEF.md.
Evidence tags as in B3.

## Initial decisions (brief B11), 2026-11-01
- **D-001** Offline record-level classification of triggered records, not real-time control.
- **D-002** Physics-based synthetic generator (sequence phasors + Bollen dip types); EMT/Simulink
  cross-check later.
- **D-003** 3-wire LL measurement; zero sequence unobservable.
- **D-004** Seven classes mapped to distinct PCS responses; single-label records; label margins.
- **D-005** Fair comparison: the same features for rules and gradient boosting; tuning on train/val
  only; sealed test sets.
- **D-006** Shifted test set with an unseen nuisance, to measure robustness rather than iid accuracy.
- **D-007** (v0.2) Monitor rate 5 kHz (100 samples/cycle) from a 20 kHz synthesis: exact integer
  decimation from common C2000 control rates; replaces 3.2 kHz.
- **D-008** (v0.2) int16 12-bit ADC-count storage: matches what the MCU sees; halves dataset size.
- **D-009** (v0.2) Port-exact DSP definitions (non-recursive DFT at 1 kHz, bin DFTs, Tustin
  discretisation, bounded memory) plus golden vectors, so a C core can be verified without hardware.
- **D-010** (v0.2) One portable C99 float32 core, three targets: PC (SIL), C2000 (monitor-only
  module), browser (WebAssembly). Three-phase TIDA-01606-class hardware first.
- **D-011** (v0.2) Gradient boosting is the embedded ML component (<= 64 KB); the CNN becomes
  optional offline, with an embedded CNN deferred to an NPU device (Stage 2).

## Implementation decisions

- **D-012** — 2026-11-01 — *Record timeline is FIR-group-delay compensated.* The generator
  synthesises the 20 kHz analog signal on t_k = (k - gd)/20000 (gd = 24 samples), so stored sample
  n corresponds to analog time n/5000 exactly and the manifest onset `ev_t0` is on the record
  timeline. The FIR itself is causal; a real-time firmware decimator adds gd/20 kHz = 1.2 ms delay
  (recorded in meta.json). *Why:* onset times and margins refer to the record the methods see.
  *Alternative:* uncompensated (every event 1.2 ms later than `ev_t0`).
- **D-013** — 2026-11-01 — *Anti-alias FIR:* Kaiser-window lowpass, 49 taps, beta 5.0, cutoff
  1750 Hz at 20 kHz, normalised to |H(50 Hz)| = 1 (exact in float64). Realised: passband ripple
  -0.003/+0.026 dB to 1 kHz, 53.9 dB attenuation at >= 2.5 kHz, group delay 24 input samples
  (= 6 output samples). Odd length gives integer delay. *Why:* meets B4 with margin, short.
  *Alternative:* Parks-McClellan (similar length). The coefficients are for a 20 kHz input; a
  firmware rate of 50 kHz (TIDA-01606) needs a redesign to the same spec
  (`design_aa_fir(fcfg, fs_in)` is parametric).
- **D-014** — 2026-11-01 — *Seeding.* Record i of a split uses child i of
  SeedSequence(split_seed).spawn(N); its 63-bit integer `seed` (child.generate_state, shifted)
  is stored in the manifest and seeds SeedSequence(seed).spawn(3) = (pre, event, noise) streams.
  The label order is shuffled with the next child. Any record can be regenerated from
  (seed, label, nuisance config). `--fast` uses the first 10 % of children (labels re-shuffled).
- **D-015** — 2026-11-01 — *Generator interpretations where B6 is silent* [ASSUMED]:
  (a) V1_pre has angle 0; theta0 ~ U[0, 2pi) carries the absolute phase; f_pre = f(t=0).
  (b) Background V2 is constant except: sag_bal (scaled by V* during the dip and by the recovery
  multiplier after it, "scales with V1"), swell (extra |V2| ~ U[0, 1 %] x U_pre, random angle,
  added during the swell). Phase jumps do not rotate the background V2.
  (c) Harmonic amplitudes = level x U_pre, constant through events (except the normal-class
  harmonic step); phases fixed relative to h*theta(t), so harmonics track f(t) but not V1 angle steps.
  (d) Dip recovery: at the dip end the dip's own V2 disappears; V1 recovers abruptly or to
  r*U_pre then exponentially (sag_bal's background V2 follows the same multiplier).
  (e) freq_event |V1| change is a step U[-1, +1] % at t0. (f) Oscillation: |V1| and angle are
  modulated by the same sinusoid with a random relative phase phi_rel ~ U[0, 2pi); the 50 ms
  ramp-in is a raised cosine. (g) Normal hard-negative times ~ U[0.20, 0.90] s; the harmonic step
  picks h5 or h7 with p = 0.5. (h) Shift transient: balanced (positive-sequence) damped sinusoid
  added to the phase voltages before the measurement chain, so its LL amplitude equals the drawn
  initial amplitude; may start before or after the onset. (i) The onset t0 is an event parameter
  (redrawn with the event on margin rejection).
- **D-016** — 2026-11-01 — *ROCOF as a direct 100-tap least-squares-slope FIR* over f_dft - 50,
  instead of running sums (B7.1 step 3). Identical in exact arithmetic; no float32 drift from
  add/subtract recursion; ~85 k MAC per record (fits the budget). *Alternative:* running sums
  with periodic re-synchronisation.
- **D-017** — 2026-11-01 — *Sealed-split guard and its test exception.* `io.load_split` /
  `load_waveforms` / `load_labels` raise for test_iid/test_shift unless `allow_test=True`;
  `load_manifest` needs `allow_test_manifest=True`. Generator tests read sealed MANIFESTS for
  counts, seed disjointness, saturation, and to re-derive every record's analytic label margins
  from (seed, label) (A3 g: "every split"); no sealed waveform is loaded, plotted or evaluated.
- **D-018** — 2026-11-01 — `--fast` writes to data/fast/ and reports/fast/ (git-ignored) so it never
  overwrites full-run outputs or committed figures.
- **D-019** — 2026-11-01 — *ADC saturation* is modelled (clip to [-2048, 2047]) and counted per
  record (`n_sat`); none occurred (max |count| 1879 = 1.83 pu, test_shift manifest). The test
  asserts zero saturation in every split.
- **D-020** — 2026-11-01 — *Preview splits*: 200 records each with round-robin labels
  (29 or 28 per class), used only for the iid-vs-shift nuisance figure.
- **D-021** — 2026-11-01 — No v0.1 work existed in this folder: no migration was needed.

## M2 (baseline) decisions
- **D-022** — 2026-11-01 — *SOGI discretisation and PLL start-up.* Tustin with prewarping at 50 Hz
  (w = (2/Ts) tan(w0 Ts/2)) so the discrete SOGI resonates exactly at 50 Hz; DF-II-transposed
  biquads. PLL angle initialised to atan2(v_beta[0], v_alpha[0]); all other states 0. Clean-input
  lock in 68 ms (< 150 ms). *Alternatives:* plain Tustin (0.03 deg phase error at 50 Hz), theta = 0
  (slower, start-dependent lock).
- **D-023** — 2026-11-01 — *Bollen ratio frame.* V2 is de-rotated by the reference frequency
  (exp(-j 2 pi (f_ref - 50) t)) before taking V2_ref (component-wise median over REF) and the
  difference at m* = argmin |V1| over OBS. *Why:* the DFT phasors rotate at (f - 50) Hz; without
  de-rotation the background V2 would bias rho by up to ~0.2 in the shift set (f_pre up to 50.5 Hz).
- **D-024** — 2026-11-01 — *Envelope details.* e_mag is normalised by v1_ref and reported in %;
  e_ang in deg. Spectral windows start at the first valid envelope sample (p = 2 + 10q, q = 0..4),
  so the last 80 ms are not covered; this avoids the truncated running-median edge at the record
  end, where a monotone trend (frequency ramp) leaves a residual of up to ~5 samples x slope.
  *Alternative:* end-aligned windows (cover the end, but false oscillation peaks on freq events).
- **D-025** — 2026-11-01 — *j_sign* = sign of J at the first OBS sample with |J| >= 0.5 max|J| (the
  first lobe carries the jump sign); the sign at argmax |J| is arbitrary because a step gives
  +dphi and -dphi lobes of almost equal size.
- **D-026** — 2026-11-01 — *Feature windows and frequency masking.* REF = n <= 749, OBS = n >= 750,
  END = last 100 phasor samples. f_pll features use OBS only (excludes PLL start-up); they are not
  jump-masked (they document the PLL transient, B7.4). The frequency mask uses a fixed 3 deg
  detection level (independent of the tunable 5 deg rule threshold; between the 2 deg normal
  hard-negative maximum and the 6 deg jump minimum), +/- 40 phasor samples around argmax |J|;
  ROCOF values whose 100-sample window touches the mask are excluded; if END is fully masked,
  f_end falls back to the unmasked mean.
- **D-027** — 2026-11-01 — *float32 tolerance semantics.* Continuous features: absolute tolerances
  (FEATURES.md), ~10-100x the observed float32 error. Argmax-type features are compared only when
  well defined: env_*_freq/prom when the top-2 spectral cells differ by >= 0.1 % and the peak is
  >= 0.01 % / 0.01 deg (float32 rounding of a tens-of-rad unwrapped angle limits the angle envelope
  to ~1e-4 deg); prominences also get a 1 % relative term; j_sign when j_max_deg >= 1. The same
  semantics apply to the C SIL check (M6).
- **D-028** — 2026-11-01 — *Feature set:* the B7.2 minimum plus v1_end_rel, u2_end, rocof_end,
  env_mag_rms_pct, env_ang_rms_deg (37 features). Onset times and spectral gaps are auxiliary
  outputs, never ML features.
- **D-029** — 2026-11-01 — *Rule details.* Dip/swell use "any Urms(1/2) value beyond the threshold";
  the IEC hysteresis end condition affects duration only and is not needed for the label. The
  jump rule's "|V1| change" = max(|v1_min_rel - 1|, |v1_max_rel - 1|) over OBS. Onset estimate =
  first detection time of the firing detector (indicative; frequency events are detected ~70 ms late).
- **D-030** — 2026-11-01 — *Threshold tuning:* exhaustive grid (21,000 combinations, configs/baseline.yaml)
  maximising TRAIN macro-F1; ties broken towards the defaults (smallest sum |log(th/default)|) to
  avoid drifting to grid edges on plateaus. Dip/swell thresholds fixed at 0.90/1.10 [STD].
- **D-031** — 2026-11-01 — *Pre-event-only control:* the same feature function on the first 750
  samples with OBS = REF (ROCOF and envelope spectra become NaN); sklearn
  HistGradientBoostingClassifier with default hyper-parameters, random_state 0; all-NaN columns dropped.
- **D-032** — 2026-11-01 — *Golden vectors:* the first 3 val records per class in val order + 6
  noise-free unit signals (theta0 = 0.4 rad; sags and jump at t0 = 0.3013 s; ramp from 0.2 s);
  rule outputs with default thresholds; trace tolerances = max(10 x observed float32 error, floor).
- **D-033** — 2026-11-01 — *Naive PLL diagnostic* evaluated over OBS (|f_pll - f_ref| > 1 Hz at any
  phasor-rate instant after 0.15 s), which also excludes the PLL start-up transient.

## Review decisions (project lead, after session 1; PROMPT_02), 2026-11-01
- **D-034** — 2026-11-01 — *Oscillation label margin (R1, generator).* For oscillation records the
  analytic LL fundamental magnitudes (|V1 r1_k + V2 r2_k| incl. modulation, growth envelope and
  background unbalance; 2 kHz analytic grid) must stay within [0.92, 1.08] pu for the whole record
  (`margins.osc_ll`; manifest columns `mm_min_ll` / `mm_max_ll`, present since M1, are the realised
  margin metric). A failing draw redraws ONLY the oscillation event (depth, angle amplitude, f_m,
  sigma, and with them t0 / phi_m / phi_rel as for every other class, D-015 i) from the same
  distributions; pre-event conditions are never redrawn. *Why:* single-label classes must be
  mutually exclusive; an oscillation that grows into a dip/swell is a multi-event case (Stage 2).
  Session-1 open question Q1: 124/250 val oscillation records left [0.90, 1.10] and the rules called
  104 of them sag_bal/swell. *Decided BEFORE any test split was evaluated* (no sealed waveform had
  been loaded, plotted or evaluated), so regenerating the sealed splits is legitimate.
  *Effect:* oscillation rejection rate 0.73 (train) / 0.74 (val) / 0.75 (test_iid) / 0.78 (test_shift);
  the realised parameters are conditioned on the band (train: depth median 3.3 % (2-7.2 %), sigma
  median -0.24 /s (-1..1.8), depth x envelope at the record end 1-7.9 %, was up to 34 %).
  *Unchanged-record check:* per-record SHA-1 of X before/after (same platform): identical for every
  non-oscillation record of train/val/preview_iid/preview_shift (full: 4800/4800, 1500/1500, 172/172,
  172/172; --fast likewise) and for every oscillation record accepted at its first draw (271/271,
  77/77, 8/8, 13/13); sealed splits compared by manifest rows (all 1500 + 1500 non-oscillation rows
  identical), not by waveforms (rule A1.4; synthesis is a deterministic function of the manifest
  row, test_determinism). *Alternatives:* (a) keep (a standards-based monitor would also report the
  dip/swell); (c) an oscillation check inside the dip/swell rule branch.
- **D-035** — 2026-11-01 — *Phase jumps that transiently cross 0.90/1.10 in Urms(1/2)* (5/250 val:
  a one-cycle window spanning the jump with U_pre near the band edge) are accepted as physical
  behaviour of Urms(1/2) and documented as a known baseline limitation; no change. *Alternative:*
  a minimum dip/swell duration of two refreshes (changes the IEC-style detector).
- **D-036** — 2026-11-01 — *Frozen rules configuration for M4 = the TRAIN-tuned thresholds*, re-tuned
  after R1 (configs/frozen_methods.yaml). After R1 the tuning keeps every default (rho_th 0.25,
  osc_mag_pct 1.0, osc_ang_deg 1.0, osc_prom 4.0, jump_deg 5.0, df_hz 0.2; train macro-F1 0.9975;
  dip/swell 0.90/1.10 fixed [STD]).
- **D-037** — 2026-11-01 — *DSOGI fixed-w0 100/300 Hz ripple in f_pll accepted for Stage 1* (the rules
  use f_dft; f_pll only enters the naive-PLL diagnostic and two ML features). A frequency-locked loop
  is a Stage 2 option.
- **D-038** — 2026-11-01 — *Platform-stable AA FIR file.* A redesign on Linux (this session) differs
  from the committed Windows coefficients in the last bit of 3 taps (libm), which changes the config
  hash. make_dataset now keeps configs/aa_fir.json when the redesign matches it to 1e-12 (rewrites
  otherwise). The committed file now holds the Linux-designed values (the R1 data use them). The
  int16 inputs of all 27 golden vectors were bit-identical between the Windows (session 1) and Linux
  runs; trace values differed by <= 3e-14 (float64 libm). *Alternative:* hard-code coefficients.
- **D-039** — 2026-11-01 — *Infeasible oscillation band (D-034 edge case).* For a few pre-event
  draws (U_pre near 1.04 with background unbalance near 2.5 %, shift config) even the smallest
  allowed oscillation (depth 2 %) leaves [0.92, 1.08]; no event redraw can pass. Then the
  least-violating of the max_redraws + 1 draws is kept (pre-event unchanged, so no class-dependent
  pre-event distribution) and flagged with n_reject = max_redraws + 1 (meta `osc_band_fallback`).
  Occurs 0 times in the full splits (every oscillation record passes within <= 51 redraws) and once
  in the --fast test_shift subset; such records still stay inside (0.90, 1.10) (tested).
  *Alternatives:* redraw the pre-event (leaks class information into pre-event conditions); allow
  depth < 2 % (violates B5); raise an error (breaks --fast).

## M3 (gradient boosting) decisions
- **D-040** — 2026-11-01 — *ML input mapping of non-finite features:* x = isfinite(f) ? f : 0.0 for every
  model (GBM-embedded, GBM-unconstrained, C walker; `input_nan_fill` in the model JSON). Only
  `rocof_end` is ever NaN in full records (jump mask covering the whole END window: 32/5600 train,
  12/1750 val). The feature code is unchanged (PROMPT_02: "features unchanged"). *Alternatives:* a
  fallback in the feature definition (like df_end; changes M2 code, spec and golden vectors); native
  NaN handling (HGB only; complicates the C walker).
- **D-041** — 2026-11-01 — *GBM-embedded grid via stages.* For each (max_leaf_nodes, learning_rate,
  subsample, seed) one 100-stage fit is made and n_estimators 20/30/40/60 are its first n stages:
  sklearn draws the per-stage subsample and tree randomness from one stream in stage order, so the
  truncated model IS the model trained with n_estimators = n (bit-identical decision function,
  tested). 36 fits instead of 180. Fixed: max_depth = None (trees bounded by max_leaf_nodes only),
  min_samples_leaf 1, log-loss. Seeds = sklearn random_state 0, 1, 2.
- **D-042** — 2026-11-01 — *Export format details (docs/MODEL_FORMAT.md).* Besides 14 B/node and 4 B/class
  init scores the walker needs the tree roots: uint16 tree_root[T] (2 B/tree) is counted in the size.
  Leaves use the same 14-byte stride (fixed-stride arrays, no per-node type tag); internal nodes
  store value 0. Class order = gridmon.CLASSES; label = first argmax of the float32 raw scores.
- **D-043** — 2026-11-01 — *Selection details.* A configuration is feasible if the exported size of
  EVERY seed is <= 65,536 B (max over seeds); among feasible ones the highest mean val macro-F1 wins;
  configurations within 0.002 of it are ties, resolved by the smallest max size, then higher F1,
  fewer stages, fewer leaves. Result: val is saturated (33 feasible configurations within 0.002 of
  1.000), so the rule picks the smallest: 20 stages x 4 leaves, lr 0.1, subsample 1.0, 13,972 B.
- **D-044** — 2026-11-01 — *GBM-unconstrained reference:* HistGradientBoostingClassifier, max_iter 1000,
  early stopping on the log-loss of an internal 15 % split of TRAIN (n_iter_no_change 10), grid
  max_leaf_nodes {15, 31} x learning_rate {0.05, 0.1}; highest mean val macro-F1, ties -> smaller
  (all four tie at 0.9998 -> 15 leaves, lr 0.1). Its size in the same 14 B/node format is counted
  from sklearn's private `_predictors` (accounting only; not exported).
- **D-045** — 2026-11-01 — *Committed model artefacts (models/):* the three GBM-embedded JSON models
  (seed 0 = models/gbm_embedded.json = the C model), their sklearn pickles (cross-check of the walker
  on the test sets) and the three GBM-unconstrained pickles (~1 MB in total; needed by M4 because the
  frozen methods are evaluated from files, not retrained). Pickles are tied to scikit-learn 1.9.1.
- **D-046** — 2026-11-01 — *M3 controls and importance:* the pre-event-only control refits the SELECTED
  GBM-embedded configuration (3 seeds) on features of the first 750 samples (OBS = REF, D-031) with the
  D-040 mapping (all-NaN columns become constant). Permutation importance: sklearn
  permutation_importance on VAL, seed-0 model, macro-F1 scoring, 10 repeats, random_state 0.

## M4 (sealed evaluation) decisions
- **D-047** — 2026-11-01 — *Freeze mechanics and pre-registered evaluation plan.* configs/frozen_methods.yaml
  (scripts/freeze_methods.py, from the committed M2/M3 reports only) records the rule thresholds, the
  GBM configurations and seeds, the SHA-256 of the 6 model files and of the 16 method-defining files
  (dsp/*, features, rules, units, ml_gbm, export_gbm, configs/baseline.yaml, configs/ml.yaml, the C
  tables), the feature-code commit and the evaluation plan. scripts/evaluate_test.py (the only caller
  with allow_test, tested) exits with code 3 before loading any test data unless gridmon.frozen.verify
  passes: hashes match, no unlisted method file, and no difference from the M4-freeze commit (git).
  The evaluation code itself was dry-run on VAL before the freeze (no test data). After the single
  run, RESULTS.md is re-rendered from results.json (`--render-only`, loads no data) to add findings.
  Full-mode run_gbm keeps frozen models (no retraining without --force).
  *Evaluation conventions (fixed before the run):* GBM-embedded is evaluated through the float32
  walker on the exported JSON (the deployable model; walker-vs-sklearn agreement is reported); ML =
  mean +/- sd over seeds 0/1/2; confusion matrices and figures use seed 0; paired bootstrap of the
  seed-mean macro-F1 difference (2,000 class-stratified resamples, seed 20261104) with per-seed CIs;
  exact McNemar per seed; GBM-unc vs GBM-emb uses GBM-emb seed 0 as the reference; SNR bins
  25/30/35/40/50/60 dB; severity bins in gridmon.evaluation.SEVERITY; iid -> shift change with
  independent stratified resamples. "Better" is claimed only if the 95 % CI excludes zero.
