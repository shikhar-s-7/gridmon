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
