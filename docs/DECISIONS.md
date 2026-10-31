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
