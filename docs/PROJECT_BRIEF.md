# gridmon — Project Brief v0.2 (Stage 1), 2026-11-01

## B1. Purpose and question
Build a reproducible, offline monitor that detects and classifies grid events in triggered
three-phase voltage records at the AC terminals of a BESS PCS, and measure honestly where a learned
classifier helps or fails relative to a conventional, standards-based signal-processing baseline.
Make the monitor portable: one C99 core that is verified against the Python reference, compiles for
TI C2000 (C28x) microcontrollers as a monitor-only module, and runs in a static website via
WebAssembly.

Primary question: on physics-based synthetic records, how do (1) transparent rules and
(2) gradient boosting on the same physics features compare in accuracy, false alarms,
safety-critical confusions, robustness to shift, and embedded cost? (A raw-waveform 1-D CNN is an
optional third method.) No outcome is presumed; "the rules win" is an acceptable result.

## B2. Scope (Stage 1)
In: synthetic data generator; conventional DSP baseline + rules; gradient boosting with an
embedded size budget; evaluation on sealed iid and shifted test sets; portable C core with
software-in-the-loop (SIL) equivalence tests; C2000 port layer, integration notes and a C28x
compile/size check; static website (results, record explorer, in-browser classification).
Optional if time allows: 1-D CNN on raw waveforms (offline comparison only).
Out (Stage 2+): any run on real hardware (LaunchPad processor-in-the-loop, DAC-to-ADC replay, PCS
boards); single-phase variant for TIDM-HV-1PH-DCAC; CNN on an NPU device (F28P55x); measured data;
converter/PCS dynamics; weak grids; closed-loop control; VSG; multi-event records; streaming
latency optimisation.
Nothing from this project may be used in control or protection without separate validation
(hardware-in-the-loop, then field). The C2000 module is monitor-only.

## B3. Evidence tags (use in all docs)
[STD] standard/regulatory document | [LIT] literature or incident report | [ASSUMED] illustrative
engineering assumption | [SIM] finding from this project's simulations | [EXP] experimentally
validated (none exist in Stage 1).

## B4. System boundary and measurement
- Grid: 50 Hz, NEM-like. A voltage source whose fundamental sequence phasors and frequency follow
  the event models. The PCS is a passive observer: its current does not influence the PCC voltage
  (strong-grid assumption) [ASSUMED].
- Measurement: three LL voltages v_ab, v_bc, v_ca at the PCS LV terminals (3-wire), so zero
  sequence is unobservable [ASSUMED]. If hardware senses phase-to-neutral, firmware forms LL
  differences first. Bollen types B and E therefore appear as D and G (verified in tests).
- Sampling: synthesise at 20 kHz (stand-in for a PCS control-loop ADC rate); causal linear-phase
  FIR anti-alias filter (passband >= 1.0 kHz, >= 40 dB attenuation at >= 2.5 kHz, gain exactly 1 at
  50 Hz); decimate x4 to the monitor rate fs = 5 kHz (N = 100 samples per nominal cycle).
  5 kHz is an integer divisor of common C2000 control rates (10, 20, 40, 50, 100 kHz), so firmware
  can decimate exactly [ASSUMED; TIDA-01606 SDK uses 50 kHz PWM]. Coefficients saved in
  configs/aa_fir.json; group delay in meta.json.
- Quantisation and storage: 12-bit over +/- 2 pu (LSB = 4/4096 pu), stored as int16 ADC counts —
  the same representation the MCU sees.
- Records: 1.0 s (5,000 samples x 3 channels), as from a triggered recorder with >= 200 ms
  pre-trigger. Event onset t0 ~ U[0.20, 0.40] s, continuous (random point-on-wave). Methods may
  assume the first 0.15 s (750 samples) of every record is pre-event; they may not use t0.

## B5. Classes (single label per record) and label margins
| label | meaning / typical cause | PCS response it would drive (context) |
|---|---|---|
| normal | no event; realistic background + small "hard-negative" changes | none |
| sag_bal | balanced dip, Bollen type A (three-phase fault) | LVRT, positive-sequence reactive current |
| sag_unbal | unbalanced dip, types C/D/F/G as seen on a 3-wire LL measurement | LVRT with negative-sequence handling |
| swell | balanced overvoltage (load rejection, capacitor switching) | HVRT, absorb reactive power |
| phase_jump | step in positive-sequence angle without a large magnitude change | ride through; must NOT be read as a frequency event |
| freq_event | sustained frequency ramp/step (generation or load contingency) | frequency response |
| oscillation | sustained low-frequency modulation of fundamental magnitude/angle | alarm / damping |

Label margins (rejection sampling; computed analytically from fundamental phasors incl. background;
rejection redraws ONLY event or hard-negative parameters, never pre-event conditions):
- sag_bal, sag_unbal: min over LL channels of the fundamental magnitude during the event <= 0.87 pu.
- swell: max LL fundamental magnitude during the event >= 1.13 pu.
- normal: all LL fundamental magnitudes within [0.92, 1.08] pu at all times; angle steps <= 2 deg;
  frequency drift <= 0.03 Hz/s.
- phase_jump: |dphi| >= 6 deg; |V1| change <= 3 %.
- freq_event: |f(record end) - f_pre| >= 0.25 Hz.
- oscillation: magnitude-modulation depth parameter >= 2 % (by construction).
The 0.9/1.1 pu dip/swell thresholds follow IEC 61000-4-30 practice and IEEE 1159 categories [STD];
the ~3 % margins avoid label noise from threshold-straddling cases [ASSUMED].

## B6. Generator
### B6.1 Construction
Per record: piecewise-defined fundamental sequence phasors V1(t), V2(t) (complex pu, relative to a
rotating reference theta(t)); frequency trajectory f(t); theta(t) = theta0 + 2*pi*integral(f dt),
theta0 ~ U[0, 2*pi). Synthesis at 20 kHz.
Phase voltages (zero-sequence-free):
  v_k(t) = Re{ (V1(t)*r1_k + V2(t)*r2_k) * exp(j*theta(t)) } + harmonics,
  r1 = [1, a^2, a], r2 = [1, a, a^2], a = exp(j*2*pi/3), k = 0, 1, 2 for phases a, b, c.
Harmonic h: v_k,h = A_h * cos(h*(theta(t) - 2*pi*k/3) + phi_h)
  (5th, 11th negative sequence; 7th, 13th positive sequence; frequencies track f(t)).
LL outputs: v_ab = (v_a - v_b)/sqrt(3), v_bc = (v_b - v_c)/sqrt(3), v_ca = (v_c - v_a)/sqrt(3).

### B6.2 Pre-event and nuisance conditions (same code path for all classes; iid splits)
- U_pre = |V1| pre-event ~ U[0.96, 1.04]; background |V2|/|V1| ~ U[0, 1.5 %], random angle.
- Harmonics (% of fundamental): h5 U[0,3], h7 U[0,2], h11 U[0,1], h13 U[0,1]; random phases.
- f_pre ~ U[49.9, 50.1] Hz, drift ~ U[-0.03, 0.03] Hz/s (context: NEM normal operating frequency
  band 49.85-50.15 Hz [STD]).
- Sensor gain error per channel ~ U[-0.5, 0.5] %; offset ~ U[-0.5, 0.5] % of nominal peak.
- White noise, SNR ~ U[35, 60] dB relative to nominal LL RMS, added after decimation.
- 12-bit quantisation over +/- 2 pu, stored as int16 counts.
- Normal-class hard negatives (independent, each at a random time >= 0.2 s):
  |V1| step +/-U[0.5, 2] % (p = 0.3); angle step +/-U[0.5, 2] deg (p = 0.2);
  harmonic-level step, add U[0.5, 2] % to h5 or h7 (p = 0.2);
  slow |V1| ramp +/-U[0.5, 1.5] % over U[0.2, 0.5] s (p = 0.1).

### B6.3 Event models
Closed forms (psi = 0, relative to pre-event, phase a special) [LIT: Bollen; Zhang & Bollen]:
- A: V1 = V, V2 = 0.
- C: V1 = (1+V)/2, V2 = (1-V)/2.      D: V1 = (1+V)/2, V2 = -(1-V)/2.
- F: V1 = (1+2V)/3, V2 = -(1-V)/3.    G: V1 = (1+2V)/3, V2 = (1-V)/3.
Phase-angle jump: replace V by V* = V*exp(j*psi). Special phase b or c: multiply V2 by a or a^2.
Multiply the result by the pre-event V1 phasor.

- sag_bal: V ~ U[0.10, 0.85]; psi ~ U[-30, 0] deg; background V2 scales with V1; duration
  ~ U[0.06, 0.50] s (may run past record end); recovery 50 % abrupt, 50 % to U[0.90, 0.98]*U_pre
  then exponential to U_pre with tau ~ U[0.05, 0.30] s. Real unbalanced recovery happens in steps
  at current zeros; simplified here [ASSUMED; LIT: Bollen 2003].
- sag_unbal: family CD or FG (p = 0.5); sign and special phase uniform over the 6 combinations;
  V ~ U[0.0, 0.85]; psi ~ U[-30, 0] deg; background V2 added; duration and recovery as sag_bal.
  Store sag type and special phase in the manifest.
- swell: |V1| -> S*U_pre with S ~ U[1.10, 1.30] (rejection to meet the 1.13 margin); angle step
  +/-U[0, 3] deg; extra |V2| <= 1 %; duration ~ U[0.05, 0.60] s; abrupt recovery.
- phase_jump: angle step dphi = +/-U[6, 30] deg (random sign); |V1| step U[-3, +3] %; persists.
  Anchored on the IEEE 2800-2022 25 deg positive-sequence phase-jump ride-through requirement [STD].
- freq_event: ROCOF r = +/-U[0.5, 4.0] Hz/s (60 % negative); ramp until |df| reaches U[0.3, 1.5] Hz,
  then hold; |V1| change <= 1 %. Anchors: NEM frequency operating standard ROCOF limits 1 Hz/s
  (credible contingency, over 500 ms) and 3 Hz/s (severe events, reasonable endeavours, over
  300 ms); IEEE 2800 5 Hz/s ride-through [STD].
- oscillation: modulation of |V1| (depth U[2, 8] %) and of angle (amplitude U[0, 4] deg),
  modulation frequency f_m ~ U[5, 25] Hz, envelope exp(sigma*(t - t0)) with sigma ~ U[-1, 2] 1/s,
  smooth 50 ms ramp-in, persists. Motivated by sub-synchronous oscillations in IBR-dense networks,
  which AEMO characterises as 5-40 Hz [LIT]; mapping to a modulation frequency is frame-dependent,
  so the range is [ASSUMED].

### B6.4 Measurement chain
synthesise at 20 kHz -> per-channel gain/offset -> FIR anti-alias -> decimate x4 to 5 kHz
-> add noise -> quantise to 12 bits -> store int16 counts.

### B6.5 Splits and seeds
| split | records/class | total | seed | config |
|---|---|---|---|---|
| train | 800 | 5,600 | 1001 | iid |
| val | 250 | 1,750 | 2002 | iid |
| test_iid | 250 | 1,750 | 3003 | iid (sealed) |
| test_shift | 250 | 1,750 | 4004 | shift (sealed) |
| preview | 200 total each | - | 9009 / 9010 | iid / shift, figures only |
Per-record seeds: SeedSequence(split_seed).spawn(N). Classes balanced, order shuffled.
About 0.33 GB of int16 in total (data/ is git-ignored).

### B6.6 Shift configuration (test_shift)
SNR U[25, 35] dB; f_pre U[49.5, 50.5] Hz; harmonics h5 U[0,5], h7 U[0,4], h11 U[0,2],
h13 U[0,2] %; background unbalance U[0, 2.5] %; gain U[-2, 2] %; offset U[-2, 2] %.
Plus an unseen nuisance in 30 % of records (any class, label unchanged): damped
capacitor-switching-like oscillatory transient, 300-900 Hz, initial amplitude U[0.2, 0.6] pu,
tau U[3, 15] ms, random time in [0.15, 0.95] s [ASSUMED; IEEE 1159 low-frequency oscillatory
transient category for context]. Event parameters unchanged; margins still enforced.

## B7. Conventional baseline (M2) — port-exact definitions
Input: int16 counts x_c[n], c in {ab, bc, ca}, n = 0..4999; v_c[n] = LSB * x_c[n] (pu).
N = 100 samples per nominal cycle; Ts = 1/5000 s.
### B7.1 DSP chain
1. Urms(1/2): window of N samples refreshed every N/2; value k uses samples [50k, 50k+99]
   (first value at n = 99); RMS_pu = sqrt(2 * mean(v^2)) per channel. Fixed grid (IEC 61000-4-30
   aligns to zero crossings; simplification logged) [STD-inspired].
2. Phasors at 1 kHz (every 5th sample, n = 5m + 4, valid for n >= 99): non-recursive one-cycle DFT
   X_c(m) = (2/N) * sum_{i=0}^{N-1} v_c[n-i] * W[(n-i) mod N], W[k] = exp(-j*2*pi*k/N)
   (twiddle table; no recursive sliding DFT, which drifts in float32).
   Sequence components from LL phasors: V1 = (Xab + a*Xbc + a^2*Xca)/3, V2 = (Xab + a^2*Xbc + a*Xca)/3.
3. Frequency: f_dft(m) = 50 + wrap(angV1(m) - angV1(m-20)) / (2*pi*0.02), then a 20-sample
   (one-cycle) moving average. ROCOF(m) = least-squares slope of f_dft over the last 100 phasor
   samples (0.1 s), implemented with running sums.
4. Jump metric (second difference with one-cycle lags; insensitive to frequency offset and ramps):
   J(m) = wrap(angV1(m) - 2*angV1(m-20) + angV1(m-40)). A step dphi gives a +dphi/-dphi pair;
   a 4 Hz/s ramp gives about 0.58 deg.
5. DSOGI-PLL at 5 kHz [LIT: Rodriguez et al. 2006]: alpha-beta from LL: v_alpha = (v_ab - v_ca)/sqrt(3),
   v_beta = v_bc. SOGI quadrature-signal generators on alpha and beta: transfer functions
   D(s) = k*w*s/(s^2 + k*w*s + w^2), Q(s) = k*w^2/(s^2 + k*w*s + w^2), k = sqrt(2), w = 2*pi*50
   fixed (frequency-locked loop is a later extension), discretised with Tustin (bilinear).
   Positive-sequence calculator: v_alpha+ = (v_alpha' - qv_beta')/2, v_beta+ = (qv_alpha' + v_beta')/2.
   SRF-PLL: e = (-v_alpha+ * sin(th) + v_beta+ * cos(th)) / max(|v+|, 0.05); PI (Tustin) tuned for
   wn = 2*pi*20 rad/s, zeta = 0.707; w_hat = 2*pi*50 + PI output; th += w_hat*Ts (wrapped).
   f_pll = w_hat/(2*pi), kept at the phasor rate (every 5th sample) for features and golden vectors.
   Document coefficient formulas in DSP_SPEC.md. (TI's C2000Ware DigitalPower SDK has SPLL
   modules, e.g. SPLL_3PH_SRF/DDSRF; firmware may later swap to them only if equivalence is re-shown.)
6. Envelope: e_mag(p) = mean |V1| over phasor samples [10p, 10p+9] (100 Hz); e_ang(p) = mean of
   unwrapped angV1 minus the reference-frequency ramp 2*pi*(f_ref - 50)*t. Subtract a centred
   21-sample running median (edges truncated). Spectra over 50-sample Hann windows (0.5 s), hop
   10 (0.1 s), by direct DFT at 3, 4, ..., 30 Hz, scaled so a sinusoid of amplitude A peaks at A.
   Report peak amplitude, peak frequency and prominence (peak / band median) for both signals.
7. THD: 500-sample rectangular windows (5 cycles), hop 500; per channel DFT at harmonic bins
   k = 5h, h = 1..15; THD = sqrt(sum_{h=2..15} |X_h|^2) / |X_1|; mean over channels per window.
### B7.2 Features (one row per record; phase-permutation invariant: sort per-channel statistics)
Reference = median over phasor samples with n <= 749 (first 0.15 s). Include at least: sorted min
and max Urms(1/2) (3 + 3); time below 0.90 and above 1.10; |V1| min and max relative to reference;
|V2|/|V1| reference and max; Bollen ratio rho = |V2(m*) - V2_ref| / (|V1_ref| - |V1(m*)|) at
m* = argmin |V1| (denominator floored at 0.01); max |J| and its sign; f_ref; f_end - f_ref
(f_end = mean f_dft over the last 0.1 s); max |f_dft - f_ref|; max |ROCOF|; time with
|ROCOF| > 0.2 Hz/s; max |f_pll - f_ref| and time with |f_pll - f_ref| > 0.5 Hz; envelope peak
amplitude, frequency and prominence (magnitude and angle); THD first window, last window and max.
Aim for 30-45 features, each with a documented float32 tolerance for the C port.
### B7.3 Rule classifier (priority order; returns label, reason code, onset estimate)
1. Dip: any Urms(1/2) < 0.90 for >= one refresh (ends when all >= 0.92) [STD]
   -> rho > rho_th (default 0.25) ? sag_unbal : sag_bal.
   (With psi = 0: rho ~ 0 for type A, ~ 0.5 for F/G, ~ 1 for C/D, independent of depth;
   phase-angle jumps push unbalanced cases higher, never towards 0.)
2. Swell: any Urms(1/2) > 1.10 (ends when all <= 1.08) [STD] -> swell.
3. Oscillation: envelope peak > 1.0 % (magnitude) or > 1.0 deg (angle) with prominence >= 4
   [ASSUMED] -> oscillation.
4. Phase jump: max |J| > 5 deg with |V1| change < 10 % [ASSUMED] -> phase_jump.
5. Frequency: |f_end - f_ref| > 0.2 Hz [ASSUMED] -> freq_event. Frequency features exclude
   +/- 2 cycles around a detected jump (TSO practice: filter phase jumps out of ROCOF).
6. Otherwise normal.
Order rationale: magnitude events dominate ride-through; oscillation precedes jump because angle
modulation can exceed the jump threshold; jump precedes frequency because a jump produces a
transient frequency estimate. Tunable on TRAIN: rho_th, oscillation thresholds, jump threshold,
df threshold. Dip/swell thresholds stay at standard values.
### B7.4 Diagnostics
- Pre-event-only control: features from the first 0.15 s only; fit sklearn
  HistGradientBoostingClassifier on train, evaluate on val. Expect macro-F1 ~ 1/7 ~ 0.14.
  > 0.25 means the generator leaks class information: stop and fix.
- Naive PLL frequency rule (the Blue Cut Fire failure mode [LIT: NERC 2017]): flag "frequency
  event" if |f_pll - f_ref| > 1.0 Hz at any instant. Report false-flag rates for sag_bal,
  sag_unbal, phase_jump and normal, and versus jump size (6-12, 12-20, 20-30 deg).
### B7.5 Golden vectors (for the C port)
golden/: 21 val records (3 per class) + 6 unit signals (clean nominal, balanced 0.5 pu dip, type C
dip, 20 deg jump, 1 Hz/s ramp, 10 Hz modulation). Each .npz holds: int16 input; Urms(1/2); V1 and
V2 at the phasor rate; f_dft; ROCOF; J; f_pll at the phasor rate; envelope spectra; THD per window;
the feature vector; rule label and reason code. golden/README.md documents array names, shapes,
units and tolerances. Total <= 10 MB.
### B7.6 Embedded budget estimate
Count multiply-accumulates (or equivalent operations) per 1 s record per block, and working memory
per block; expected order of magnitude about 1.5 M operations and <= 32 KB working state besides
the 30 KB int16 record. Write reports/m2_embedded_budget.json; flag anything that does not fit.

## B8. ML (M3 — reference only, not this session)
- Gradient boosting: sklearn HistGradientBoostingClassifier on B7.2 features; small grid; early
  stopping on an internal train split; select on val; 3 seeds. Embedded budget: model tables
  <= 64 KB (float32 thresholds and leaf values, int16 indices). Report macro-F1 versus model size.
  Export to C with a project-owned exporter (generated const arrays + generic tree walker);
  equivalence: max |proba difference| <= 1e-3 and label agreement >= 99.5 % on val.
- Optional 1-D CNN (offline comparison only): PyTorch CPU; input [3, 5000] pu LL waveforms; 5-6
  Conv1d blocks, global average pooling, linear(7); 0.1-0.3 M params; 3 seeds. An embedded CNN is
  Stage 2 (TI NPU device F28P55x via TI's Tiny ML ModelZoo / neural network compiler).
- Same pre-event-only control for every model.

## B9. Evaluation (M4 — reference only)
Run once on sealed test_iid and test_shift with frozen methods; log the git hash. Metrics:
macro-F1 (primary); per-class P/R/F1; confusion matrices; false-alarm rate on normal; miss rate;
critical confusions; breakdowns by SNR, event severity and shift; paired bootstrap 95 % CIs (2,000
resamples) for differences between methods; McNemar test; mean +/- sd over 3 seeds for ML; model
size vs macro-F1; inference cost per record (Python indicative; C2000 cycles only after Stage 2 PIL).
C-core agreement with Python on both test sets is reported in M6.

## B10. Definition of done (Stage 1)
1. Fresh clone -> venv -> python scripts/run_all.py regenerates data, runs baseline and ML,
   evaluates and writes reports; --fast completes in < 10 min.
2. pytest passes (generator, DSP, leakage, determinism, float32 robustness, C SIL equivalence).
3. reports/RESULTS.md + results.json: rules vs gradient boosting (+ CNN if done) on test_iid and
   test_shift with CIs, confusions, breakdowns, size-vs-F1, pre-event-only control, naive-PLL
   diagnostic, and a "Findings and limitations" section using evidence tags.
4. Portable C core (c/): SIL equivalence on val, test_iid and test_shift (rule labels >= 99.5 %
   agreement, features within documented tolerances, GBM as in B8); compiles for C28x with TI's
   C2000 compiler with a code/data size report (or a documented blocker).
5. firmware/c2000/: monitor-only port layer and integration guide (TIDA-01606 first; notes for
   TIDM-HV-1PH-DCAC); docs/VALIDATION.md shows ladder status (L0-L2 done in Stage 1).
6. web/: static site that runs locally and is deployable to GitHub Pages: overview, results,
   record explorer, in-browser classification (C core via WebAssembly, or a documented fallback).
7. Docs current: brief, DSP_SPEC, FEATURES, decisions, progress, handoff, validation; README and
   website carry "simulation-only, not validated on measured data, not for protection or control".
Success is a trustworthy, reproducible measurement and a portable monitor, not a particular winner.

## B11. Initial decisions (seed docs/DECISIONS.md)
- D-001 Offline record-level classification of triggered records, not real-time control.
- D-002 Physics-based synthetic generator (sequence phasors + Bollen dip types); EMT/Simulink
  cross-check later.
- D-003 3-wire LL measurement; zero sequence unobservable.
- D-004 Seven classes mapped to distinct PCS responses; single-label records; label margins.
- D-005 Fair comparison: the same features for rules and gradient boosting; tuning on train/val
  only; sealed test sets.
- D-006 Shifted test set with an unseen nuisance, to measure robustness rather than iid accuracy.
- D-007 (v0.2) Monitor rate 5 kHz (100 samples/cycle) from a 20 kHz synthesis: exact integer
  decimation from common C2000 control rates; replaces 3.2 kHz.
- D-008 (v0.2) int16 12-bit ADC-count storage: matches what the MCU sees; halves dataset size.
- D-009 (v0.2) Port-exact DSP definitions (non-recursive DFT at 1 kHz, bin DFTs, Tustin
  discretisation, bounded memory) plus golden vectors, so a C core can be verified without hardware.
- D-010 (v0.2) One portable C99 float32 core, three targets: PC (SIL), C2000 (monitor-only module),
  browser (WebAssembly). Three-phase TIDA-01606-class hardware first.
- D-011 (v0.2) Gradient boosting is the embedded ML component (<= 64 KB); the CNN becomes optional
  offline, with an embedded CNN deferred to an NPU device (Stage 2).

## B12. References
1. IEC 61000-4-30:2015, Power quality measurement methods (Urms(1/2); dip/swell detection with hysteresis).
2. IEEE Std 1159-2019, Recommended Practice for Monitoring Electric Power Quality.
3. M. H. J. Bollen, Understanding Power Quality Problems, IEEE Press, 2000; M. H. J. Bollen and
   L. D. Zhang, "Different methods for classification of three-phase unbalanced voltage dips due to
   faults," Electric Power Systems Research, 2003; M. H. J. Bollen, "Voltage recovery after
   unbalanced and balanced voltage dips in three-phase systems," IEEE Trans. Power Delivery 18(4), 2003.
4. P. Rodriguez et al., "New positive-sequence voltage detector for grid synchronization of power
   converters under faulty grid conditions," IEEE PESC 2006.
5. NERC/WECC, 1,200 MW Fault Induced Solar Photovoltaic Resource Interruption Disturbance Report
   (Blue Cut Fire), June 2017.
6. AEMC Reliability Panel, Frequency Operating Standard, effective 9 October 2023.
7. IEEE Std 2800-2022 (IBR interconnection: 25 deg phase-jump and 5 Hz/s ROCOF ride-through).
8. Energinet, Guidelines for validation of RoCoF, 2024 (filter phase jumps out of ROCOF).
9. AEMO, Analysis of Sub-synchronous Oscillations in West Murray Zone Power System in Australia, 2025.
10. TI, TIDA-01606 bidirectional three-phase T-type inverter and PFC reference design (product page
    and C2000Ware DigitalPower SDK guide: F2837xD / F280039C / F28P65x; 50 kHz PWM in the SDK).
11. TI, TIDM-HV-1PH-DCAC single-phase inverter reference design and TIEVM-HV-1PH-DCAC kit
    (F2837xD / F28004x; 380 V DC in; 220 V 50 Hz or 110 V 60 Hz out).
12. TI, C2000Ware DigitalPower SDK software PLL API (SPLL_3PH_SRF, SPLL_3PH_DDSRF, SPLL_1PH_SOGI).
13. TI, C2000 code generation tools (C2000-CGT), free for Windows, Linux and macOS.
14. TI, Tiny ML ModelZoo / Edge AI tools for C2000, and TMS320F28P55x with integrated NPU.
15. TI, LAUNCHXL-F28379D LaunchPad (dual C28x 200 MHz, CLAs, 12-bit DACs, isolated XDS100v2).

## B13. Embedded and web architecture (context for M6-M8; not this session)
### B13.1 Portable C core (M6)
c/include/gridmon.h, c/src/*.c: C99, float only (no double: build with -Wdouble-promotion
-Wfloat-conversion), no malloc, no stdio in the core, static buffers, deterministic. API sketch:
gm_init(ctx, cfg); gm_process_record(ctx, const int16_t *x /*3 x 5000, planar*/, n, lsb, &out)
-> out.features[], out.rule_label, out.rule_reason, out.onset_estimate, out.gbm_proba[7],
out.gbm_label; optional traces under GM_TRACES for SIL. Generated tables: c/src/gm_model_gbm.c,
c/src/gm_config.h. SIL harness: Python ctypes against golden vectors and full val/test splits.
Native build: system C compiler, or `pip install ziglang` and `python -m ziglang cc` if none exists.
### B13.2 C2000 port (M7)
firmware/c2000/: gm_port_c2000.{h,c} — decimator from the control-loop ADC rate to 5 kHz using
configs/aa_fir.json (polyphase), int16 ring buffer, simple real-time trigger (Urms(1/2) and |J|),
record capture (0.2 s pre, 0.8 s post) and background processing; result to telemetry (SCI/CAN).
Rules: monitor-only (never writes control variables); the control ISR does only the decimator push;
record processing runs in the background loop, a low-priority interrupt, or on CPU2 of dual-core
F2837xD devices. Integration guide: TIDA-01606 (three-phase; its labs use TI's SRF PLL; form LL
from sensed voltages; decimate 50 kHz -> 5 kHz by 10); notes for TIDM-HV-1PH-DCAC (single-phase:
needs the Stage 2 1-ph variant). Compile check with TI C2000-CGT (cl2000, C28x, FPU32; TMU where
available; restrict floating point to 32-bit if the compiler supports it): code/data size from the
map file. If the compiler cannot be installed, document the blocker.
### B13.3 Validation ladder (docs/VALIDATION.md)
L0 Python reference [SIM] -> L1 SIL: C core on PC vs golden vectors and splits -> L2 C28x compile and
size -> L3 PIL on LAUNCHXL-F28379D: stored or UART-streamed records, cycle counts, bit-match vs L1
-> L4 signal-level replay: DAC -> ADC loopback on the LaunchPad (two DACs suffice: v_ca = -(v_ab +
v_bc) in a 3-wire system) -> L5 monitor-only on a PCS board at low voltage, supervised -> L6
grid-connected, monitor-only. Stage 1 = L0-L2. L3+ are Stage 2 and need hardware.
### B13.4 Website (M8)
web/: static HTML/JS, no backend. Pages: overview (boundary, assumptions with tags), results (from
reports/results.json), explorer (precomputed traces for about 70 val/test records incl.
disagreements, exported by scripts/export_web.py), try-it (pick a sample or upload a CSV of 3 LL or
3 LN channels with sample rate and nominal voltage; resampled to 5 kHz in the browser; classified by
the C core compiled to WebAssembly; shows label, rule trace, probabilities, key traces), firmware
(budgets, validation-ladder status). Toolchain: Emscripten preferred; fallback zig cc to
wasm32-wasi; last resort Pyodide running the Python reference, or precomputed-only. Deploy with a
GitHub Pages workflow; local preview with python -m http.server. Banner: simulation-only.
### B13.5 Safety
TIDA-01606 (800 V DC nominal) and TIEVM-HV-1PH-DCAC (380 V DC) are high-voltage hardware. Any board
work only under qualified supervision, with isolated debug connections, after L3-L4 pass on a
LaunchPad. The firmware module stays monitor-only until separately validated.
