# DSP specification (port-exact), gridmon v0.2

This document defines every step of the conventional baseline (brief B7) precisely enough for an
independent float32 C99 implementation to reproduce the Python reference (`src/gridmon/dsp/*`,
`features.py`, `baseline_rules.py`) within the tolerances in golden/README.md and docs/FEATURES.md.
Evidence tags: [STD] standard, [LIT] literature, [ASSUMED] engineering choice. Simulation-only.

## 0. Conventions
- Input: `int16_t x[3][5000]` planar, channel c = 0 (v_ab), 1 (v_bc), 2 (v_ca); 12-bit counts.
- Scaling: `v = LSB * (float)x`, LSB = 4/4096 = 2^-10 pu (exact in float32). pu of nominal LL peak.
- Arithmetic: float32 throughout. Constants are computed in double offline and stored as float
  literals (`const` tables). No dynamic allocation, no stdio, no look-ahead beyond the record.
- Record-level processing: the whole 1 s record is available; a block may make a second pass over
  a buffer it stored (e.g. the jump mask before ROCOF). Nothing reads beyond sample 4999.
- `wrap(x) = x - 2*pi*floor((x + pi)/(2*pi))` maps to [-pi, pi).
- `atan2f`, `sinf`, `cosf`, `sqrtf` from the C library (or the C28x TMU). Angle of a complex z is
  `atan2f(Im z, Re z)`.
- Argmax / argmin return the FIRST index attaining the extreme. Medians of an even count are the
  mean of the two middle order statistics.
- NaN in the Python reference marks "not computed": the C core keeps explicit validity ranges.

### Index grids and validity
| grid | rate | index | sample position | valid range (5000-sample record) |
|---|---|---|---|---|
| samples | 5 kHz | n | n | 0..4999 |
| Urms(1/2) | 100 Hz | k | window [50k, 50k+99] | k = 0..98 |
| phasors | 1 kHz | m | n_m = 5m + 4 | m = 19..999 (981 values) |
| f_dft | 1 kHz | m | | m >= 58 |
| J | 1 kHz | m | | m >= 59 |
| ROCOF | 1 kHz | m | | m >= 157 |
| f_pll (stored) | 1 kHz | m | n_m | m = 0..999 |
| envelope | 100 Hz | p | phasors m = 10p..10p+9 | p = 2..99 |
| envelope spectra | - | q | envelope p = 2+10q .. 2+10q+49 | q = 0..4 |
| THD | 10 Hz | w | samples [500w, 500w+499] | w = 0..9 |

Feature windows on the phasor grid: REF = {m >= 19 : n_m <= 749} = m 19..149 (131 values);
OBS = {m : n_m >= 750} = m 150..999; END = m 900..999.

## 1. Urms(1/2) [STD-inspired: IEC 61000-4-30]
For k = 0..98, c = 0..2: `acc = sum_{i=0}^{99} v[c][50k+i]^2`, `U[c][k] = sqrtf(2*acc/100)` (pu RMS).
Fixed grid (no zero-crossing alignment; logged simplification). Window centre time
`t_k = (50k + 50)/5000` s is used for onset estimates.

## 2. One-cycle DFT phasors (non-recursive)
Twiddle table `W[i] = exp(-j 2 pi i / 100)`, i = 0..99 (computed in double, stored as float).
For m = 19..999, n = 5m + 4, each channel:
`X_c(m) = (2/100) * sum_{i=0}^{99} v[c][n-i] * W[(n-i) mod 100]`.
Twiddles are indexed by absolute sample index mod 100, so a 50 Hz sinusoid A cos(2 pi 50 n/fs + phi)
gives the stationary phasor A e^{j phi}. No sliding/recursive DFT (float32 drift).
Sequence components (a = exp(j 2 pi/3) = -0.5 + j sqrt(3)/2):
`V1 = (X0 + a X1 + a^2 X2)/3`, `V2 = (X0 + a^2 X1 + a X2)/3`.
For LL inputs V1 = V1_phase e^{j pi/6}, V2 = V2_phase e^{-j pi/6}: magnitudes equal the phase
sequence magnitudes in pu of nominal LL peak.

## 3. Frequency and ROCOF
`ang(m) = atan2f(Im V1(m), Re V1(m))`.
`df_raw(m) = wrap(ang(m) - ang(m-20)) / (2 pi * 0.02)` for m >= 39 (Hz deviation from 50).
`df(m) = (1/20) * sum_{j=0}^{19} df_raw(m-j)` for m >= 58 (direct 20-term sum); `f_dft = 50 + df`.
All later frequency arithmetic uses the deviation df (float32 resolution).
ROCOF (least-squares slope over the last 100 phasor samples, D-016):
`ROCOF(m) = sum_{j=0}^{99} c_j * df(m-99+j)`, `c_j = (j - 49.5) / (83325 * 0.001)` Hz/s
(83325 = sum_j (j-49.5)^2), for m >= 157. Direct 100-tap sum (no running sums).

## 4. Phase-jump metric
`J(m) = wrap(ang(m) - 2*ang(m-20) + ang(m-40))` for m >= 59 (rad). A step dphi gives J ~ +dphi then
-dphi (one cycle each, smeared by the DFT window); a frequency offset cancels; a ramp r Hz/s
gives 2 pi r 0.02^2 (0.58 deg at 4 Hz/s).

## 5. Jump mask for frequency features (TSO practice [LIT: Energinet 2024])
`mJ = argmax_{m in OBS, m >= 59} |J(m)|`; detected if `|J(mJ)| > 3 deg` (fixed, not tuned).
If detected: f-mask = {m : mJ-40 <= m <= mJ+40} (+/- 2 cycles); ROCOF-mask = {m : mJ-40 <= m <= mJ+139}
(any ROCOF whose 100-sample window touches the f-mask). Masked samples are excluded from the
frequency features (section 9).

## 6. DSOGI-PLL at 5 kHz [LIT: Rodriguez et al. 2006]
alpha-beta from LL: `va = (v_ab - v_ca)/sqrt(3)`, `vb = v_bc` (balanced: va = cos theta, vb = sin theta).
SOGI-QSG, k = sqrt(2), fixed w0 = 2 pi 50 (no FLL). Tustin with prewarping at 50 Hz:
c = 2/Ts = 10000, w = c tan(w0 Ts / 2) = 314.26266 rad/s, a0 = c^2 + k w c + w^2.

| filter | b0 | b1 | b2 | a1 | a2 |
|---|---|---|---|---|---|
| D(s) = k w s/(s^2+k w s+w^2) | k w c/a0 = 0.0425120826 | 0 | -0.0425120826 | 2(w^2-c^2)/a0 = -1.91119707 | (c^2-k w c+w^2)/a0 = 0.914975835 |
| Q(s) = k w^2/(s^2+k w s+w^2) | k w^2/a0 = 0.00133599602 | 0.00267199204 | 0.00133599602 | same | same |

Each of the four filters (D, Q on va and on vb) is a direct-form-II-transposed biquad:
`y = b0 x + s1; s1 = b1 x - a1 y + s2; s2 = b2 x - a2 y`.
Positive sequence: `vap = (Da - Qb)/2`, `vbp = (Qa + Db)/2`.
Error: `e = (-vap sinf(th) + vbp cosf(th)) / fmaxf(sqrtf(vap^2 + vbp^2), 0.05)`.
PI (Tustin), wn = 2 pi 20 rad/s, zeta = 0.707: Kp = 2 zeta wn = 177.68848, Ki = wn^2 = 15791.367;
`u = u + b0 e + b1 e_prev` with b0 = Kp + Ki Ts/2 = 179.267617, b1 = -Kp + Ki Ts/2 = -176.109344.
`w_hat = 2 pi 50 + u`; `f_pll[n] = w_hat/(2 pi)`; `th = wrap(th + w_hat Ts)`.
Order per sample n: filters -> positive sequence -> e (with th[n]) -> PI -> f_pll[n] -> th[n+1].
Initial conditions: all filter states 0, u = 0, e_prev = 0, th[0] = atan2f(vb[0], va[0]).
Stored: f_pll at n = 5m+4 (m = 0..999). Clean-input lock (|f - 50| <= 0.01 Hz) in 68 ms.
(TI SPLL_3PH_SRF/DDSRF may replace this only after equivalence is re-shown.)

## 7. Envelope and its spectra
References from REF (section 9): v1_ref = median |V1|, f_ref = median f_dft.
Unwrapped angle from m = 19: `angU(19) = ang(19)`, `angU(m) = angU(m-1) + wrap(ang(m) - ang(m-1))`.
De-rotated angle: `angD(m) = angU(m) - 2 pi (f_ref - 50) t_m`, t_m = n_m/5000.
Envelope p = 2..99 (10-sample blocks; one block = one 100 Hz period, cancels unbalance ripple):
`e_mag(p) = (1/10) sum_{m=10p}^{10p+9} |V1(m)| / v1_ref`, `e_ang(p) = (1/10) sum angD(m)` (rad).
Detrending (centred 21-sample running median, truncated at the edges):
`r(p) = e(p) - median{ e(q) : max(2, p-10) <= q <= min(99, p+10) }` (11..21 values). Monotone
trends (ramps, steps) are removed; oscillations remain.
Spectra: q = 0..4, window start s_q = 2 + 10q, L = 50 samples (0.5 s), symmetric Hann
`w[i] = 0.5 - 0.5 cos(2 pi i/49)`, scale = 2 / sum(w) (= 2/24.5). For f = 3..30 Hz (28 bins):
`S_q(f) = scale * | sum_{i=0}^{49} w[i] r(s_q+i) E[(f i) mod 100] |`, E[i] = exp(-j 2 pi i/100).
The magnitude spectrum uses 100 r_mag (percent) and degrees(r_ang). The last 8 envelope samples
(p = 92..99) are not covered (D-024).
Peak statistics per signal: q* = window with the largest max_f S; peak = max_f S_{q*};
f_peak = argmax_f S_{q*}; prominence = peak / median_f S_{q*} (28 values; mean of middle two);
gap = (S1 - S2)/S1 with S1 >= S2 the two largest of all 5 x 28 cells (near-tie indicator).

## 8. THD
For w = 0..9, c = 0..2, h = 1..15: `X_h = sum_{j=0}^{499} v[c][500w+j] T[(5 h j) mod 500]`,
T[i] = exp(-j 2 pi i/500) (bin 5h = 50h Hz). `THD_c = sqrtf(sum_{h=2}^{15} |X_h|^2) / |X_1|`;
`THD(w) = mean_c THD_c`. Rectangular window: at off-nominal frequency the fundamental leaks
(THD floor ~ a few % at |f - 50| ~ 0.5 Hz) — part of the feature, not corrected.

## 9. Feature reductions
All feature definitions, units, tolerances and comparison gates: docs/FEATURES.md. Reductions use
the REF / OBS / END windows of section 0. Bollen ratio: V2 is de-rotated by f_ref,
`V2d(m) = V2(m) exp(-j 2 pi (f_ref - 50) t_m)`; V2d_ref = median(Re V2d over REF) + j median(Im V2d over REF);
m* = argmin_{OBS} |V1|; `rho = |V2d(m*) - V2d_ref| / max(v1_ref - |V1(m*)|, 0.01)` (D-023).

## 10. Rule classifier (priority order) — baseline_rules.py
| step | condition (default thresholds) | label | reason code |
|---|---|---|---|
| 1 | urms_min_1 < 0.90 [STD] and rho > rho_th (0.25) | sag_unbal | 2 |
| 1 | urms_min_1 < 0.90 | sag_bal | 1 |
| 2 | urms_max_3 > 1.10 [STD] | swell | 3 |
| 3 | env_mag_amp_pct > 1.0 and env_mag_prom >= 4 [ASSUMED] | oscillation | 4 |
| 3 | env_ang_amp_deg > 1.0 and env_ang_prom >= 4 [ASSUMED] | oscillation | 5 |
| 4 | j_max_deg > 5 and max(abs(v1_min_rel-1), abs(v1_max_rel-1)) < 0.10 [ASSUMED] | phase_jump | 6 |
| 5 | abs(df_end) > 0.2 Hz (jump-masked) [ASSUMED] | freq_event | 7 |
| 6 | otherwise | normal | 0 |

Comparisons with NaN are false. Train-tuned thresholds are listed in reports/m2_baseline_val.json.
Onset estimate = first detection time of the firing detector (aux_t_dip, aux_t_swell,
aux_t_osc_mag/ang, aux_t_jump, aux_t_freq): indicative only, biased late by detector latency.

## 11. Memory and cost (reports/m2_embedded_budget.json)
Per 1 s record: ~1.38 M MAC + 0.12 M other ops + 24 k transcendental calls (~1.52 M total);
streaming working state ~21 KB float32 (phasor-rate buffers |V1|, angV1, |V2|/|V1|, f_dft dominate)
+ 6 KB const tables, besides the 30 KB int16 record. THD (450 k MAC) and the phasor DFT (589 k MAC)
dominate.

## 12. Float32 notes
- Frequencies are carried as deviations from 50 Hz before averaging and slope fitting.
- The unwrapped angle grows to tens of rad in frequency events; float32 rounding then limits the
  angle envelope to ~1e-4 deg — hence the amplitude gate on argmax-type envelope features.
- Python reference sums the DFT in 5-sample blocks then 20 blocks and uses pairwise summation in
  reductions; C uses straight loops. Both stay within the golden tolerances.
- C28x: 16-bit char (sizes in 16-bit words); compile float-only (-Wdouble-promotion); TMU
  accelerates sin/cos/atan2/sqrt/div where present.
