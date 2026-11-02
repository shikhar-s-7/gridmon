# Features (brief B7.2), gridmon v0.2

37 features per record, computed by `gridmon.features` from the int16 waveform only
(no manifest field ever enters). Phase-permutation invariant: per-channel statistics are sorted
(urms_*) or averaged (thd_*); everything else is built from sequence quantities. A cyclic channel
rotation leaves every feature unchanged (tested). DSP definitions: docs/DSP_SPEC.md.

Windows on the phasor grid (n_m = 5m + 4): REF = m 19..149 (n <= 749, first 0.15 s, assumed
pre-event), OBS = m 150..999 (n >= 750), END = m 900..999 (last 0.1 s). Ratios use |V1| floored at
0.05 pu; |V1| and |V2| are in pu of nominal LL peak (sequence magnitudes).

## Feature table
Tolerance = allowed |float32 - float64 reference| (also the C-port SIL criterion). A gate means the
feature is compared only on records where the gate holds (argmax-type features: no near-tie between
candidates, peak above float32 envelope rounding); on other records any value is accepted.

| name | unit | definition | tolerance | gate | single-pass C note |
|---|---|---|---|---|---|
| `urms_min_1` | pu | smallest per-channel min of Urms(1/2) | 1e-05 | - | per-channel running min/max over k; sort 3 values at the end |
| `urms_min_2` | pu | middle per-channel min of Urms(1/2) | 1e-05 | - | per-channel running min/max over k; sort 3 values at the end |
| `urms_min_3` | pu | largest per-channel min of Urms(1/2) | 1e-05 | - | per-channel running min/max over k; sort 3 values at the end |
| `urms_max_1` | pu | smallest per-channel max of Urms(1/2) | 1e-05 | - | per-channel running min/max over k; sort 3 values at the end |
| `urms_max_2` | pu | middle per-channel max of Urms(1/2) | 1e-05 | - | per-channel running min/max over k; sort 3 values at the end |
| `urms_max_3` | pu | largest per-channel max of Urms(1/2) | 1e-05 | - | per-channel running min/max over k; sort 3 values at the end |
| `t_below_090` | s | time with any channel Urms(1/2) < 0.90 (0.01 s per value) | 0.0101 | - | counter incremented per value/sample |
| `t_above_110` | s | time with any channel Urms(1/2) > 1.10 | 0.0101 | - | counter incremented per value/sample |
| `v1_ref` | pu | median abs(V1) over REF | 1e-05 | - | median of 131 stored abs(V1) values (REF buffer, sorted copy) |
| `v1_min_rel` | 1 | min abs(V1) over OBS / v1_ref | 1e-05 | - | running min over OBS |
| `v1_max_rel` | 1 | max abs(V1) over OBS / v1_ref | 1e-05 | - | running max over OBS |
| `v1_end_rel` | 1 | mean abs(V1) over END / v1_ref | 1e-05 | - | running sum over END |
| `u2_ref` | 1 | median abs(V2)/max(abs(V1),0.05) over REF | 1e-05 | - | median of 131 stored ratios |
| `u2_max` | 1 | max abs(V2)/max(abs(V1),0.05) over OBS | 1e-05 | - | running max |
| `u2_end` | 1 | mean abs(V2)/max(abs(V1),0.05) over END | 1e-05 | - | running sum |
| `rho` | 1 | Bollen ratio abs(V2d(m*)-V2d_ref)/max(v1_ref-abs(V1(m*)),0.01), m* = argmin_OBS abs(V1) | 0.0005 | - (SIL: urms_min_1 < 0.95, D-069) | needs f_ref (after REF) and V2 at the running argmin of abs(V1) (store V2(m*) and t_m*) |
| `j_max_deg` | deg | max abs(J) over OBS | 0.001 | - | running max over OBS (J from the angle buffer) |
| `j_sign` | 1 | sign of J at the first OBS sample with abs(J) >= 0.5 maxabs(J) (first lobe = jump sign); compared when j_max_deg >= 1 | 0 | j_max_deg >= 1 | second pass over J for the first sample >= 0.5 max |
| `f_ref` | Hz | median f_dft over REF | 0.0001 | - | median of f_dft over REF (92 valid values) |
| `df_end` | Hz | mean f_dft over END (jump-masked) - f_ref | 0.0001 | - | second pass after the jump mask (f_dft buffer) |
| `df_max` | Hz | max abs(f_dft - f_ref) over OBS (jump-masked) | 0.0001 | - | second pass after the jump mask |
| `rocof_max` | Hz/s | max abs(ROCOF) over OBS (windows touching the jump mask excluded) | 0.001 | - | second pass: 100-tap kernel on the f_dft buffer, skip masked windows |
| `t_rocof_02` | s | time with abs(ROCOF) > 0.2 Hz/s over OBS (0.001 s per sample) | 0.0021 | - | counter incremented per value/sample |
| `rocof_end` | Hz/s | mean ROCOF over END (masked) | 0.001 | - | second pass |
| `dfpll_max` | Hz | max abs(f_pll - f_ref) over OBS | 0.003 | - | running max over OBS (f_ref known from REF before OBS starts) |
| `t_fpll_05` | s | time with abs(f_pll - f_ref) > 0.5 Hz over OBS | 0.0021 | - | counter incremented per value/sample |
| `env_mag_amp_pct` | % | envelope abs(V1) spectral peak amplitude (3-30 Hz) | 0.001 | - | after the phasor pass: 98 block means per signal, running median, 5 x 28 DFT bins |
| `env_mag_freq` | Hz | frequency of that peak; compared when aux_env_mag_gap >= 1e-3 (no near-tie) | 0 | aux_env_mag_gap >= 0.001; env_mag_amp_pct >= 0.01 | after the phasor pass: 98 block means per signal, running median, 5 x 28 DFT bins |
| `env_mag_prom` | 1 | peak / band median of that spectrum; compared when aux_env_mag_gap >= 1e-3 | 0.05 + 0.01 x abs(ref) | aux_env_mag_gap >= 0.001; env_mag_amp_pct >= 0.01 | after the phasor pass: 98 block means per signal, running median, 5 x 28 DFT bins |
| `env_ang_amp_deg` | deg | envelope angle spectral peak amplitude | 0.001 | - | after the phasor pass: 98 block means per signal, running median, 5 x 28 DFT bins |
| `env_ang_freq` | Hz | frequency of that peak; compared when aux_env_ang_gap >= 1e-3 | 0 | aux_env_ang_gap >= 0.001; env_ang_amp_deg >= 0.01 | after the phasor pass: 98 block means per signal, running median, 5 x 28 DFT bins |
| `env_ang_prom` | 1 | peak / band median; compared when aux_env_ang_gap >= 1e-3 | 0.05 + 0.01 x abs(ref) | aux_env_ang_gap >= 0.001; env_ang_amp_deg >= 0.01 | after the phasor pass: 98 block means per signal, running median, 5 x 28 DFT bins |
| `env_mag_rms_pct` | % | RMS of the detrended abs(V1) envelope | 0.001 | - | after the phasor pass: 98 block means per signal, running median, 5 x 28 DFT bins |
| `env_ang_rms_deg` | deg | RMS of the detrended angle envelope | 0.001 | - | after the phasor pass: 98 block means per signal, running median, 5 x 28 DFT bins |
| `thd_first` | 1 | THD of the first 500-sample window (mean over channels) | 1e-06 | - | per 500-sample window, 15 bins x 3 channels |
| `thd_last` | 1 | THD of the last window | 1e-06 | - | per 500-sample window, 15 bins x 3 channels |
| `thd_max` | 1 | max THD over windows | 1e-06 | - | per 500-sample window, 15 bins x 3 channels |

## Auxiliary outputs (not features; never used by ML)
| name | unit | definition |
|---|---|---|
| `aux_t_dip` | s | centre time of the first Urms(1/2) window with any channel < 0.90 (NaN if none) |
| `aux_t_swell` | s | same for > 1.10 |
| `aux_t_jump` | s | first OBS phasor time with abs(J) > 3 deg, if a jump is detected |
| `aux_t_freq` | s | first OBS phasor time with abs(f_dft - f_ref) > 0.1 Hz (jump-masked) |
| `aux_t_osc_mag` | s | first envelope time with abs(r_mag) > half the magnitude spectral peak |
| `aux_t_osc_ang` | s | same for the angle envelope |
| `aux_env_mag_gap` | 1 | (S1 - S2)/S1 over all 5 x 28 magnitude-spectrum cells (near-tie indicator) |
| `aux_env_ang_gap` | 1 | same for the angle spectrum |

## Notes
- Envelope features (env_*) are in percent of v1_ref (magnitude) and degrees (angle), so the rule
  thresholds read directly as "1 %" and "1 deg".
- `j_sign` is the sign of the first J lobe (a step dphi produces +dphi then -dphi of nearly equal
  size, so the sign at argmax abs(J) would be arbitrary; D-025).
- Frequency features exclude +/- 2 cycles around a detected jump (|J| > 3 deg); the PLL features
  are deliberately unmasked (they document the PLL's transient response, B7.4).
- Pre-event-only control (B7.4): the same function on the first 750 samples with OBS = REF; features
  needing more data (ROCOF, envelope spectra) are NaN there.
- Measured float32-vs-float64 differences on val: reports/m2_baseline_val.json (`float32_val`).
- C core (M6, docs/C_CORE.md): same definitions and tolerances. Known float32 difference: `rho` uses the
  FIRST argmin of |V1| over OBS; when two OBS samples have the same float32 |V1| (flat |V1|, no dip) the C
  index can differ from float64 and rho moves by the V2 noise (D-060). Since D-069 the C-port comparison (SIL)
  compares rho only where the rules use it (reference urms_min_1 < 0.95); a documentation/comparison gate, not a
  method change (the float32-numpy check above is unchanged). Invalid values are flagged
  (feature_valid) exactly where this reference yields NaN (rocof_end, D-051).
