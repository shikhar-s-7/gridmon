# RESULTS — gridmon Stage 1, sealed test evaluation (M4) [SIM]

> Simulation-only: synthetic records from the project's own generator; not validated on measured data;
> not for protection or control. Every number below is a [SIM] result.

## 1. Setup
- Test sets (sealed until this run): test_iid (seed 3003, iid nuisances) and test_shift (seed 4004: SNR 25-35 dB,
  f_pre 49.5-50.5 Hz, more harmonics/unbalance, +/-2 % gain/offset, unseen damped 300-900 Hz transient in 572 of 1750 records); 250 records per class each; data generated at commit d713ccc.
- Frozen methods (M4-freeze commit 6240f60; evaluated at 6240f60; one run):
  rules with the TRAIN-tuned thresholds (D-036: dip_pu 0.9, swell_pu 1.1, rho_th 0.25, osc_mag_pct 1.0, osc_ang_deg 1.0, osc_prom 4.0, jump_deg 5.0, jump_max_dv1 0.1, df_hz 0.2);
  GBM-embedded = sklearn GradientBoostingClassifier, 20 stages x 4 leaves, lr 0.1, subsample 1.0, 13972 B exported, seeds [0, 1, 2], run through the float32 reference walker on the exported JSON (= the C model);
  GBM-unconstrained = HistGradientBoostingClassifier (max_leaf_nodes 15, lr 0.1, early stopping), seeds [0, 1, 2]; reference for the cost of the 64 KB budget.
- All methods use the same 37 M2 features computed from the int16 waveforms only; labels and manifests are
  used for scoring and breakdowns only. Selection used TRAIN/VAL only (M2, M3).
- Statistics: macro-F1 primary; ML = mean +/- sd over 3 seeds; confusions and figures use seed 0 (the exported
  model); paired bootstrap 2000 resamples stratified by class (seed-mean difference); exact McNemar on per-record correctness (per seed).

## 2. Headline results (macro-F1; ML mean +/- sd over seeds)

| method | test_iid | test_shift | iid -> shift (95 % CI) | false alarms iid / shift | misses iid / shift |
|---|---|---|---|---|---|
| rules | 0.9983 | 0.9903 | -0.0080 [-0.0132, -0.0034] | 0.0 % / 0.0 % | 0.0 % / 0.0 % |
| GBM-embedded | 1.0000 | 0.8653 +/- 0.0014 | -0.1347 [-0.1484, -0.1215] | 0.0 % / 23.3 % | 0.0 % / 7.3 % |
| GBM-unconstrained | 1.0000 | 0.8484 +/- 0.0066 | -0.1516 [-0.1654, -0.1380] | 0.0 % / 58.1 % | 0.0 % / 0.8 % |

## 3. Method differences (paired bootstrap 95 % CI of the macro-F1 difference; exact McNemar)

| comparison | split | diff | 95 % CI | McNemar A-only / B-only correct, p (seeds 0/1/2) | reading |
|---|---|---|---|---|---|
| GBM-emb - rules | test_iid | +0.0017 | [+0.0000, +0.0040] | 3/0, p=0.25; 3/0, p=0.25; 3/0, p=0.25 | no significant difference |
| GBM-emb - rules | test_shift | -0.1250 | [-0.1407, -0.1101] | 11/209, p=1.4e-48; 11/209, p=1.4e-48; 11/214, p=5.7e-50 | B better |
| GBM-unc - rules | test_iid | +0.0017 | [+0.0000, +0.0040] | 3/0, p=0.25; 3/0, p=0.25; 3/0, p=0.25 | no significant difference |
| GBM-unc - rules | test_shift | -0.1419 | [-0.1567, -0.1280] | 16/225, p=2.3e-48; 13/254, p=3.7e-59; 12/235, p=7.6e-55 | B better |
| GBM-unc - GBM-emb (seed 0) | test_iid | +0.0000 | [+0.0000, +0.0000] | 0/0, p=1; 0/0, p=1; 0/0, p=1 | no significant difference |
| GBM-unc - GBM-emb (seed 0) | test_shift | -0.0178 | [-0.0312, -0.0043] | 94/105, p=0.48; 49/92, p=0.00037; 81/106, p=0.079 | B better |

## 4. Confusion matrices (rows true, columns predicted)

```
test_iid
                     rules             |    GBM-embedded (seed 0)     |  GBM-unconstrained (seed 0) 
true\pred  nor sgB sgU swl pjp frq osc |  nor sgB sgU swl pjp frq osc |  nor sgB sgU swl pjp frq osc
nor        250   0   0   0   0   0   0 |  250   0   0   0   0   0   0 |  250   0   0   0   0   0   0
sgB          0 249   1   0   0   0   0 |    0 250   0   0   0   0   0 |    0 250   0   0   0   0   0
sgU          0   0 250   0   0   0   0 |    0   0 250   0   0   0   0 |    0   0 250   0   0   0   0
swl          0   0   0 250   0   0   0 |    0   0   0 250   0   0   0 |    0   0   0 250   0   0   0
pjp          0   0   0   2 248   0   0 |    0   0   0   0 250   0   0 |    0   0   0   0 250   0   0
frq          0   0   0   0   0 250   0 |    0   0   0   0   0 250   0 |    0   0   0   0   0 250   0
osc          0   0   0   0   0   0 250 |    0   0   0   0   0   0 250 |    0   0   0   0   0   0 250
```
```
test_shift
                     rules             |    GBM-embedded (seed 0)     |  GBM-unconstrained (seed 0) 
true\pred  nor sgB sgU swl pjp frq osc |  nor sgB sgU swl pjp frq osc |  nor sgB sgU swl pjp frq osc
nor        250   0   0   0   0   0   0 |  194   0   0   0   3   0  53 |   92   2  54   4  60   7  31
sgB          0 250   0   0   0   0   0 |    0 250   0   0   0   0   0 |    0 250   0   0   0   0   0
sgU          0   0 250   0   0   0   0 |    0   0 250   0   0   0   0 |    0   0 250   0   0   0   0
swl          0   0   0 250   0   0   0 |    0   0   0 250   0   0   0 |    0   0   0 250   0   0   0
pjp          0   1   6   6 237   0   0 |  107   0   9   1  95   0  38 |    0   0  54  12 182   0   2
frq          0   0   0   2   0 248   0 |    0   0   0   0   0 248   2 |    0   0   0   0   0 250   0
osc          0   1   0   1   0   0 248 |    2   0   0   0   0   0 248 |    0   0   0   0   0   0 250
```

## 5. Per-class F1 and critical confusions (ML: seed mean)

| class | rules iid | rules shift | GBM-embedded iid | GBM-embedded shift | GBM-unconstrained iid | GBM-unconstrained shift |
|---|---|---|---|---|---|---|
| normal | 1.0000 | 1.0000 | 1.0000 | 0.6961 | 1.0000 | 0.5680 |
| sag_bal | 0.9980 | 0.9960 | 1.0000 | 1.0000 | 1.0000 | 0.9907 |
| sag_unbal | 0.9980 | 0.9881 | 1.0000 | 0.9836 | 1.0000 | 0.8675 |
| swell | 0.9960 | 0.9823 | 1.0000 | 0.9980 | 1.0000 | 0.9517 |
| phase_jump | 0.9960 | 0.9733 | 1.0000 | 0.5460 | 1.0000 | 0.6809 |
| freq_event | 1.0000 | 0.9960 | 1.0000 | 0.9973 | 1.0000 | 0.9664 |
| oscillation | 1.0000 | 0.9960 | 1.0000 | 0.8360 | 1.0000 | 0.9133 |

| critical confusion (count) | rules iid | rules shift | GBM-embedded iid | GBM-embedded shift | GBM-unconstrained iid | GBM-unconstrained shift |
|---|---|---|---|---|---|---|
| phase_jump->freq_event | 0 | 0 | 0 | 0 | 0 | 2.67 |
| freq_event->phase_jump | 0 | 0 | 0 | 0 | 0 | 1.33 |
| sag_*->normal | 0 | 0 | 0 | 0 | 0 | 0 |
| sag_bal->sag_unbal | 1 | 0 | 0 | 0 | 0 | 0.667 |
| sag_unbal->sag_bal | 0 | 0 | 0 | 0 | 0 | 0 |

## 6. Breakdowns

SNR (macro-F1 per bin; bins below 35 dB exist only in test_shift, above only in test_iid):

| SNR bin (dB) | n | rules | GBM-embedded | GBM-unconstrained |
|---|---|---|---|---|
| 25-30 | 894 | 0.9943 | 0.7819 | 0.7980 |
| 30-35 | 856 | 0.9862 | 0.9222 | 0.8903 |
| 35-40 | 359 | 1.0000 | 1.0000 | 1.0000 |
| 40-50 | 709 | 1.0000 | 1.0000 | 1.0000 |
| 50-60 | 682 | 0.9956 | 1.0000 | 1.0000 |

Severity (recall per bin, ML seed mean; iid / shift). Listed: bins where some method is below 0.99 on
either set; every unlisted bin (bins: sag depth, swell height, abs(dphi), abs(ROCOF), oscillation depth)
is >= 0.99 for every method on both sets (all bins in results.json):

| class | variable | bin | n iid / shift | rules | GBM-embedded | GBM-unconstrained |
|---|---|---|---|---|---|---|
| sag_bal | min LL fundamental in the dip (pu) | 0.7-0.87 | 49 / 38 | 0.98 / 1.00 | 1.00 / 1.00 | 1.00 / 0.98 |
| phase_jump | abs(dphi) (deg) | 6-12 | 52 / 65 | 1.00 / 1.00 | 1.00 / 0.25 | 1.00 / 0.70 |
| phase_jump | abs(dphi) (deg) | 12-20 | 85 / 78 | 1.00 / 0.97 | 1.00 / 0.37 | 1.00 / 0.64 |
| phase_jump | abs(dphi) (deg) | 20-30 | 113 / 107 | 0.98 / 0.90 | 1.00 / 0.47 | 1.00 / 0.63 |
| freq_event | abs(ROCOF) (Hz/s) | 0.5-1 | 36 / 30 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.97 |
| freq_event | abs(ROCOF) (Hz/s) | 1-2 | 80 / 80 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.98 |
| freq_event | abs(ROCOF) (Hz/s) | 2-4 | 134 / 140 | 1.00 / 0.99 | 1.00 / 0.99 | 1.00 / 0.97 |
| oscillation | modulation depth (%) | 2-4 | 169 / 189 | 1.00 / 0.99 | 1.00 / 0.99 | 1.00 / 1.00 |

Unseen shift transient (test_shift, macro-F1):

| subset | n | rules | GBM-embedded | GBM-unconstrained |
|---|---|---|---|---|
| without transient | 1178 | 0.9925 | 0.8806 | 0.8698 |
| with transient | 572 | 0.9863 | 0.8275 | 0.8004 |

## 7. Naive-PLL diagnostic (flag 'frequency event' if |f_pll - f_ref| > 1 Hz at any instant)

| class | test_iid flagged | test_shift flagged |
|---|---|---|
| normal | 0/250 (0.0 %) | 31/250 (12.4 %) |
| sag_bal | 250/250 (100.0 %) | 250/250 (100.0 %) |
| sag_unbal | 247/250 (98.8 %) | 249/250 (99.6 %) |
| swell | 90/250 (36.0 %) | 138/250 (55.2 %) |
| phase_jump | 250/250 (100.0 %) | 250/250 (100.0 %) |
| freq_event | 106/250 (42.4 %) | 214/250 (85.6 %) |
| oscillation | 81/250 (32.4 %) | 152/250 (60.8 %) |
| phase_jump by abs(dphi) | 6-12 deg 52/52, 12-20 deg 85/85, 20-30 deg 113/113 | 6-12 deg 65/65, 12-20 deg 78/78, 20-30 deg 107/107 |

## 8. Embedded cost (per 1 s record)

| block | operations | RAM | const/flash |
|---|---|---|---|
| DSP + 37 features (shared by all methods; M2 estimate) | 1.52 M ops | 21.17 KB (+30 KB int16 record) | 6.05 KB |
| rules | <= 12 threshold comparisons | negligible | thresholds |
| GBM-embedded | 239 comparisons mean (worst 357), 140 adds, 7 expf | 176 B | 13.6 KB |
| GBM-unconstrained (reference, not exported) | - | - | ~171 KB equivalent |

Python timings (indicative, this VM): features 3.34 ms/record; rules 0.0020; GBM-embedded float32 walker 0.011; HGB 0.014 ms/record. C2000 cycles need Stage 2 PIL.

## 9. Findings (all [SIM]; frozen methods, one run)
1. **iid: no significant difference.** GBM-embedded 1.0000 vs rules 0.9983 on test_iid: diff +0.0017, 95 % CI
   [+0.0000, +0.0040], McNemar 3 vs 0 discordant records (p = 0.25). The rules' 3 errors: 2 phase jumps -> swell
   (Urms(1/2) transient across the jump, D-035) and 1 sag_bal -> sag_unbal.
   *Takeaway: on data like the training data, the 14 KB tree ensemble and the standards-based rules are equally good.*
2. **Shift: the rules are significantly better.** test_shift: rules 0.9903, GBM-embedded 0.8653 +/- 0.0014,
   GBM-unconstrained 0.8484 +/- 0.0066; GBM-embedded - rules = -0.1250, CI [-0.1407, -0.1101], McNemar 11 vs 209
   (p <= 1.4e-48 for every seed). iid -> shift: rules -0.008, GBM-embedded -0.135.
   *Takeaway: when noise and nuisances move beyond the training range, the learned classifier degrades sharply;
   the rules barely move.*
3. **Where ML fails: low SNR, phase jumps and false alarms.** GBM-embedded seed 0 on test_shift: phase_jump recall
   0.38 (107/250 called normal, 38 oscillation), 56/250 normal records flagged (53 as oscillation); seed mean:
   false alarms 23.3 %, misses 7.3 %.
   The drop sits at 25-30 dB SNR (macro-F1 0.78 vs rules 0.99) and is present without the unseen transient
   (0.88 vs 0.83 with it). Post-hoc diagnosis (not a method change): the selected trees separate jump from
   normal partly with noise-sensitive cues (rocof_max split near 0.25 Hz/s, env_mag_prom near 1.4); at 25-35 dB
   the median rocof_max of normal records rises from 0.10 (train) to about 0.5 Hz/s, while j_max_deg of the
   missed jumps stays large (median 15 deg).
   *Takeaway: the model learned shortcuts that only hold at the training noise level; the physical jump
   threshold (|J| > 5 deg) does not depend on noise.*
4. **The 64 KB budget costs nothing here.** GBM-unconstrained (~171 KB equivalent) equals GBM-embedded on test_iid
   (both 1.000) and is worse on test_shift (-0.018 vs seed 0, CI [-0.031, -0.004]; 58 % false alarms).
   *Takeaway: a bigger model did not buy robustness.*
5. **Safety-critical confusions stay at zero for the rules and GBM-embedded.** phase_jump <-> freq_event 0 and
   sag -> normal 0 on both test sets (GBM-unconstrained: ~4 jump <-> frequency confusions on test_shift). The
   naive PLL rule flags 100 % of phase jumps, >= 98.8 % of dips and 12.4 % of normal shift records as frequency
   events. *Takeaway: the DFT jump metric with jump-masked f_dft is what prevents the Blue Cut Fire failure mode.*
6. **Rule errors come from the measurement chain.** The rules' 17 test_shift errors (3 on test_iid): 13 phase
   jumps called dip/swell because Urms(1/2) crosses 0.90/1.10 transiently (D-035; 2 on test_iid), 2 oscillations
   (one pushed below 0.90 by a negative channel gain error, #356) and 2 frequency events called swell (an unseen
   transient lifts Urms(1/2) above 1.10, #50).
   *Takeaway: fixed IEC thresholds are robust to noise but sensitive to calibration error and transients.*
7. **Post-hoc observation (hypothesis only):** all 215 GBM-embedded errors on test_shift had max probability
   < 0.5, against 5 of 1,750 correct test_iid records. A confidence gate with rule fallback is a candidate Stage 2
   design; it was chosen after seeing test data and must be evaluated on fresh sealed data.
8. **Embedded cost does not decide.** GBM-embedded adds ~240 comparisons, 140 adds, 7 expf, 176 B RAM and 13.6 KB
   flash to a 1.52 M-operation feature pipeline shared by both methods; robustness decides.

## 10. Limitations
- Synthetic data from this project's own generator; training, val and both test sets share its physics, and
  the shift is one designed nuisance configuration (lower SNR, wider f_pre, more harmonics/unbalance, +/-2 %
  calibration error, one unseen transient type). The rules were written by people who know the generator.
- Strong-grid assumption (PCS current does not affect the voltage); no converter/PCS dynamics; no weak grids.
- Single-label records with label margins; no multi-event records, no threshold-straddling events.
- No measured data ([EXP] none); timings are Python on a cloud VM; C2000 cycles need Stage 2 PIL.
- VAL was saturated (33 GBM configurations tied at 1.000), so model selection could not prefer robust models;
  the ML was trained without noise augmentation. The sealed sets were regenerated after R1 (D-034), which was
  decided before any test evaluation.

## 11. What would change the conclusion
- GBM trained on the shift nuisance ranges (SNR 25-60 dB, +/-2 % gain): if it then matches the rules on
  test_shift-like data, the finding becomes "ML needs representative training conditions", not "rules are
  more robust".
- Restricting the GBM to physically motivated features (or monotonic constraints), removing noise-sensitive cues.
- Measured PCS records with real noise, converter dynamics, calibration drift and multi-event cases: the rules'
  fixed thresholds could fail there too (finding 6 already shows calibration sensitivity).
- A confidence-gated hybrid (finding 7) evaluated on new sealed data.

Figures: reports/figures/m4_confusions.png, m4_f1_by_snr.png, m4_shift_degradation.png, m4_disagreements.png; all numbers: reports/results.json.
