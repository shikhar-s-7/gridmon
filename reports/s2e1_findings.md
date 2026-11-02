## 7. Findings (all [SIM]; frozen arms, one run)
1. **Pre-registered verdict: "ML better" on both sealed sets (H1, H2 supported).** GW - R: test2_shift +0.0080 [+0.0028, +0.0136],
   test2_novel +0.0048 [+0.0008, +0.0089]; GP - R: +0.0108 [+0.0063, +0.0159] and +0.0058 [+0.0023, +0.0098]. The effects are
   small (<= 1.1 macro-F1 points; GW on test2_novel: McNemar p 0.02-0.06 per seed).
   *Takeaway: trained on the conditions it will meet, the same 14 KB tree model no longer loses to the rules; here it is slightly
   better. Stage 1's "rules are more robust" becomes "ML needs representative training conditions".*
2. **Stage 1 replicated on fresh data, and retraining removes the failure.** R 0.9880 (Stage 1 test_shift 0.9903), G0 0.8581 +/- 0.0019
   (0.8653), G0 - R -0.130 (not pre-registered). Same recipe, wide training ranges: GW - G0 +0.138 [+0.125, +0.152] (H3). G0 also
   degrades on test2_novel (0.940, 18.5 % false alarms) where there is no transient: the cause is the low-SNR range.
   *Takeaway: the Stage 1 failure was a training-coverage problem, not a limit of tree models.*
3. **The rules' errors come from fixed thresholds, not from noise.** 21 errors on test2_shift and 11 on test2_novel; 13 and 11 are
   phase jumps called dip or swell (half-cycle RMS crosses 0.90/1.10 transiently, D-035; recall at 20-30 deg 0.91 / 0.90); 6 are
   oscillations called swell on test2_shift (post hoc: 5 peaks near 1.08 pu lifted over 1.10 by +1.7-2.0 % gain error, 1 by the
   transient; cf. Stage 1 finding 6). *Takeaway: the rules are limited by threshold artefacts and calibration, as in Stage 1.*
4. **The unseen interharmonic is survived by every arm except G0 (H5, H6 supported).** Change with - without it: R -0.0017
   [-0.0089, +0.0055], GW -0.0031 [-0.0072, 0.0000], GP -0.0013 [-0.0040, 0.0000]; G0 inconclusive. The intended oscillation
   mimicry appeared only for beats of 20-25 Hz (n = 8): accuracy GW/HW 0.67, H0 0.50, G0 0.38 (jumps and frequency events called
   oscillation), R and GP 1.00. *Takeaway: a 0.5-2 % interharmonic is mostly filtered out by the one-cycle DFT and the 100 Hz
   envelope; the residual risk is a beat inside the oscillation band, where most learned models wobble (n too small for a claim).*
5. **The unseen Stage 1 transient** (test2_shift, 495 records): macro-F1 without / with it R 0.991 / 0.980, GW 0.998 / 0.990,
   GP 0.998 / 1.000. *Takeaway: neither approach collapses on a transient it never saw.*
6. **Confidence-gated hybrids work, also as a rescue (Stage 1 finding 7 replicated on fresh data).** HW - R +0.0097 / +0.0048
   (ML better). H0 (the mis-trained G0 gated at 0.5) rises from 0.858 to 0.993 on test2_shift (H0 - R +0.0051 [+0.0017, +0.0090])
   and matches R on test2_novel, but defers 22 % of test2_shift records to the rules.
   *Takeaway: "use the model only when it is confident" is a cheap safety net when training coverage is uncertain.*
7. **Safety-critical confusions stay at zero:** phase_jump <-> freq_event and sag -> normal are 0 for every arm on both sets.
8. **Post hoc (not pre-registered, untested):** with identical hyperparameters GP made fewer errors than GW (seed 0: 2 vs 7 on
   test2_shift, 1 vs 3 on test2_novel); GW's largest splits use env_mag_prom and dfpll_max, GP's rho, v1_max_rel and u2_max.
   A GP-vs-GW claim needs another sealed set.

## 8. Limitations
- One generator for training and test. test2_shift's nuisance ranges lie inside the wide training ranges (its data and the
  transient are new, the ranges are not); the interharmonic is the only genuinely unseen nuisance, and it proved mild.
- val_wide was saturated again (31 of 33 and 33 of 33 feasible configurations within 0.002 of 1.000), so selection took the
  smallest model; GW and GP ended with identical hyperparameters.
- Small effects; four primary tests without multiplicity correction (GW's test2_novel lower bound is +0.0008, the reading a
  correction could change). The rules were kept frozen; a re-tuned or D-035-corrected rule set was not tested.
- Strong-grid assumption, single-label records, interharmonic modelled as one balanced positive-sequence component; no
  measured data ([EXP] none).
