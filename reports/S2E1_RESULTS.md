# S2E1 RESULTS — does ML catch up with representative training data? [SIM]

> Simulation-only: synthetic records from the project's own generator; not validated on measured data; not for
> protection or control. Every number is a [SIM] result. Pre-registration: docs/S2E1_PREREG.md, configs/s2e1.yaml.

## 1. Setup
- Commits: prereg `3fd3286`, S2E1-freeze `b11b67a`, evaluated at `b11b67aa3868` (one run); Stage 1 freeze `b3afdec`.
- Sealed sets (1750 + 1750 records, 250/class): test2_shift (seed 5005, Stage 1 `shift` config incl. the unseen transient in 495 records) and test2_novel (seed 6006, `wide` + an unseen interharmonic in 872 records).
- GW: 37 features (all37), selected on val_wide: 20 stages x 4 leaves, lr 0.2, subsample 1.0; 13972 B; val_wide macro-F1 1.0000; pre-event-only control 0.133.
- GP: 23 features (physics), selected on val_wide: 20 stages x 4 leaves, lr 0.2, subsample 1.0; 13972 B; val_wide macro-F1 1.0000; pre-event-only control 0.142.
- R, G0: Stage 1 frozen rules / GBM-embedded (seeds 0-2), unchanged. H0/HW: ML label if max probability >= 0.5, else the rules' label (per seed).
- Statistics: macro-F1 primary; ML mean +/- sd over 3 seeds; paired bootstrap (2,000 class-stratified resamples per set, shared by all arms) of the seed-mean difference; exact McNemar per seed; confusions = seed 0.

## 2. Headline (macro-F1; false alarms = normal called an event; misses = event called normal)

| arm | test2_shift | test2_novel | false alarms shift / novel | misses shift / novel | ML share (hybrids) |
|---|---|---|---|---|---|
| R (Stage 1 rules) | 0.9880 | 0.9937 | 0.4 % / 0.0 % | 0.0 % / 0.0 % |  |
| G0 (Stage 1 GBM) | 0.8581 +/- 0.0019 | 0.9396 +/- 0.0006 | 27.9 % / 18.5 % | 6.3 % / 1.9 % |  |
| GW (GBM, wide training) | 0.9960 +/- 0.0000 | 0.9985 +/- 0.0003 | 1.2 % / 0.0 % | 0.2 % / 0.0 % |  |
| GP (GBM, physics subset) | 0.9989 +/- 0.0000 | 0.9994 +/- 0.0000 | 0.0 % / 0.0 % | 0.0 % / 0.1 % |  |
| HW (hybrid GW/R) | 0.9977 +/- 0.0000 | 0.9985 +/- 0.0003 | 1.2 % / 0.0 % | 0.0 % / 0.0 % | 98.5 % / 100.0 % |
| H0 (hybrid G0/R) | 0.9931 +/- 0.0000 | 0.9937 +/- 0.0000 | 0.8 % / 0.8 % | 0.0 % / 0.0 % | 77.8 % / 89.0 % |

## 3. Comparisons (paired bootstrap 95 % CI of the seed-mean macro-F1 difference; exact McNemar)

| comparison | role | set | diff | 95 % CI | CI vs 0 | McNemar A-only / B-only correct, p (seeds) |
|---|---|---|---|---|---|---|
| GW-R | primary | test2_shift | +0.0080 | [+0.0028, +0.0136] | above 0 | 20/6, p=0.0094; 20/6, p=0.0094; 20/6, p=0.0094 |
| GW-R | primary | test2_novel | +0.0048 | [+0.0008, +0.0089] | above 0 | 11/3, p=0.057; 11/2, p=0.022; 11/3, p=0.057 |
| GP-R | primary | test2_shift | +0.0108 | [+0.0063, +0.0159] | above 0 | 20/1, p=2.1e-05; 20/1, p=2.1e-05; 20/1, p=2.1e-05 |
| GP-R | primary | test2_novel | +0.0058 | [+0.0023, +0.0098] | above 0 | 11/1, p=0.0063; 11/1, p=0.0063; 11/1, p=0.0063 |
| HW-R | secondary | test2_shift | +0.0097 | [+0.0051, +0.0148] | above 0 | 19/2, p=0.00022; 19/2, p=0.00022; 19/2, p=0.00022 |
| HW-R | secondary | test2_novel | +0.0048 | [+0.0008, +0.0089] | above 0 | 11/3, p=0.057; 11/2, p=0.022; 11/3, p=0.057 |
| H0-R | secondary | test2_shift | +0.0051 | [+0.0017, +0.0090] | above 0 | 10/1, p=0.012; 10/1, p=0.012; 10/1, p=0.012 |
| H0-R | secondary | test2_novel | +0.0000 | [-0.0045, +0.0046] | includes 0 | 8/8, p=1; 8/8, p=1; 8/8, p=1 |
| GW-G0 | secondary | test2_shift | +0.1379 | [+0.1246, +0.1521] | above 0 | 220/1, p=1.3e-64; 221/1, p=6.6e-65; 227/1, p=1.1e-66 |
| GW-G0 | secondary | test2_novel | +0.0589 | [+0.0487, +0.0701] | above 0 | 102/0, p=3.9e-31; 103/0, p=2e-31; 104/0, p=9.9e-32 |
| G0-R | additional (not pre-registered) | test2_shift | -0.1299 | [-0.1445, -0.1158] | below 0 | 12/217, p=7.9e-50; 12/218, p=4.2e-50; 12/224, p=9e-52 |
| G0-R | additional (not pre-registered) | test2_novel | -0.0541 | [-0.0656, -0.0433] | below 0 | 8/102, p=6.8e-22; 8/102, p=6.8e-22; 8/104, p=2e-22 |

## 4. Pre-registered decision rule and hypotheses

| arm | role | CI (arm - R) test2_shift | CI test2_novel | verdict |
|---|---|---|---|---|
| GW | primary | [+0.0028, +0.0136] | [+0.0008, +0.0089] | **ML better** |
| GP | primary | [+0.0063, +0.0159] | [+0.0023, +0.0098] | **ML better** |
| HW | secondary | [+0.0051, +0.0148] | [+0.0008, +0.0089] | **ML better** |
| H0 | secondary | [+0.0017, +0.0090] | [-0.0045, +0.0046] | **ML matches** |

- H1 (GW matches or beats the rules (decision rule) on both sealed sets): **supported**
- H2 (GP matches or beats the rules on both sealed sets): **supported**
- H3 (GW is better than G0 on test2_shift (CI of GW - G0 above 0)): **supported**
- H4 (HW matches or beats the rules on both sealed sets): **supported**
- H5 (R survives the unseen interharmonic (survival rule)): **supported**
- H6 (GW survives the unseen interharmonic (survival rule)): **supported**

Unseen-nuisance survival on test2_novel (delta = macro-F1 with - without interharmonic; survives if CI lower >= -0.03):

| arm | without | with | delta | 95 % CI | verdict |
|---|---|---|---|---|---|
| R | 0.9944 | 0.9928 | -0.0017 | [-0.0089, +0.0055] | survives |
| G0 | 0.9427 | 0.9348 | -0.0079 | [-0.0300, +0.0119] | inconclusive |
| GW | 1.0000 | 0.9969 | -0.0031 | [-0.0072, +0.0000] | survives |
| GP | 1.0000 | 0.9987 | -0.0013 | [-0.0040, +0.0000] | survives |
| H0 | 0.9967 | 0.9910 | -0.0057 | [-0.0134, +0.0018] | survives |
| HW | 1.0000 | 0.9969 | -0.0031 | [-0.0074, +0.0000] | survives |

## 5. Per-class F1 and critical confusions (seed mean)

| class | R shift / novel | G0 shift / novel | GW shift / novel | GP shift / novel | HW shift / novel |
|---|---|---|---|---|---|
| normal | 0.998 / 1.000 | 0.688 / 0.844 | 0.988 / 1.000 | 1.000 / 0.998 | 0.994 / 1.000 |
| sag_bal | 0.998 / 0.998 | 1.000 / 1.000 | 0.998 / 1.000 | 0.998 / 1.000 | 0.998 / 1.000 |
| sag_unbal | 0.992 / 0.994 | 0.989 / 0.990 | 0.998 / 1.000 | 0.998 / 1.000 | 0.998 / 1.000 |
| swell | 0.967 / 0.986 | 0.995 / 0.998 | 1.000 / 1.000 | 1.000 / 1.000 | 0.998 / 1.000 |
| phase_jump | 0.973 / 0.978 | 0.534 / 0.880 | 0.988 / 0.996 | 0.998 / 0.998 | 0.996 / 0.996 |
| freq_event | 1.000 / 1.000 | 1.000 / 0.992 | 1.000 / 0.999 | 1.000 / 1.000 | 1.000 / 0.999 |
| oscillation | 0.988 / 1.000 | 0.800 / 0.874 | 1.000 / 0.995 | 0.998 / 1.000 | 1.000 / 0.995 |
| phase_jump<->freq_event (count) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| sag_*->normal (count) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| sag_bal<->sag_unbal (count) | 1 / 0 | 0 / 0 | 1 / 0 | 1 / 0 | 1 / 0 |

## 6. Breakdowns (macro-F1 per bin unless stated; seed mean)

| set | subset | n | R | G0 | GW | GP | HW | H0 |
|---|---|---|---|---|---|---|---|---|
| test2_shift | SNR 25-30 dB | 854 | 0.9881 | 0.7802 | 0.9954 | 0.9988 | 0.9965 | 0.9917 |
| test2_shift | SNR 30-35 dB | 896 | 0.9879 | 0.9189 | 0.9966 | 0.9988 | 0.9989 | 0.9944 |
| test2_novel | SNR 25-30 dB | 253 | 0.9884 | 0.7680 | 1.0000 | 0.9964 | 1.0000 | 0.9884 |
| test2_novel | SNR 30-35 dB | 252 | 0.9914 | 0.9189 | 0.9961 | 1.0000 | 0.9961 | 0.9961 |
| test2_novel | SNR 35-40 dB | 260 | 1.0000 | 0.9706 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test2_novel | SNR 40-50 dB | 484 | 0.9923 | 0.9755 | 0.9966 | 1.0000 | 0.9966 | 0.9919 |
| test2_novel | SNR 50-60 dB | 501 | 0.9958 | 0.9756 | 1.0000 | 1.0000 | 1.0000 | 0.9941 |
| test2_shift | without transient | 1255 | 0.9912 | 0.8668 | 0.9984 | 0.9984 | 0.9992 | 0.9952 |
| test2_shift | with transient | 495 | 0.9804 | 0.8343 | 0.9903 | 1.0000 | 0.9942 | 0.9879 |
| test2_novel | without interharmonic | 878 | 0.9944 | 0.9427 | 1.0000 | 1.0000 | 1.0000 | 0.9967 |
| test2_novel | with interharmonic | 872 | 0.9928 | 0.9348 | 0.9969 | 0.9987 | 0.9969 | 0.9910 |
| test2_novel | accuracy, beat < 25 Hz | 8 | 1.0000 | 0.3750 | 0.6667 | 1.0000 | 0.6667 | 0.5000 |
| test2_novel | accuracy, beat >= 25 Hz | 864 | 0.9931 | 0.9448 | 1.0000 | 0.9988 | 1.0000 | 0.9954 |
| test2_novel | accuracy, beat 20-25 Hz | 8 | 1.0000 | 0.3750 | 0.6667 | 1.0000 | 0.6667 | 0.5000 |
| test2_novel | accuracy, beat 25-40 Hz | 24 | 1.0000 | 0.7500 | 1.0000 | 1.0000 | 1.0000 | 0.8750 |
| test2_novel | accuracy, beat 40-100 Hz | 40 | 1.0000 | 0.9333 | 1.0000 | 1.0000 | 1.0000 | 0.9750 |
| test2_novel | accuracy, beat 100-900 Hz | 800 | 0.9925 | 0.9513 | 1.0000 | 0.9987 | 1.0000 | 1.0000 |

Errors on interharmonic records (seed 0, true->predicted: count): R: phase_jump->swell 5, phase_jump->sag_unbal 1; G0: normal->oscillation 21, phase_jump->normal 16, phase_jump->oscillation 10, freq_event->oscillation 4; GW: phase_jump->oscillation 2, freq_event->oscillation 1; GP: phase_jump->normal 1

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

Figures: reports/figures/s2e1_confusions.png, s2e1_f1_by_snr.png, s2e1_novel_breakdown.png; all numbers: reports/s2e1_results.json.
