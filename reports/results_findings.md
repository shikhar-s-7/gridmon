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
