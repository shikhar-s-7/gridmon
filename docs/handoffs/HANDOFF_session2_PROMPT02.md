# HANDOFF — gridmon session 2 (PROMPT_02: review decisions, R1, M3, M4), 2026-11-01

Simulation-only; every result is [SIM]. Previous handoff: docs/handoffs/HANDOFF_session1_PROMPT01.md.

## 1. Status and commits
- Done: review decisions D-034..D-037, **R1** (oscillation label margin + regeneration), **M3** (gradient
  boosting, 64 KB budget, C-ready export), **M4** (freeze, one-shot sealed evaluation, RESULTS.md = M5).
  Not started: C core (Prompt 03).
- Commits: R1 code `d713ccc`, R1 data/M2 re-run `4006c08`, M3 code `fd105b4`, M3 results `e11cc1d`,
  **M4-freeze `6240f60`**, M4 results `da44023`, wrap-up (this handoff) = next commit. Branch claude/zen-bell-5wqk7y.
- Environment: Linux cloud VM, Python 3.12.3, every pin of requirements.txt installs unchanged. M1/M2
  reproduce exactly on Linux (val 0.933, same confusion; golden int16 inputs bit-identical). FIR file kept
  platform-stable (D-038). Data meta.json now records clean hashes (code committed before generation).
- Runtimes (4 vCPU): dataset 145 s, baseline 45 s, golden 1 s, run_gbm 290 s (grid 261 s), evaluate_test 26 s;
  `run_all.py --fast` 88 s (all 5 stages); pytest 46 s: **81 passed, 0 failed, 0 skipped**.
- Git author in this VM is its global config "Claude <noreply@anthropic.com>" (repo has no local author; kept
  as configured, PROMPT_02 A1.1). No attribution lines in messages.

## 2. R1 — oscillation margin (D-034) and regeneration
- Rule: analytic LL fundamental of oscillation records within [0.92, 1.08] pu over the whole record; a failing
  draw redraws only the oscillation event. Margin metric: manifest `mm_min_ll` / `mm_max_ll`.
- Rejection rate (oscillation): train 0.73, val 0.74, test_iid 0.75, test_shift 0.78 (max 51 redraws). Realised
  depth median 3.3 % (2-7.2 %), sigma median -0.24 /s, depth x envelope at end <= 7.9 % (was <= 34 %).
- Edge case D-039: in the --fast test_shift subset one pre-event (U_pre 1.039, unbalance 2.45 %) makes the band
  infeasible; the least-violating draw is kept and flagged. Never happens in the full splits.
- Unchanged-record check (same platform, before/after): per-record SHA-1 of X identical for EVERY
  non-oscillation record of train/val/preview_iid/preview_shift (4800/4800, 1500/1500, 172/172, 172/172; fast
  likewise) and for every oscillation record accepted at its first draw (271/271, 77/77, 8/8, 13/13). Sealed
  splits compared by manifest rows (1500/1500 each), not waveforms (rule A1.4).
- M2 after R1 (rules, val): macro-F1 **0.997** (train-tuned = defaults, D-036), false alarms 0 %, misses 0.07 %;
  pre-event control 0.140. DSP/feature docs unchanged (no DSP change); golden vectors regenerated.
```
true \ pred     normal  sag_bal sag_unbal  swell  phase_jump freq_event oscillation
normal             250        0        0       0          0          0          0
sag_bal              0      250        0       0          0          0          0
sag_unbal            0        0      250       0          0          0          0
swell                0        0        0     250          0          0          0
phase_jump           0        0        3       2        245          0          0     (D-035, accepted)
freq_event           0        0        0       0          0        250          0
oscillation          1        0        0       0          0          0        249
```

## 3. M3 — gradient boosting (TRAIN/VAL only)
- Same 37 M2 features; non-finite inputs -> 0.0 (D-040; only rocof_end, 32 train / 12 val records).
- GBM-embedded grid: 60 configurations x 3 seeds (36 fits via stages, D-041); exported size 13,972-303,072 B;
  36/60 fit 64 KB; val macro-F1 0.9968-1.0000 (val saturated: 33 feasible configs within 0.002 of 1.000).
- **Selected: 20 stages x 4 leaves, lr 0.1, subsample 1.0 = 13,972 B (13.6 KB), 140 trees, 976 nodes;
  val macro-F1 1.0000 for all 3 seeds** (rules 0.9966; GBM-unconstrained HGB 15 leaves lr 0.1: 0.9998, ~171 KB).
- Export: models/gbm_embedded.json (+ seed1/2), c/generated/gm_model_gbm.{c,h} (const arrays), docs/MODEL_FORMAT.md
  (14 B/node + 2 B/tree + 4 B/class; thresholds = largest float32 <= float64; lr folded into leaves).
- **Equivalence on val (all 3 seeds):** float32 walker vs sklearn predict_proba max |diff| 1.96e-7, labels 100 %,
  0 / 731,500 branch mismatches (~135 thresholds per model needed the downward float32 step). Informal extra:
  a throwaway C walker over the generated tables matched the Python walker bit-exactly on all 1,750 val records.
- Pre-event-only control (selected config, 3 seeds): macro-F1 **0.124** (chance 0.143) -> no leakage.
- Permutation importance (val): urms_min_3, rho, rocof_max, env_mag_prom, j_max_deg, env_ang_rms_deg dominate.
- Embedded cost: ~240 comparisons/record (worst 357), 140 float adds, 7 expf, 176 B RAM, 13.6 KB flash.

## 4. M4 — sealed evaluation (frozen at 6240f60, evaluated once at the same commit)
| method | test_iid | test_shift | iid -> shift | false alarms iid/shift | misses iid/shift |
|---|---|---|---|---|---|
| rules | 0.9983 | 0.9903 | -0.0080 [-0.013, -0.003] | 0 % / 0 % | 0 % / 0 % |
| GBM-embedded | 1.0000 | 0.8653 +/- 0.0014 | -0.135 [-0.148, -0.122] | 0 % / 23.3 % | 0 % / 7.3 % |
| GBM-unconstrained | 1.0000 | 0.8484 +/- 0.0066 | -0.152 [-0.165, -0.138] | 0 % / 58.1 % | 0 % / 0.8 % |

- GBM-embedded - rules: test_iid +0.0017, CI [+0.0000, +0.0040], McNemar 3/0 p = 0.25 -> **no significant
  difference**; test_shift **-0.1250, CI [-0.1407, -0.1101]**, McNemar 11/209 p = 1.4e-48 -> **rules better**.
- GBM-unconstrained - GBM-embedded (seed 0): test_shift -0.018 [-0.031, -0.004] (McNemar mixed per seed:
  p = 0.48 / 0.0004 / 0.08) -> the size budget costs nothing.
- Critical confusions: phase_jump <-> freq_event and sag -> normal = 0 for rules and GBM-embedded on both sets.
- Naive PLL (test_shift): 100 % of phase jumps, >= 98.8 % of dips and 12.4 % of normal records flagged.
- Walker vs sklearn on the test sets: labels 100 % agree (all seeds). No bug found -> no pre/post-fix pair (D-048).

## 5. Top findings in plain language
1. On test data like the training data, the standards-based rules and the 14 KB tree model are equally good.
2. When noise rises and calibration/transient nuisances appear (test_shift), the rules stay at 99 % while the
   tree model drops to 87 %: it misses most phase jumps and flags normal records, because it learned
   noise-dependent shortcuts (rocof_max, envelope prominence) instead of the physical jump size.
3. A bigger unconstrained model does not help (worse under shift); and neither the rules nor GBM-embedded ever
   confuse a phase jump with a frequency event, unlike a naive PLL rule, which flags every phase jump.
   (Post hoc: all 215 GBM errors under shift had max probability < 0.5 — a possible gate, hypothesis only.)

## 6. Deviations and interpretations (docs/DECISIONS.md)
D-034..D-037 (review, as specified); D-038 platform-stable FIR file; D-039 infeasible oscillation band
fallback; D-040 NaN -> 0 model input; D-041 grid via stages; D-042 format adds 2 B/tree root table;
D-043 selection details (max size over seeds, tie order); D-044 HGB reference config; D-045 committed model
artefacts (JSON + sklearn pickles, ~1 MB); D-046 control/importance; D-047 freeze mechanics + pre-registered
evaluation plan (dry run on VAL before the freeze); D-048 post-run reporting-only changes, post-hoc diagnostics.

## 7. Open questions for the project lead
1. Shift result: add a GBM arm trained on shift-range nuisances (noise/calibration augmentation) or with
   physics-only features? Any new arm needs NEW sealed test splits (test_shift is no longer blind).
2. Confidence-gated hybrid (GBM with rule fallback below p = 0.5) as a Stage 2 candidate (post-hoc idea only)?
3. Rules are calibration-sensitive (a -2 % channel gain pushed an oscillation below 0.90; transients lift
   Urms(1/2) above 1.10): require per-channel calibration in the C2000 integration guide (M7)?
4. rocof_end NaN (D-040): keep NaN -> 0 at the model input in the C core, or define a feature fallback in
   FEATURES.md/DSP_SPEC.md before M6 (that would change a frozen method)?
5. VAL is saturated (33 tied configs): for any future selection, use a harder (shift-like) validation split?
6. Git author: cloud commits are authored "Claude <noreply@anthropic.com>" (VM default) — keep or re-author?

## 8. Figures (reports/figures/)
- m3_size_vs_f1.png, m3_feature_importance.png, m3_val_confusion.png
- m4_confusions.png, m4_f1_by_snr.png, m4_shift_degradation.png, m4_disagreements.png
- regenerated after R1: m1_examples_per_class.png, m1_param_distributions.png, m1_shift_preview.png,
  m2_signal_chain_examples.png, m2_baseline_val_confusion.png, m2_naive_pll_false_alarms.png

**Bring back to the project-lead chat:** docs/HANDOFF.md, reports/RESULTS.md, reports/figures/m3_size_vs_f1.png,
reports/figures/m4_confusions.png, reports/figures/m4_disagreements.png
