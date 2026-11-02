# HANDOFF — gridmon session 5 (PROMPT_05: Stage 2 experiment S2E1), 2026-11-02

Simulation-only; every result is [SIM]. Stage 1 complete and frozen; **S2E1 (STAGE2_PLAN item 3) done**. Previous handoffs:
docs/handoffs/ (session 4 archived as HANDOFF_session4_PROMPT04.md). Branch `claude/new-session-2q6d5l`, author shikhar-s-7.

## 1. Commits
| commit | what |
|---|---|
| `8996518` | Housekeeping: rho SIL gate urms_min_1 < 0.90 (D-079); SIL re-run on regenerated data |
| `3fd3286` | **S2E1-prereg**: docs/S2E1_PREREG.md + configs/s2e1.yaml, interharmonic generator option, gridmon.s2e1, data script, tests (D-080..D-082) — before any S2E1 split existed |
| `886999e` | data generated at the prereg commit (reports/s2e1_data_summary.json) + training script |
| `ac850d8` | GW/GP trained and selected on train_wide/val_wide; models/s2e1, c/generated_s2e1, freeze/eval scripts, figures/report code |
| `b11b67a` | **S2E1-freeze**: configs/s2e1_frozen.yaml (model SHA-256, S2E1 method files, Stage 1 freeze ref, plan) |
| `e7d39e6` | **S2E1-results**: one evaluation at b11b67a -> reports/s2e1_results.json, S2E1_RESULTS.md, 3 figures |
| `9f4d674` | Website Experiment 2 section, README line, STAGE2_PLAN/PROGRESS/DECISIONS (D-083), CLAUDE.md |
| wrap-up | this handoff |
Stage 1 files: no method file touched; gridmon.frozen.verify and gridmon.s2e1.verify pass. Prompt says branch main; the session's
designated branch was used (nothing pushed to main).

## 2. Housekeeping (A2)
- rho gate 0.95 -> 0.90 (= dip_pu): SIL re-run (full regenerated data, ziglang 0.17.0 for -m32/msp430): 0 gated values outside
  tolerance on val/test_iid/test_shift/train; ungated test_iid 452/535 unchanged; golden 27/27, rules 100 %, GBM bit-exact.
- No LICENSE file in the repo -> nothing to mention or create (owner chooses).

## 3. Data (scripts/s2e1_make_data.py at 3fd3286, 78 s)
| split | n | seed | nuisance | checks |
|---|---|---|---|---|
| train_wide | 5,600 | 1101 | wide | margins hold; 3 D-039 oscillation fallbacks |
| val_wide | 1,750 | 2202 | wide | margins hold |
| test2_shift | 1,750 | 5005 | Stage 1 shift (by reference) | margins hold; 1 D-039 fallback; transient in 495 |
| test2_novel | 1,750 | 6006 | wide + interharmonic p 0.5 | margins hold; 1 D-039 fallback; interharmonic in 872 (8 with beat < 25 Hz), 70.6-939.9 Hz, >= 10.0 Hz from harmonics, 0.50-2.00 % |
| preview_wide / preview_novel | 70 / 140 | 9101 / 9102 | dev / dry run | margins hold |
Interharmonic = one balanced positive-sequence component, fixed frequency, whole record (D-080); Stage 1 data/manifests
bit-identical with the new generator code (preview_iid/preview_shift regenerated and compared; tested).

## 4. Arms (scripts/s2e1_train.py, 412 s)
- GW (37 features) and GP (23-feature physics subset, D-081): Stage 1 recipe unchanged; val_wide saturated again (31/33 and 33/33
  feasible configs within 0.002 of 1.000) -> both **20 stages x 4 leaves, lr 0.2, subsample 1.0, 13,972 B** (seeds 0-2).
- Float32 walker == sklearn (labels 100 %, 0 branch mismatches); C SIL with c/generated_s2e1/{gw,gp} tables: raw scores bit-exact
  on 1,750 val_wide records; pre-event-only control 0.133 (GW), 0.142 (GP) (<= 0.25).
- R, G0 = Stage 1 frozen; H0/HW = ML label if max p >= 0.5 else rules.

## 5. Headline (macro-F1, ML mean +/- sd over 3 seeds; reports/S2E1_RESULTS.md)
| arm | test2_shift | test2_novel | false alarms shift / novel |
|---|---|---|---|
| R rules | 0.9880 | 0.9937 | 0.4 % / 0.0 % |
| G0 Stage 1 GBM | 0.8581 +/- 0.0019 | 0.9396 +/- 0.0006 | 27.9 % / 18.5 % |
| GW wide GBM | 0.9960 | 0.9985 +/- 0.0003 | 1.2 % / 0.0 % |
| GP physics GBM | 0.9989 | 0.9994 | 0.0 % / 0.0 % |
| HW hybrid GW/R | 0.9977 | 0.9985 +/- 0.0003 | 1.2 % / 0.0 % |
| H0 hybrid G0/R | 0.9931 | 0.9937 | 0.8 % / 0.8 % |

CIs (paired bootstrap, 2,000 class-stratified resamples; arm - R): GW +0.0080 [+0.0028, +0.0136] / +0.0048 [+0.0008, +0.0089];
GP +0.0108 [+0.0063, +0.0159] / +0.0058 [+0.0023, +0.0098]; HW +0.0097 [+0.0051, +0.0148] / +0.0048 [+0.0008, +0.0089];
H0 +0.0051 [+0.0017, +0.0090] / +0.0000 [-0.0045, +0.0046]; GW - G0 +0.1379 [+0.1246, +0.1521] (shift).
**Decision rule verdict: GW "ML better", GP "ML better"** (primary); HW "ML better", H0 "ML matches" (secondary).
Survival of the interharmonic (CI lower >= -0.03): R, GW, GP, HW, H0 survive; G0 inconclusive. H1-H6 all supported.
Safety-critical confusions (phase_jump <-> freq_event, sag -> normal): 0 for every arm on both sets.

## 6. Top 3 findings in plain language
1. Trained on the conditions it will meet, the same 14 KB tree model no longer loses to the standards-based rules; on these synthetic
   sets it is slightly better (<= 1.1 macro-F1 points). Stage 1's "rules are more robust" becomes "ML needs representative training".
2. Stage 1 replicated exactly on fresh data (R 0.988, G0 0.858); the failure was training coverage (low SNR), not tree models.
3. The unseen interharmonic was mostly harmless (filtered by the one-cycle DFT/100 Hz envelope); oscillation look-alikes appeared only
   at 20-25 Hz beats (8 records), for the learned models except GP. A confidence gate rescued the mis-trained G0 (0.858 -> 0.993).

## 7. Website, docs, tests
- Results page: "Experiment 2 (Stage 2)" section at the end (headline, decision rule, comparisons, survival, SNR chart + table,
  plain-language findings; tokens/JS from site_data.js `s2e1`, exported from s2e1_results.json; no hand-typed numbers).
  build_site.py --check: WASM == native C 97/97; headless check: 0 console errors, no unresolved tokens, phone width OK.
- README results block: one S2E1 line (update_readme.py). STAGE2_PLAN item 3 done; PROGRESS; DECISIONS D-079..D-083; CLAUDE.md
  (rule 4 incl. test2_*, commands, layout). VALIDATION unchanged.
- pytest (full data): **165 passed, 0 skipped** (32 s; session 4: 148; new: 16 in tests/test_s2e1.py, 1 rho-gate test).
- Compute this session ~15 min (Stage 1 data 2.8 min, SIL 2 x 25 s, S2E1 data 78 s, training 7 min, evaluation 10 s, site 33 s).

## 8. Deviations / notes (D-083)
- Dry run used val_wide + preview_novel (pre-registered stand-ins) after training, before the freeze; no method change afterwards.
- Added and labelled as not pre-registered: G0 - R comparison, C SIL of the GW/GP tables, post-hoc cause of the rules' 6
  oscillation -> swell errors (gain error / transient). S2E1_RESULTS.md re-rendered with the findings after the run (no number changed).
- Stale hash: CLAUDE.md, RESULTS.md and the S2E1 prereg cite the M4-freeze as 6240f60; in this history it is b3afdec (CLAUDE.md fixed).

## 9. Open questions for the project lead
1. Validation saturates even with wide ranges (selection = smallest model). Use a harder selection split (e.g. 25-30 dB only) or
   a proper scoring rule (log-loss) in the next experiment?
2. GP looked more robust than GW (2 vs 7 errors on test2_shift, post hoc). Worth a dedicated sealed comparison, or adopt GP as the
   C2000 model candidate (same size, same C tables format) after such a test?
3. The rules' remaining errors are the D-035 half-cycle-RMS artefact at large phase jumps and calibration: fix the rule (new method,
   new sealed splits) or rely on HW (rules as fallback only)?
4. Next Stage 2 item: L3 LaunchPad PIL (hardware evidence) or item 4 (FLL, multi-event, long-stream false triggers)?
5. Publishing: merge this branch and enable Pages (docs/DEPLOY.md); the explorer still shows Stage 1 sealed records (D-077).

**Bring back:** tell the project lead S2E1 is done (the repo is public).
