# HANDOFF — gridmon session 3 (PROMPT_03: M6 C core + SIL, M7 C2000 port layer), 2026-11-01

Simulation-only; every result is [SIM]. Previous handoffs: docs/handoffs/ (session 1, session 2).

## 1. Status, commits, runtimes
- Done: **M6** (portable C99 float32 core, SIL equivalence), **M7** (monitor-only C2000 port layer, host stream
  simulation, integration guide, LaunchPad PIL template). L2 (cl2000 compile) = script ready, **run locally** (sec. 5).
- Commits (branch claude/new-session-7mqosi, author shikhar-s-7, D-053): M6 `8100769`, M7 `1e9438a`, wrap-up = next.
- Methods unchanged: `gridmon.frozen.verify()` passes (no frozen file touched, c/generated/* read-only).
- Runtimes (4 vCPU Linux VM): data regeneration 123 s; `build_c.py` ~15 s; `run_sil.py` 32-37 s (all splits);
  `run_port_stream.py` 12-14 s; `pytest` 52 s: **141 passed, 0 failed, 0 skipped** (81 old + 60 new).

## 2. M6 — SIL table (reports/m6_sil.json; sealed splits read for equivalence only, no labels loaded)
| set | n | features within tol. (all records) | values outside | rule labels (= reasons) | GBM (a) same C features | GBM (b) end to end |
|---|---|---|---|---|---|---|
| golden vectors | 27 | 37/37, every trace within golden/README tolerances | 0 | 27/27 | (selftest rows: = sklearn) | — |
| val | 1750 | 37/37 (worst t_fpll_05 0.48 x tol) | 0 | 1750/1750 | bit-exact, max dp 1.2e-7, raw diff 0 | 1750/1750 |
| test_iid | 1750 | 36/37 (rho) | 2 (rho) | 1750/1750 | bit-exact, 6.0e-8 | 1750/1750 |
| test_shift | 1750 | 37/37 (worst t_fpll_05 0.48 x tol) | 0 | 1750/1750 | bit-exact, 6.0e-8 | 1750/1750 |
| train (every 10th) | 560 | 37/37 | 0 | 560/560 | bit-exact, 7.5e-9 | 560/560 |
- Rule disagreements: **none** (target 100 % met everywhere). The 2 rho values: float32 tie of |V1| at the OBS minimum
  (first-argmin m* moves by 1-3 samples, V2 noise 2-7e-4 pu / floored denominator 0.01); both records are non-dips
  (urms_min_1 >= 0.94), so rho is not used by the rules and GBM labels agree (D-060, C_CORE.md sec. 9).
- rocof_end validity (D-051): the C flag is invalid exactly where Python is NaN (12 val, 4 test_iid, 23 test_shift,
  2 train records; NaN-pattern mismatches 0).
- Worst golden trace errors vs tolerance: f_pll 1.8e-4 (3e-3), ROCOF 2e-5 (1e-3), e_ang 1.8e-6 (1.8e-4), THD 2.6e-8 (1e-6).
- Memory: sizeof(gm_ctx_t) **15,816 B** (7,908 16-bit words; budget 32 KB; record buffer is the caller's 30,000 B);
  const tables: DSP 5,400 B + GBM model 13,972 B; host code 33.9 KB (x86-64 -O2). Host time **~1.3 ms/record**
  (1.2-1.7 ms, indicative). Sanitizer build (ASan+UBSan, no recover): all 27 golden vectors clean.
- Portability: static checks pass (no double, no char/int8 data, no plain int, no malloc/stdio, no static/global
  mutable state, all float literals f, no punning, no recursion, sizeof only in gm_ctx_words); ILP32 build (zig cc
  x86-linux-musl: the VM gcc lacks 32-bit headers) and a 16-bit-int build (zig msp430, stub math.h) pass with -Werror.

## 3. M7 — port layer and stream (reports/m7_port_stream.json, reports/figures/m7_port_stream.png)
- ISR path: trim (D-052) -> LN->LL (optional) -> polyphase FIR -> 5 kHz int16 ring (1.6 s, 48,000 B); background:
  trigger, placement, gm_process_record, telemetry. gm_port_t 50,352 B on the host (~25 K words on C28x incl. ring).
- ISR MACs per input sample (3 channels): **45** at every rate (39 FIR + 6 trim/LN): 20 kHz L 49/D 4, 40 kHz 99/8,
  50 kHz 123/10, 100 kHz 245/20, all K = 13, stopband 53.4-53.9 dB; 0.90 / 1.80 / 2.25 / 4.50 M MAC/s.
  C decimator = float64 reference within 0.51 LSB (2^-14 pu) at every rate.
- Stream (seed 7007, 120 s at 50 kHz, 14 events = 2 per class, iid parameters + margins, 6.65 s gaps each with a
  hard-negative episode, SNR 44 dB + 12-bit ADC at 50 kHz, iid gain/offset removed by the trim; background every
  1 ms, blocked 50 ms per record):
  | outcome | value |
  |---|---|
  | events captured | **12/12** (latency Urms/J 4-15 ms, oscillation 66-114 ms, frequency 181-215 ms) |
  | true onset inside the captured record | 0.225-0.288 s (mean 0.276; target 0.28) |
  | rule label = truth / GBM advisory = truth | 12/12 / 12/12 (GBM max proba 0.94-0.96) |
  | normal-class episodes triggered / gap hard negatives triggered | 0/2 / 0/14 |
  | false triggers / secondary captures | 0 / 3 (frequency recoveries at 0.2 Hz/s, classified normal) |
  | overruns / resyncs / clipped records | 0 / 0 / 0 |
  | LN input mode | same outcome for 14/14 events |
  Integration test only (n = 14), not an accuracy result.
- Tests: ring overrun never blocks the ISR and is counted; resync after a 3 s background stall; calibration trim
  undoes a +2 % / 0.01 pu error; stream option leaves record generation bit-identical (val records, golden unit signals).

## 4. Files added
c/include/gridmon.h, c/src/gm_*.c (+gm_internal.h), c/tables/gm_tables.{h,c} (generated), c/host/{gm_cli.c,
gm_sil.c, gm_names.h}; firmware/c2000/{gm_port.h,gm_port.c,gm_port_fir.{h,c},gm_port_c2000.c,host/gm_port_sim.c,
pil_launchpad/{main.c,pil_records.{h,c},README.md}}; scripts/{gen_c_tables,build_c,run_sil,run_port_stream,
c2000_compile_check,make_pil_records}.py; src/gridmon/{c_build,c_sil,c_portability,port_sim,port_stream}.py,
generator/stream.py; tests/{test_c_sil,test_c_portability,test_port_sil}.py; docs/{C_CORE,C2000_INTEGRATION,
C2000_BUILD}.md. Changed: tests/test_freeze.py (run_sil.py is the second allow_test caller, CLAUDE.md rule 4),
DSP_SPEC/FEATURES (clarifications only), DECISIONS, VALIDATION, PROGRESS, ENVIRONMENT, README, CLAUDE.md.

## 5. C28x compile and size (L2) — exact blocker
TI C2000-CGT was **not installed**: the scripted fetch of https://www.ti.com/tool/C2000-CGT was refused by this cloud
session's permission policy (auto-mode classifier), and per PROMPT_03 A4 it was not worked around (D-067).
Ready for a local run: `python scripts/c2000_compile_check.py --cgt-root "C:\ti\ti-cgt-c2000_<ver>.LTS"` (steps in
docs/C2000_BUILD.md, plain language). It checks every flag against `cl2000 --help` (-v28 -ml -mt
--float_support=fpu32 --tmu_support=tmu0 --abi=eabi -O2 --fp_mode=relaxed --c99 --float_operations_allowed=32),
compiles core + port with GBM on/off and writes per-object code/const/data words to reports/m7_c28x_size.json.
Risks to watch: C99 hex float literals (needs --c99), CHAR_BIT 16 (only cl2000 proves it).

## 6. Deviations and interpretations (docs/DECISIONS.md)
D-049..D-054 as specified by the review. M6: D-055 record-level multi-pass core (15.8 KB ctx); D-056 angU via an
integer turn count (same value, no float32 accumulation); D-057 THD partial sums; D-058 directed rounding of
thresholds; D-059 validity flags (NO_SIGNAL, THD zero fundamental beyond the reference's domain); D-060 SIL scope
and the rho argmin tie; D-061 toolchain (-ffp-contract=off, -Wvla, ziglang for ILP32/16-bit-int). M7: D-062 ring and
polyphase data path (LSB 2^-14, drop-newest), D-063 taps per f_in, D-064 trigger details, D-065 stream generator
option, D-066 stream evaluation conventions, D-067 L2 blocker, D-068 PIL template.

## 7. Open questions for the project lead
1. rho float32 argmin ties (2/1750 test_iid, non-dips): accept as a documented difference (current), or add a gate
   to FEATURES.md (e.g. compare rho only when urms_min_1 < 0.95)? A gate is a documentation change, not a method change.
2. GM_ENABLE_GBM on C2000: default 1 (advisory, +7 K words flash, estimated < 0.1 ms per record). Keep on for logging, or off to save flash?
3. Envelope trigger 1.5 % (prompt) vs oscillation rule 1.0 %: keep, or lower to 1.0 % after a false-trigger study
   on longer normal streams (Stage 2)?
4. Secondary captures during frequency recovery (3 in 120 s): merge captures within an event / rate-limit, or keep
   every capture (current; they are labelled normal)?
5. Frequency recovery model (0.2 Hz/s after a 1 s hold) and the 50 ms record-time budget are [ASSUMED]; replace by
   measured values at L3 (cycles) and by a grid-code-based recovery profile if needed.
6. L2: please run scripts/c2000_compile_check.py locally (docs/C2000_BUILD.md) and bring back reports/m7_c28x_size.json.
7. TIDA-01606 sensing (LN vs LL) and ISR variable names are [ASSUMED] from the design summary; confirm from the SDK.

## 8. Figures
- reports/figures/m7_port_stream.png: |V1| and frequency over the 120 s stream, event windows, primary and secondary
  captures, labels "truth -> rule label".
- Earlier figures unchanged (m1-m4).

**Bring back to the project-lead chat:** docs/HANDOFF.md, reports/m6_sil.json, reports/figures/m7_port_stream.png
