# Validation ladder (brief B13.3)

Simulation-only project. Nothing here is validated on measured data; nothing may be used for
protection or control. Status as of 2026-11-01 (session 2: R1, M3, M4).

| level | what | evidence | status |
|---|---|---|---|
| L0 | Python reference implementation (generator + DSP + rules + gradient boosting), unit tests, golden vectors, sealed evaluation | [SIM] | **Complete (incl. ML).** M1+M2 (session 1) and R1 regeneration; M3: GBM-embedded exported to JSON + C const tables (13,972 B), float32 reference walker == sklearn on val (max proba diff 2e-7, labels 100 %, 0 branch mismatches); M4: frozen methods evaluated once on test_iid/test_shift (reports/RESULTS.md); 81 pytest tests pass |
| L1 | SIL: portable C99 float32 core on PC vs golden vectors and full val/test splits | [SIM] | not started (M6, Prompt 03). Pre-checks only: c/generated/gm_model_gbm.{c,h} compile with gcc -std=c99 -pedantic -Wdouble-promotion -Wfloat-conversion; an uncommitted throwaway C walker over those tables matched the Python float32 walker bit-exactly on all 1,750 val records |
| L2 | C28x compile with TI C2000-CGT (cl2000, FPU32/TMU), code/data size from the map file | — | not started (M7) |
| L3 | PIL on LAUNCHXL-F28379D: stored or UART-streamed records, cycle counts, bit-match vs L1 | [EXP] | Stage 2 (needs hardware) |
| L4 | Signal-level replay: DAC -> ADC loopback on the LaunchPad (two DACs; v_ca = -(v_ab + v_bc)) | [EXP] | Stage 2 |
| L5 | Monitor-only on a PCS board at low voltage, supervised | [EXP] | Stage 2 |
| L6 | Grid-connected, monitor-only | [EXP] | Stage 2+ |

Stage 1 = L0-L2. L3+ need hardware and qualified supervision (B13.5: TIDA-01606 is 800 V DC
nominal; TIEVM-HV-1PH-DCAC 380 V DC). The firmware module stays monitor-only until separately validated.
