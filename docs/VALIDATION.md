# Validation ladder (brief B13.3)

Simulation-only project. Nothing here is validated on measured data; nothing may be used for
protection or control. Status as of 2026-11-02 (session 4: L2 integrated, M8).

| level | what | evidence | status |
|---|---|---|---|
| L0 | Python reference implementation (generator + DSP + rules + gradient boosting), unit tests, golden vectors, sealed evaluation | [SIM] | **Complete (incl. ML).** M1+M2 (session 1) and R1 regeneration; M3: GBM-embedded exported to JSON + C const tables (13,972 B), float32 reference walker == sklearn on val (max proba diff 2e-7, labels 100 %, 0 branch mismatches); M4: frozen methods evaluated once on test_iid/test_shift (reports/RESULTS.md); 81 pytest tests pass |
| L1 | SIL: portable C99 float32 core on PC vs golden vectors and full val/test splits | [SIM] | **Done (M6, session 3).** c/ core (gridmon-core 0.6.0, ctx 15,816 B, const tables 5,400 B + model 13,972 B), scripts/run_sil.py -> reports/m6_sil.json: 27/27 golden vectors (every trace, feature, label, reason); rule labels 1750/1750 on val, test_iid and test_shift and 560/560 train (every 10th); GBM on the same C features bit-exact (max |proba diff| 1.2e-7), end-to-end 100 %; features within FEATURES.md tolerances except rho on test_iid: 2 values ungated, 1 under the D-069 SIL gate (float32 argmin tie, D-060); NaN/validity pattern identical (rocof_end, D-051); ASan/UBSan clean; static C28x rules, -m32 and 16-bit-int compiles pass; ~1.3 ms/record on the host. Sealed splits read for equivalence only |
| L2 | C28x compile with TI C2000-CGT (cl2000, FPU32/TMU), code/data size from the map file | — | **Done (2026-11-02, user's local run, D-076).** CCS 20.5.1, cl2000 25.11.0.LTS, `-ml -mt --float_support=fpu32 --tmu_support=tmu0 --abi=eabi -O2 --fp_mode=relaxed --c99 --float_operations_allowed=32`: all 15 core + port files compile, GBM on and off, no warnings. Compile-only objects (16-bit words): code 13,785 (GBM off 13,540), const 10,913 (DSP tables 2,700, GBM model 6,986, FIR taps 1,124), data 0 (reports/m7_c28x_size.json). Linked/map-file sizes come with L3 |
| L3 | PIL on LAUNCHXL-F28379D: stored or UART-streamed records, cycle counts, bit-match vs L1 | [EXP] | Stage 2 (needs hardware); UNTESTED template ready: firmware/c2000/pil_launchpad (3 golden records, L1 expectations, cycle count) |
| L4 | Signal-level replay: DAC -> ADC loopback on the LaunchPad (two DACs; v_ca = -(v_ab + v_bc)) | [EXP] | Stage 2 |
| L5 | Monitor-only on a PCS board at low voltage, supervised | [EXP] | Stage 2 |
| L6 | Grid-connected, monitor-only | [EXP] | Stage 2+ |

Stage 1 = L0-L2. L3+ need hardware and qualified supervision (B13.5: TIDA-01606 is 800 V DC
nominal; TIEVM-HV-1PH-DCAC 380 V DC). The firmware module stays monitor-only until separately validated.

M7 host evidence (not a ladder level): the port layer (calibration, LN->LL, polyphase decimation, ring buffer,
trigger, record placement, telemetry) on a 120 s, 50 kHz stream with 14 events: 12/12 events captured, onset at
0.225-0.288 s of the record, rules and GBM 12/12, 0 false triggers, 0 overruns (reports/m7_port_stream.json;
docs/C2000_INTEGRATION.md section 7). What L3-L6 prove: docs/C2000_INTEGRATION.md section 8.
