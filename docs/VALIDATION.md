# Validation ladder (brief B13.3)

Simulation-only project. Nothing here is validated on measured data; nothing may be used for
protection or control. Status as of 2026-11-01.

| level | what | evidence | status |
|---|---|---|---|
| L0 | Python reference implementation (generator + DSP + rules), unit tests, golden vectors | [SIM] | in progress: M1 generator done; M2 baseline see PROGRESS.md |
| L1 | SIL: portable C99 float32 core on PC vs golden vectors and full val/test splits | [SIM] | not started (M6) |
| L2 | C28x compile with TI C2000-CGT (cl2000, FPU32/TMU), code/data size from the map file | — | not started (M7) |
| L3 | PIL on LAUNCHXL-F28379D: stored or UART-streamed records, cycle counts, bit-match vs L1 | [EXP] | Stage 2 (needs hardware) |
| L4 | Signal-level replay: DAC -> ADC loopback on the LaunchPad (two DACs; v_ca = -(v_ab + v_bc)) | [EXP] | Stage 2 |
| L5 | Monitor-only on a PCS board at low voltage, supervised | [EXP] | Stage 2 |
| L6 | Grid-connected, monitor-only | [EXP] | Stage 2+ |

Stage 1 = L0-L2. L3+ need hardware and qualified supervision (B13.5: TIDA-01606 is 800 V DC
nominal; TIEVM-HV-1PH-DCAC 380 V DC). The firmware module stays monitor-only until separately validated.
