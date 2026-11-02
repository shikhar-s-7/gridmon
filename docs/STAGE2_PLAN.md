# Stage 2 plan — gridmon

Stage 1 (M1-M8) ended with a frozen, evaluated method pair, a verified portable C core, a C2000 port layer that
compiles for C28x (L2) and a static website. Everything is [SIM]; nothing has run on hardware. This plan lists the
next steps in priority order. Effort: **S** = days, **M** = 1-3 weeks, **L** = a month or more (one person,
part-time hardware access). Items 1-2 are the critical path to any hardware claim; item 3 is the critical path to
any new accuracy claim.

| # | item | effort | depends on |
|---|---|---|---|
| 1 | Single-phase variant (TIDM-HV-1PH-DCAC) + three-board architecture with a separate three-phase monitor node | L | — |
| 2 | L3 LaunchPad PIL, then L4 DAC -> ADC replay | M + M | LaunchPad, CCS |
| 3 | Robustness experiment with NEW sealed splits | M | — |
| 4 | FLL, multi-event records, capture merging, long-stream false-trigger study | M | 3 (new data) |
| 5 | EMT / Simulink cross-check of the generator | M | a PSCAD/Simulink model |
| 6 | CNN on an NPU device (F28P55x) via TI's model tools | L | 3, F28P55x LaunchPad |
| 7 | L5 / L6 monitor-only on PCS hardware | L | 2, safety prerequisites |
| 8 | Measured data | L (ongoing) | 7 or an external data source |

## 1. Single-phase variant and the three-board architecture (L)
- **Goal:** a 1-phase classifier for TIDM-HV-1PH-DCAC-class inverters (6 classes: normal, sag, swell, phase jump,
  frequency event, oscillation — "unbalanced" does not exist on one phase), with a SOGI-FLL front end (frequency-adaptive,
  replacing the fixed-w0 DSOGI of D-037), its own features (single-phase RMS, SOGI amplitude/angle, jump metric, f_dft),
  rules, sealed splits and C core variant. For three single-phase boards 120 deg apart, keep the unchanged three-phase core
  on a **separate monitor node** (LaunchPad with isolated L-N sensing of all three phases, input_ln = 1, calibrated),
  independent of the inverter controllers (docs/C2000_INTEGRATION.md section 6).
- **Proves:** that the approach transfers to single-phase hardware; gives the three-board setup a monitor that sees the
  sequence quantities.
- **Prerequisites:** generator option for one LN channel; new DSP_SPEC section; new sealed splits (new seeds).
- **Risks:** without V2 the sag/swell vs oscillation distinctions lose features; a 1-phase PLL/FLL reacts to phase jumps
  differently (re-check the Blue Cut confusion first).

## 2. L3 processor-in-the-loop, then L4 signal replay (M + M)
- **Goal (L3):** run firmware/c2000/pil_launchpad on a LAUNCHXL-F28379D: the 3 embedded golden records give the host (L1)
  labels; measure cycles per record with and without TMU mapping (GM_SINF etc.) and `--fp_mode=relaxed`; read the linker
  map for the final flash/RAM (L2 sizes are compile-only upper bounds, D-076). Replaces the [ASSUMED] 20-30 ms record time
  and the 50 ms background budget (D-073).
- **Goal (L4):** two DACs generate v_ab and v_bc (v_ca = -(v_ab + v_bc) in a 3-wire system) from stored records, looped back
  into the ADCs; the port layer (gm_port_push in a real ISR at 50 kHz) triggers, captures and classifies in real time.
- **Proves:** L3 — the code runs on the target and matches L1 bit-for-bit or within tolerance; L4 — the real-time path (ADC,
  calibration, decimation, ring buffer, trigger, placement) works on analogue signals with real ISR timing.
- **Prerequisites:** LaunchPad, CCS 20+, C2000Ware; for L4 an analogue front end/level shifting; a host script to stream records.
- **Risks:** TMU per-unit angle conventions (C_CORE.md section 2); DAC resolution/bandwidth; ADC noise worse than the
  simulated sensor model.

## 3. Robustness experiment with new sealed splits (M)
- **Goal:** test the Stage 1 explanations of the shift failure (RESULTS.md findings 3 and 7, "what would change the conclusion"):
  (a) noise/calibration-augmented GBM (train on SNR 25-60 dB, +/-2 % gain), (b) physics-only / monotonic-constrained GBM,
  (c) confidence-gated hybrid (GBM only above a probability threshold, rules otherwise — chosen post hoc, so it must be
  pre-registered), (d) a shift-like validation split for model selection (VAL was saturated, D-043), (e) a **second unseen
  nuisance** in a new test set (e.g. sensor bandwidth limit, sub-synchronous interharmonics, CT/VT phase error).
- **Proves:** whether "rules are more robust" survives a fair ML training setup, or becomes "ML needs representative training
  conditions".
- **Prerequisites:** NEW sealed test splits with new seeds (test_shift is no longer blind, CLAUDE.md rule 4); a pre-registered
  evaluation plan and freeze, as in M4.
- **Risks:** designing the new nuisance with knowledge of the old failure (keep it independent and documented beforehand).

## 4. FLL, multi-event records, capture merging, long-stream false-trigger study (M)
- **Goal:** SOGI-FLL instead of fixed w0 (removes the 100/300 Hz PLL ripple, D-037); records with two events or an event at the
  record edge (labels: primary event + flags); merge port-layer captures that belong to one event (D-072); run the port layer
  on hours of normal-only streams (with hard negatives and slow drifts) to measure false triggers per hour and tune the
  envelope trigger threshold (D-071, 1.5 % vs the 1.0 % rule).
- **Proves:** behaviour outside the single-event, 1 s record assumption; a false-alarm rate per hour for operation.
- **Prerequisites:** stream generator extensions (multi-event), new labels and splits; changes are new methods (new freeze).
- **Risks:** label definitions for overlapping events; long-stream generation time (vectorised chunks exist).

## 5. EMT / Simulink cross-check of the generator (M)
- **Goal:** reproduce a sample of each class (dips with fault impedance, transformer connections, capacitor switching, a
  frequency event from a generator trip) in an EMT tool (PSCAD or Simulink/Simscape Electrical), measure LL voltages at a
  PCS terminal, feed them to the frozen methods.
- **Proves:** that the phasor-domain generator (Bollen dip types, simplified transients) does not miss waveform features that
  matter (point-on-wave effects, saturation, converter interaction).
- **Prerequisites:** a licensed EMT tool and a small network model; strong-grid and weak-grid cases.
- **Risks:** EMT models also embed assumptions; disagreements need careful attribution.

## 6. CNN on an NPU device (F28P55x) (L)
- **Goal:** the optional raw-waveform 1-D CNN of the brief (B1/B11 D-011), trained offline on the new splits of item 3, quantised
  and deployed to the F28P55x neural-network accelerator with TI's model-zoo / edge-AI tooling; compared with rules and GBM
  on the same sealed data, including cost (cycles, flash, RAM).
- **Proves:** whether a raw-waveform model adds robustness the feature-based GBM lacks, at what embedded cost.
- **Prerequisites:** item 3 data; PyTorch; F28P55x LaunchPad; TI's NPU toolchain.
- **Risks:** shortcut learning is likely worse for raw-waveform models; quantisation effects; vendor tool maturity.

## 7. L5 / L6 monitor-only on PCS hardware (L)
- **Goal:** L5 — gm_port in the control firmware of a PCS board at low voltage, supervised, monitor-only (no output used);
  L6 — grid-connected, monitor-only, telemetry logged next to an independent power-quality analyser.
- **Proves:** integration into a real control ISR without disturbing it (ISR timing margin), real sensor noise and
  calibration, real events.
- **Prerequisites (safety):** L3 and L4 passed; TIDA-01606 sensing (LN vs LL) and ISR names confirmed from the DigitalPower
  SDK source (D-074); qualified supervision for high-voltage hardware (TIDA-01606 800 V DC, TIEVM-HV-1PH-DCAC 380 V DC);
  isolated debug connections; written test plan; the module stays monitor-only (no control or protection use).
- **Risks:** ISR overrun under full control load; electrical safety; ground loops in debug connections.

## 8. Measured data (L, ongoing)
- **Goal:** labelled records from real PCS terminals or public power-quality datasets (with sampling rate/sensor information),
  converted with the website's CSV path or a batch tool, evaluated with the frozen methods and with item 3's methods.
- **Proves:** the first [EXP] evidence; whether the synthetic shift resembles real-world shift.
- **Prerequisites:** data agreements, labels from an independent reference (PQ analyser event logs), anonymisation.
- **Risks:** label quality; distribution far from the synthetic classes (multi-event, unknown causes).
