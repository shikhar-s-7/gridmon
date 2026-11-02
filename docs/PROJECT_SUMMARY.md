# gridmon — project summary (Stage 1)

*Simulation-only research prototype. Not validated on measured data. Not for protection or control.*

## The problem
Battery energy storage systems connect to the grid through an inverter (a power conversion system, PCS). When the grid
voltage misbehaves, the inverter must react — and the right reaction depends on what happened: a voltage sag calls for
ride-through and reactive current, a swell for absorbing reactive power, a frequency drop for frequency response, a slow
voltage wobble for an alarm. A sudden jump in the voltage's phase angle needs no frequency response at all, yet in a
2016 California event many inverters mistook exactly that for a frequency problem and tripped. So the question is:
can a small monitor reliably tell these situations apart from one second of voltage measurements — and should it use
hand-written, standards-based rules or machine learning?

## The approach
1. **A synthetic data generator** produces one-second, three-phase voltage records at 5 kHz with seven classes (normal,
   balanced sag, unbalanced sag, swell, phase jump, frequency event, oscillation), realistic background disturbances, sensor
   errors and noise, and margins so that labels are unambiguous. Two test sets were sealed before anything was tuned: one
   like the training data, one deliberately shifted (more noise, calibration errors, an unseen transient).
2. **Shared signal processing** turns each record into 37 physics features (RMS levels, sequence components, a phase-jump
   metric, frequency and its rate of change, a PLL, an envelope spectrum, harmonic distortion), specified precisely enough
   to be re-implemented in C.
3. **Two classifiers on the same features:** priority-ordered rules with thresholds from power-quality standards, and a
   gradient-boosting tree model limited to fit a microcontroller (13.6 KB).
4. **One sealed evaluation:** both methods were frozen, then evaluated once on the two test sets, with confidence intervals.
5. **Porting:** one C99 float32 core reproduces the Python reference and was verified record by record; a monitor-only
   layer for TI C2000 microcontrollers adds filtering, buffering, triggering and telemetry; the same core runs in a web
   browser.

## What was built
- Python reference: generator, signal processing, rules, tree model, evaluation, reports (`src/gridmon`, `reports/`).
- C core (`c/`): no dynamic memory, all state in a 15.8 KB caller-provided structure, constants generated from Python;
  it reproduces the Python rule labels on every one of thousands of test records.
- C2000 port layer (`firmware/c2000/`): runs inside the inverter's control interrupt at 20-100 kHz at a cost of 45
  multiply-adds per sample, never writes a control variable; tested on a 120 s simulated stream; compiles with TI's
  compiler without warnings; a LaunchPad test program is prepared but not yet run.
- Website (`web/`): results, a record explorer and a "try it" page where the C core runs in the browser.

## The results (all simulated)
- **On test data like the training data, rules and tree model are equally good** (both close to perfect; no significant
  difference).
- **Under the shifted conditions, the rules are significantly better.** The rules barely change; the tree model misses most
  phase jumps and raises false alarms on normal records, because it learned cues that only work at the training noise
  level. A larger model was not better.
- **Neither the rules nor the small tree model ever confused a phase jump with a frequency event**, while a naive PLL-based
  frequency check flags every phase jump — the failure mode of the 2016 event.
- **The rules' few errors come from the measurement chain** (calibration errors, transients), which is why calibration is a
  mandatory step in the C2000 integration guide.
- **The C core is a faithful copy:** same rule labels on all val/test records, model outputs bit-exact on the same features,
  verified also in WebAssembly.

## What it shows — and what it does not
It shows that, *on this physics-based synthetic data*, transparent standards-based rules are as accurate as a small learned
model and far more robust to a plausible shift, and that the whole monitor fits a microcontroller and can be verified
without hardware. It does **not** show that the rules would be robust on real measurements (the generator and the rules were
designed by the same people), that a better-trained model could not close the gap, or anything about behaviour on real
hardware: there is no measured data and nothing has run on a microcontroller or an inverter yet.

## Next steps (docs/STAGE2_PLAN.md)
Run the LaunchPad test and a signal-replay test (first hardware evidence); build the single-phase variant; repeat the
rules-vs-ML comparison with new sealed test sets and a fairly trained model (noise-augmented, physics-only, confidence-gated
hybrid); cross-check the generator with an electromagnetic-transient simulation; and only then, under qualified supervision,
monitor-only tests on inverter hardware and with measured data.
