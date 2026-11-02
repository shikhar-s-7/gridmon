# Compiling gridmon for the C2000 (C28x) — step by step (L2)

**What this is:** a check that the gridmon C code compiles for Texas Instruments C2000 microcontrollers
and how much memory it needs. It does **not** run anything on a board and does not touch any hardware.
**Why it runs on your computer:** the TI compiler could not be downloaded inside the cloud session (an
automated download from ti.com was not permitted there), so this one step runs locally. It takes about
15 minutes, most of it downloading.

**Status: done (2026-11-02).** Run with CCS 20.5.1 (cl2000 25.11.0.LTS) on Windows 11: every file compiles,
GBM on and off, no warnings; sizes in `reports/m7_c28x_size.json` (summary in docs/VALIDATION.md, L2, and
D-076). Re-run it after any change to `c/` or `firmware/c2000/`.

## What you need
- A Windows PC (macOS/Linux work the same way; use the Linux/macOS installer and `/` paths).
- Python with this repository set up (the same `.venv` as for the other scripts, see README.md).
- About 1 GB of free disk space.

## Steps
1. **Download the TI C2000 compiler.** Open https://www.ti.com/tool/C2000-CGT in a browser. Under
   "Downloads", choose the newest **LTS** version (for example 22.6.x.LTS) and the **Windows** installer
   (`ti_cgt_c2000_<version>_windows-x64_installer.exe`). TI may ask you to log in to a free myTI account and
   accept the license/export terms — that is normal.
   *(If you already have Code Composer Studio, the compiler is already on your PC, e.g. in
   `C:\ti\ccs1281\ccs\tools\compiler\ti-cgt-c2000_22.6.1.LTS` — skip to step 3.)*
2. **Install it.** Run the installer and accept the default folder, e.g. `C:\ti\ti-cgt-c2000_22.6.1.LTS`.
   Nothing else is needed (no IDE, no board drivers).
3. **Open a terminal in the repository folder** (in Explorer: open the `gridmon` folder, type `cmd` in the
   address bar, press Enter).
4. **Run the check** (adapt the folder to the version you installed):
   ```
   .venv\Scripts\python scripts\c2000_compile_check.py --cgt-root "C:\ti\ti-cgt-c2000_22.6.1.LTS"
   ```
   You will see one line per source file and configuration, for example
   `[gbm_on] gm_phasor.c: ok  code 812 / const 0 / data 0 words`, and at the end
   `-> ...\reports\m7_c28x_size.json`.
5. **Send back** `reports/m7_c28x_size.json` (commit it, or paste it into the project-lead chat). If a file
   failed, the error lines printed under it are the important part — send those too.

## What the script does (details)
- Reads `cl2000 --help` and keeps only the options this compiler version supports (the JSON lists any option
  it had to drop; cl2000 25.11 does not list `-v28`, which is the default target anyway).
- Passes the compiler's own header folder (`<cgt>/include`) itself, like CCS does. **You no longer need to set
  the `C2000_C_DIR` environment variable** (that was the workaround before the script fix of 2026-11-02).
- Requested options: `-v28 -ml -mt --float_support=fpu32 --tmu_support=tmu0 --abi=eabi -O2
  --fp_mode=relaxed --c99 --float_operations_allowed=32` (the last one makes the compiler reject any 64-bit
  `double` operation — the core is float-only).
- Compiles (no linking) every core file (`c/src`, `c/tables/gm_tables.c`, `c/generated/gm_model_gbm.c`) and the
  port layer (`firmware/c2000/gm_port.c`, `gm_port_fir.c`, `gm_port_c2000.c`) twice: with the GBM advisory model
  (`GM_ENABLE_GBM=1`) and without it (`GM_ENABLE_GBM=0`). Objects go to `build/c28x/` (not committed).
- Runs `ofd2000` on each object and reads its text listing ("Section Information", allocated sections only),
  adding up code (`.text`), constants (`.const`/`.econst`: DSP tables, GBM model, FIR taps) and data
  (`.bss`/`.data`/`.TI.noinit`). Sizes are in 16-bit words (the C28x memory unit); bytes = 2 x words. They are
  compile-only object sizes (an upper bound: the linker removes unused sections; the map file at L3 is final).
- `--dry-run` prints the compiler commands without running anything (useful to inspect the options).

## Result of the 2026-11-02 run (cl2000 25.11.0.LTS)
| configuration | code (words) | const (words) | data (words) |
|---|---|---|---|
| GBM on (default, D-070) | 13,785 | 10,913 | 0 |
| GBM off | 13,540 | 10,913 (model object still compiled; unreferenced at link time) | 0 |
Const = DSP tables 2,700 + GBM model 6,986 + FIR taps for all four input rates 1,124 + version string 103.
Data 0: the core and the port keep no static RAM (all state is in the caller's structs). The two earlier risks
did not materialise: C99 hexadecimal float constants compile with `--c99`, and nothing assumes 8-bit chars.
On a re-run, warnings about `#pragma` or unused functions are harmless; errors are not.
