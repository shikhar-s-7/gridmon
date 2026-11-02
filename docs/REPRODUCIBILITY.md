# Reproducibility record (brief B10.1)

## Clean-clone check, 2026-11-02 (session 4, Linux cloud VM)
Fresh `git clone` of branch `claude/new-session-7mqosi` at commit `ca85e6a` into an empty temporary directory, new
virtual environment, nothing copied from the working repository. Machine: Linux 6.18 x86-64, 4 vCPU, Python 3.12.3,
gcc 13.3 (system `cc`), Node 22 (only for the optional site checks). Script: one shell script that runs each step and
logs its exit code and wall time.

| step | command | result | time |
|---|---|---|---|
| clone | `git clone --branch claude/new-session-7mqosi ...` | ok | 2 s |
| venv | `python3.12 -m venv .venv` | ok | 3 s |
| dependencies | `.venv/bin/python -m pip install -r requirements.txt` | every pin installed unchanged | 33 s |
| package | `.venv/bin/python -m pip install -e .` | ok | 4 s |
| pipeline | `.venv/bin/python scripts/run_all.py --fast` | ok: make_dataset 14.1 s, run_baseline 8.9 s, make_golden 0.6 s, run_gbm 29.4 s, evaluate_test 8.0 s | **63 s** (limit 10 min) |
| tests | `GRIDMON_TEST_FAST=1 .venv/bin/python -m pytest` | **144 passed, 4 skipped** (reasons below) | 27 s |
| C core | `.venv/bin/python scripts/build_c.py` | ok (shared library, CLI, ASan/UBSan CLI, compile checks) | 3 s |
| SIL | `.venv/bin/python scripts/run_sil.py --fast` | golden 27/27, sanitizer pass, rule labels 175/175 (val, test_iid, test_shift) and 560/560 (train), GBM bit-exact, features within tolerance | 12 s |
| website | `.venv/bin/python scripts/build_site.py` | ok: 6 pages, 5.76 MB; export skipped automatically (no full data in a fast clone, the committed web/src/data was reused); `ziglang` installed on demand | 30 s |
| total | | | **~3 min** |

Skipped tests in the clone and why:
- 3 tests need `ziglang` (the 32-bit compile check, because the VM's gcc has no 32-bit headers, and the 2 WebAssembly
  tests). `build_site.py` installs `ziglang` on demand; re-running `tests/test_web.py tests/test_c_portability.py`
  afterwards in the same clone: **15 passed, 0 skipped**.
- 1 test (`tests/test_gbm.py:136`) compares against the full-data `data/val/features.csv`, which a `--fast` run does not
  create; it runs after `scripts/run_baseline.py` on full data (it passes in the working repository).

Website determinism: the rebuilt `web/dist` differs from the committed one only in the build-commit stamp (6 pages:
`commit e9cc0e2` -> `ca85e6a`, plus `build_info.json`); the WebAssembly module is bit-identical (SHA-256 `28beebd6...`,
same as reports/m8_wasm_equivalence.json), and so is all exported data (it was reused unchanged).

Packaging problems found: none in the clean clone. One problem was found and fixed just before it (commit `ca85e6a`):
the generic `data/` and `dist/` lines in `.gitignore` also matched `web/src/data` and `web/dist`, so the exported site
data and the built site were missing from the M8 commit; the rules are now anchored to the repository root and a test
(`tests/test_web.py::test_site_files_are_tracked_by_git`) guards it.

## Full pipeline (not part of the clean-clone check)
`scripts/run_all.py` on full data takes about 10 minutes on 4 vCPU (session 2: dataset 145 s, baseline 45 s, golden 1 s,
run_gbm 290 s, evaluate_test 26 s; in full mode run_gbm keeps the frozen models and evaluate_test re-runs the frozen
evaluation). Session 4 full-data steps in the working repository: make_dataset 123 s (session 3), run_sil 37-59 s,
build_site with export 25 s, pytest (full data) about 1 minute.

## Earlier reproductions
- Session 1 (Windows 11, Python 3.12.9) produced the original data and results; session 2 (Linux) reproduced M1/M2
  exactly: identical validation macro-F1 and confusion, bit-identical int16 inputs of all 27 golden vectors, float64 traces
  within 3e-14 (docs/ENVIRONMENT.md; the AA FIR file is kept platform-stable, D-038).
- 2026-11-02: the user ran `scripts/c2000_compile_check.py` on Windows 11 with CCS 20.5.1 (cl2000 25.11.0.LTS): all files
  compile (L2, D-076) — the repository's scripts also run on Windows.

## Determinism rules (CLAUDE.md)
NumPy generators from SeedSequence only (no global RNG); every data split records its config hash and git commit in
meta.json; models, golden vectors, generated C tables and reports are committed; the C constant tables are regenerated and
compared byte for byte by the tests (`gen_c_tables.py --check`).
