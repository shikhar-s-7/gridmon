"""M7 / L2: compile the portable C core and the port layer for TI C28x with cl2000 and report code/data sizes.

The TI C2000 Code Generation Tools (C2000-CGT, free, ti.com) are NOT installed in the cloud VM (the
automated download was not possible there), so this script is for a local machine. Steps for Windows:
docs/C2000_BUILD.md. Run by the user on 2026-11-02 with CCS 20.5.1 / cl2000 25.11.0.LTS: every file
compiles, GBM on and off, no warnings (reports/m7_c28x_size.json, D-076).

What it does
  1. finds cl2000 (argument, $C2000_CGT_ROOT, or PATH) and reads `cl2000 --help` to keep only supported flags
  2. compiles c/src/*.c, c/tables/gm_tables.c, c/generated/gm_model_gbm.c, firmware/c2000/gm_port*.c
     (compile only, no link) twice: GM_ENABLE_GBM=1 and GM_ENABLE_GBM=0; the compiler's own include
     directory (<cgt>/include, added automatically by CCS) is passed explicitly, so no C2000_C_DIR is needed
  3. reads every object's allocated sections (alloc = Y) from the ofd2000 text listing ("Section
     Information") and writes reports/m7_c28x_size.json (16-bit words = C28x addressable units, and bytes)
Wanted flags (dropped with a note if this cl2000 does not list them):
  -v28 -ml -mt --float_support=fpu32 --tmu_support=tmu0 --abi=eabi -O2 --fp_mode=relaxed --c99
  --float_operations_allowed=32 (reject any 64-bit double operation) --diag_warning=225 --display_error_number

Usage:
  python scripts/c2000_compile_check.py --cl2000 "C:/ti/ti-cgt-c2000_22.6.1.LTS/bin/cl2000.exe"
  python scripts/c2000_compile_check.py --cgt-root "C:/ti/ti-cgt-c2000_22.6.1.LTS"
  python scripts/c2000_compile_check.py --dry-run          (print the commands; no compiler needed)
Exit code 0 = every file compiled in both configurations.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (sorted((ROOT / "c" / "src").glob("*.c")) + [ROOT / "c" / "tables" / "gm_tables.c",
                                                       ROOT / "c" / "generated" / "gm_model_gbm.c"]
           + [ROOT / "firmware" / "c2000" / n for n in ("gm_port.c", "gm_port_fir.c", "gm_port_c2000.c")])
INCLUDES = [ROOT / "c" / "include", ROOT / "c" / "tables", ROOT / "c" / "generated", ROOT / "c" / "src",
            ROOT / "firmware" / "c2000"]
WANTED = ["-v28", "-ml", "-mt", "--float_support=fpu32", "--tmu_support=tmu0", "--abi=eabi", "-O2",
          "--fp_mode=relaxed", "--c99", "--float_operations_allowed=32", "--diag_warning=225",
          "--display_error_number"]
VARIANTS = {"gbm_on": ["--define=GM_ENABLE_GBM=1"], "gbm_off": ["--define=GM_ENABLE_GBM=0"]}
BUILD = ROOT / "build" / "c28x"


def find_cl2000(arg: str | None, cgt_root: str | None) -> Path | None:
    exe = "cl2000.exe" if platform.system() == "Windows" else "cl2000"
    for cand in ([Path(arg)] if arg else []) + ([Path(cgt_root) / "bin" / exe] if cgt_root else []) + \
            ([Path(os.environ["C2000_CGT_ROOT"]) / "bin" / exe] if os.environ.get("C2000_CGT_ROOT") else []):
        if cand.is_file():
            return cand
    p = shutil.which("cl2000")
    return Path(p) if p else None


def supported_flags(cl: Path) -> tuple[list[str], list[str], str]:
    """Flags of WANTED that appear in `cl2000 --help` (option name before '='), the dropped ones, the version line."""
    r = subprocess.run([str(cl), "--help"], capture_output=True, text=True)
    help_txt = r.stdout + r.stderr
    v = subprocess.run([str(cl), "-version"], capture_output=True, text=True)
    version = (v.stdout + v.stderr).strip().splitlines()[0] if (v.stdout + v.stderr).strip() else "unknown"
    keep, drop = [], []
    for f in WANTED:
        name = f.split("=")[0]
        if name.startswith("--"):
            ok = name in help_txt
        else:  # short options such as -v28 -ml -mt -O2 appear as e.g. "-v28" / "--silicon_version"
            ok = (name in help_txt) or name in ("-O2",)
        (keep if ok else drop).append(f)
    return keep, drop, version


def compile_cmds(cl: Path | str, flags: list[str], defs: list[str], outdir: Path) -> list[list[str]]:
    inc = [f"--include_path={p}" for p in INCLUDES]
    rts_inc = Path(cl).parent.parent / "include"  # the compiler's own headers (CCS adds this automatically)
    if rts_inc.is_dir():
        inc.append(f"--include_path={rts_inc}")
    return [[str(cl)] + flags + defs + inc + [f"--obj_directory={outdir}", "--compile_only", str(s)] for s in SOURCES]


_SEC_ROW = re.compile(r"^\s*\d+\s+(\S+)\s+0x[0-9a-fA-F]+\s+0x[0-9a-fA-F]+\s+(0x[0-9a-fA-F]+|\d+)\s+\d+\s+([YN])\s*$")


def section_sizes(ofd: Path, obj: Path) -> dict[str, int]:
    """{allocated section name: size in 16-bit words} from the `ofd2000` text listing ("Section Information").

    Only sections with alloc = Y count (debug/symbol sections are not loaded). Sizes in the listing are in
    C28x addressable units (16-bit words); e.g. 50 float32 values = 0x64 words. Empty dict if nothing parses.
    """
    r = subprocess.run([str(ofd), str(obj)], capture_output=True, text=True)
    out: dict[str, int] = {}
    for line in (r.stdout or "").splitlines():
        m = _SEC_ROW.match(line)
        if m and m.group(3) == "Y":
            name, size = m.group(1), int(m.group(2), 0)
            out[name] = out.get(name, 0) + size
    return out


def classify_section(name: str) -> str:
    if name.startswith(".text") or name.startswith(".TI.ramfunc"):
        return "code"
    if name.startswith((".const", ".econst", ".rodata", ".switch", ".cinit")):
        return "const"
    if name.startswith((".bss", ".ebss", ".data", ".sysmem", ".far", ".TI.noinit", ".TI.persistent")):
        return "data"
    return "other"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cl2000", help="path to cl2000(.exe)")
    ap.add_argument("--cgt-root", help="C2000-CGT install directory (contains bin/cl2000)")
    ap.add_argument("--dry-run", action="store_true", help="print the compile commands and exit")
    ap.add_argument("--out", default=str(ROOT / "reports" / "m7_c28x_size.json"))
    args = ap.parse_args(argv)
    cl = find_cl2000(args.cl2000, args.cgt_root)
    if args.dry_run:
        for name, defs in VARIANTS.items():
            for c in compile_cmds(cl or "cl2000", WANTED, defs, BUILD / name):
                print(" ".join(c))
        return 0
    if cl is None:
        print("cl2000 not found: install TI C2000-CGT and pass --cl2000 or --cgt-root (docs/C2000_BUILD.md)")
        return 2
    flags, dropped, version = supported_flags(cl)
    ofd = cl.with_name("ofd2000" + (".exe" if cl.suffix == ".exe" else ""))
    report: dict = {"compiler": str(cl), "version": version, "flags": flags, "flags_dropped_not_in_help": dropped,
                    "host": platform.platform(), "units": "16-bit words (C28x addressable units); bytes = 2 x words",
                    "variants": {}}
    ok_all = True
    for name, defs in VARIANTS.items():
        outdir = BUILD / name
        outdir.mkdir(parents=True, exist_ok=True)
        files = {}
        for cmd, src in zip(compile_cmds(cl, flags, defs, outdir), SOURCES):
            r = subprocess.run(cmd, capture_output=True, text=True)
            obj = outdir / (src.stem + ".obj")
            entry: dict = {"ok": r.returncode == 0 and obj.exists(),
                           "diagnostics": [ln for ln in (r.stdout + r.stderr).splitlines() if re.search(r"warning|error", ln, re.I)][:20]}
            if entry["ok"] and ofd.exists():
                secs = section_sizes(ofd, obj)
                entry["sections_words"] = secs
                for k in ("code", "const", "data", "other"):
                    entry[f"{k}_words"] = sum(v for s, v in secs.items() if classify_section(s) == k)
            ok_all &= entry["ok"]
            files[src.relative_to(ROOT).as_posix()] = entry
            print(f"[{name}] {src.name}: {'ok' if entry['ok'] else 'FAILED'}"
                  + (f"  code {entry.get('code_words', '?')} / const {entry.get('const_words', '?')} / data "
                     f"{entry.get('data_words', '?')} words" if entry["ok"] else ""))
            for d in entry["diagnostics"][:5]:
                print("    " + d)
        tot = {k: sum(f.get(f"{k}_words", 0) for f in files.values()) for k in ("code", "const", "data", "other")}
        report["variants"][name] = {"files": files, "total_words": tot, "total_bytes": {k: 2 * v for k, v in tot.items()}}
        print(f"[{name}] total words: {tot}")
    report["all_compiled"] = ok_all
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"-> {args.out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
