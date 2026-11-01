"""Build helpers for the M6 C core (used by scripts/build_c.py, the SIL wrapper and the tests).

Compiler: $CC, else cc / gcc / clang on PATH, else `python -m ziglang cc` (installed with
`python -m pip install ziglang` on demand, e.g. on Windows without a C compiler).
Targets (output in build/, never committed):
  sil   shared library (core + tables + model + c/host/gm_sil.c), -DGM_TRACES -DGM_REPORT_NAN=1 (ctypes)
  cli   build/gm_cli: binary int16 records in, JSON lines out (-DGM_REPORT_NAN=1)
  asan  build/gm_cli_asan: the CLI with -fsanitize=address,undefined (where supported)
  check compile-only variants of the core: GM_ENABLE_GBM=0, no traces, GM_REPORT_NAN=0
Warning flags of the portable core (prompt A3): -std=c99 -O2 -Wall -Wextra -Wpedantic -Wconversion
-Wdouble-promotion -Wfloat-conversion -Werror (+ -Wvla, -ffp-contract=off: no fused multiply-add, as on C28x).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .io import ROOT

C_DIR = ROOT / "c"
BUILD_DIR = ROOT / "build"
CORE_SOURCES = sorted((C_DIR / "src").glob("*.c")) + [C_DIR / "tables" / "gm_tables.c",
                                                      C_DIR / "generated" / "gm_model_gbm.c"]
PORT_DIR = ROOT / "firmware" / "c2000"
PORT_SOURCES = [PORT_DIR / "gm_port.c", PORT_DIR / "gm_port_fir.c", PORT_DIR / "gm_port_c2000.c"]
INCLUDES = ["-I" + str(C_DIR / d) for d in ("include", "tables", "generated", "src", "host")] + ["-I" + str(PORT_DIR)]
WARN_FLAGS = ["-std=c99", "-Wall", "-Wextra", "-Wpedantic", "-Wconversion", "-Wdouble-promotion",
              "-Wfloat-conversion", "-Wvla", "-Werror", "-ffp-contract=off"]
OPT_FLAGS = ["-O2"]


class BuildError(RuntimeError):
    pass


def find_compiler(install_zig: bool = True) -> list[str]:
    """Command prefix of a C compiler ([path] or [python, -m, ziglang, cc])."""
    env = os.environ.get("CC")
    if env:
        return env.split()
    for name in ("cc", "gcc", "clang"):
        p = shutil.which(name)
        if p:
            return [p]
    zig = [sys.executable, "-m", "ziglang", "cc"]
    if subprocess.run(zig + ["--version"], capture_output=True).returncode == 0:
        return zig
    if install_zig:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ziglang"], check=True)
        if subprocess.run(zig + ["--version"], capture_output=True).returncode == 0:
            return zig
    raise BuildError("no C compiler found (set CC, install gcc/clang, or `python -m pip install ziglang`)")


def compiler_version(cc: list[str]) -> str:
    r = subprocess.run(cc + ["--version"], capture_output=True, text=True)
    return (r.stdout or r.stderr).splitlines()[0].strip() if r.returncode == 0 else "unknown"


def shared_lib_name(stem: str) -> str:
    if sys.platform.startswith("win"):
        return stem + ".dll"
    if sys.platform == "darwin":
        return "lib" + stem + ".dylib"
    return "lib" + stem + ".so"


def exe_name(stem: str) -> str:
    return stem + (".exe" if sys.platform.startswith("win") else "")


def _stamp(cmd: list[str], sources: list[Path]) -> str:
    h = hashlib.sha256(json.dumps(cmd).encode())
    for s in sorted(set(sources) | set(_headers())):
        h.update(s.read_bytes())
    return h.hexdigest()


def _headers() -> list[Path]:
    return sorted([p for d in ("include", "tables", "generated", "src", "host") for p in (C_DIR / d).glob("*.h")]
                  + list(PORT_DIR.glob("*.h")))


def _run(cmd: list[str], out: Path, sources: list[Path], force: bool) -> Path:
    stamp_file = out.with_name(out.name + ".stamp")
    stamp = _stamp(cmd, sources)
    if not force and out.exists() and stamp_file.exists() and stamp_file.read_text() == stamp:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise BuildError(f"build of {out.name} failed:\n{' '.join(cmd)}\n{(r.stderr or r.stdout)[-4000:]}")
    stamp_file.write_text(stamp)
    return out


def build_sil(cc: list[str] | None = None, force: bool = False) -> Path:
    """Shared library for ctypes (traces + NaN reporting)."""
    cc = cc or find_compiler()
    out = BUILD_DIR / shared_lib_name("gridmon_sil")
    srcs = CORE_SOURCES + [C_DIR / "host" / "gm_sil.c"]
    pic = [] if sys.platform.startswith("win") else ["-fPIC"]
    cmd = cc + WARN_FLAGS + OPT_FLAGS + pic + ["-shared", "-DGM_TRACES", "-DGM_REPORT_NAN=1"] + INCLUDES + \
        [str(s) for s in srcs] + ["-o", str(out), "-lm"]
    return _run(cmd, out, srcs, force)


def build_cli(cc: list[str] | None = None, force: bool = False, sanitize: bool = False) -> Path:
    """gm_cli (or gm_cli_asan with address/undefined-behaviour sanitizers)."""
    cc = cc or find_compiler()
    out = BUILD_DIR / exe_name("gm_cli_asan" if sanitize else "gm_cli")
    srcs = CORE_SOURCES + [C_DIR / "host" / "gm_cli.c"]
    extra = (["-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]
             if sanitize else OPT_FLAGS)
    cmd = cc + WARN_FLAGS + extra + ["-DGM_REPORT_NAN=1"] + INCLUDES + [str(s) for s in srcs] + ["-o", str(out), "-lm"]
    return _run(cmd, out, srcs, force)


def build_port_sim(cc: list[str] | None = None, force: bool = False) -> Path:
    """Shared library: core + port layer + firmware/c2000/host/gm_port_sim.c (M7 host simulation)."""
    cc = cc or find_compiler()
    out = BUILD_DIR / shared_lib_name("gridmon_port")
    srcs = CORE_SOURCES + PORT_SOURCES + [PORT_DIR / "host" / "gm_port_sim.c"]
    pic = [] if sys.platform.startswith("win") else ["-fPIC"]
    cmd = cc + WARN_FLAGS + OPT_FLAGS + pic + ["-shared"] + INCLUDES + [str(s) for s in srcs] + ["-o", str(out), "-lm"]
    return _run(cmd, out, srcs, force)


CHECK_VARIANTS: dict[str, list[str]] = {
    "gbm_off": ["-DGM_ENABLE_GBM=0"],
    "no_traces_nan0": ["-DGM_REPORT_NAN=0"],
    "traces_nan1": ["-DGM_TRACES", "-DGM_REPORT_NAN=1"],
}


def build_checks(cc: list[str] | None = None, extra_flags: list[str] | None = None) -> dict[str, str]:
    """Compile-only builds of the core in every configuration; returns {variant: 'ok' | error text}."""
    cc = cc or find_compiler()
    res: dict[str, str] = {}
    tmp = BUILD_DIR / "check"
    tmp.mkdir(parents=True, exist_ok=True)
    for name, defs in CHECK_VARIANTS.items():
        msgs = []
        for s in CORE_SOURCES + PORT_SOURCES:
            cmd = cc + WARN_FLAGS + OPT_FLAGS + (extra_flags or []) + defs + INCLUDES + \
                ["-c", str(s), "-o", str(tmp / (s.stem + ".o"))]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                msgs.append(f"{s.name}: {(r.stderr or r.stdout).strip()[:500]}")
        res[name] = "ok" if not msgs else "\n".join(msgs)
    return res


def object_sizes(cc: list[str] | None = None, defs: list[str] | None = None) -> dict[str, dict[str, int]]:
    """Per-object text/data/bss bytes of the portable core (host compiler, `size` tool if present)."""
    cc = cc or find_compiler()
    size_tool = shutil.which("size")
    tmp = BUILD_DIR / "objsize"
    tmp.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, int]] = {}
    for s in CORE_SOURCES:
        obj = tmp / (s.stem + ".o")
        subprocess.run(cc + WARN_FLAGS + OPT_FLAGS + (defs or []) + INCLUDES + ["-c", str(s), "-o", str(obj)], check=True,
                       capture_output=True)
        if size_tool:
            r = subprocess.run([size_tool, str(obj)], capture_output=True, text=True, check=True)
            text, data, bss = (int(v) for v in r.stdout.splitlines()[1].split()[:3])
            out[s.name] = {"text": text, "data": data, "bss": bss}
    return out


def build_all(force: bool = False, sanitize: bool = True) -> dict[str, str]:
    cc = find_compiler()
    info = {"compiler": " ".join(cc), "compiler_version": compiler_version(cc), "flags": " ".join(WARN_FLAGS + OPT_FLAGS)}
    info["sil"] = str(build_sil(cc, force).relative_to(ROOT))
    info["cli"] = str(build_cli(cc, force).relative_to(ROOT))
    info["port_sim"] = str(build_port_sim(cc, force).relative_to(ROOT))
    if sanitize:
        try:
            info["asan"] = str(build_cli(cc, force, sanitize=True).relative_to(ROOT))
        except BuildError as e:
            info["asan"] = "unsupported: " + str(e).splitlines()[-1][:200]
    checks = build_checks(cc)
    info["checks"] = json.dumps(checks)
    bad = {k: v for k, v in checks.items() if v != "ok"}
    if bad:
        raise BuildError("compile checks failed: " + json.dumps(bad)[:2000])
    (BUILD_DIR / "build_info.json").write_text(json.dumps(info, indent=2))
    return info
