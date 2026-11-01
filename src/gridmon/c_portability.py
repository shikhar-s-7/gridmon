"""Static portability checks of the portable C core (c/src, c/include, c/tables) and cross compiles.

Static rules (prompt A3, C28x: CHAR_BIT = 16, 16-bit int):
  no `double`; no int8_t/uint8_t/char data (only `const char *gm_version(void)`); no plain int/long/short/
  unsigned/signed types (explicit <stdint.h> widths only); no malloc/free/stdio/stdlib/string calls; no
  static or global mutable state; every floating literal has an f suffix; no unions or pointer casts of
  floats (type punning); `sizeof` only in gm_ctx_words; no recursion (call graph is acyclic).
Cross compiles (compile only, same warning flags): -m32 (host compiler; if its 32-bit libc headers are
missing, `python -m ziglang cc -target x86-linux-musl`), and a 16-bit-int target (zig/clang msp430,
freestanding, with a stub <math.h> holding the C99 prototypes). Unavailable toolchains are reported as
skipped, never as passed.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from .c_build import BUILD_DIR, C_DIR, CORE_SOURCES, INCLUDES, WARN_FLAGS, find_compiler

PORTABLE_DIRS = ("src", "include", "tables")
FORBIDDEN_CALLS = ("malloc", "calloc", "realloc", "free", "printf", "fprintf", "sprintf", "snprintf", "puts",
                   "putchar", "fopen", "fclose", "fread", "fwrite", "exit", "abort", "memcpy", "memset", "memmove",
                   "strlen", "strcpy", "rand", "srand", "time", "clock")
FORBIDDEN_HEADERS = ("stdio.h", "stdlib.h", "string.h", "time.h", "assert.h", "float.h")


def portable_files() -> list[Path]:
    return sorted(p for d in PORTABLE_DIRS for p in (C_DIR / d).glob("*.[ch]"))


def strip_c(text: str) -> str:
    """Remove comments and the contents of string/char literals (keeps line structure)."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        elif text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            i = j
        elif text[i] in "\"'":
            q = text[i]
            j = i + 1
            while j < n and text[j] != q:
                j += 2 if text[j] == "\\" else 1
            out.append(q + q)
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


_FLOAT_LIT = re.compile(r"(?<![\w.])(0[xX][0-9a-fA-F]*\.?[0-9a-fA-F]*[pP][+-]?\d+|\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)([fFlL]?)(?![\w.])")
_PLAIN_INT = re.compile(r"\b(int|long|short|unsigned|signed)\b")
_CHAR = re.compile(r"\b(char|int8_t|uint8_t)\b")
_FUNC_DEF = re.compile(r"(?:^|\n)[A-Za-z_][\w \t\*]*?\b([A-Za-z_]\w*)\s*\(([^;{}()]*)\)\s*\{")
_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def _top_level_statements(code: str) -> list[str]:
    """Top-level declarations ending with ';' (brace bodies collapsed to '{}'; function bodies dropped)."""
    stmts, depth, cur = [], 0, []
    for ch in code:
        if ch == "{":
            if depth == 0:
                cur.append("{}")
            depth += 1
            continue
        elif ch == "}":
            depth -= 1
            if depth == 0 and "".join(cur[:-1]).rstrip().endswith(")"):  # function definition
                cur = []
            continue
        if depth == 0:
            cur.append(ch)
            if ch == ";":
                stmts.append("".join(cur).strip())
                cur = []
    return stmts


def static_checks(files: list[Path] | None = None) -> dict[str, list[str]]:
    """Violations per rule (empty lists = pass)."""
    files = files or portable_files()
    v: dict[str, list[str]] = {k: [] for k in ("double", "char_data", "plain_int", "forbidden_call", "forbidden_header",
                                               "static_mutable", "global_mutable", "float_literal_without_f",
                                               "type_punning", "sizeof", "recursion")}
    defs: dict[str, str] = {}
    for f in files:
        rel = f.resolve().relative_to(C_DIR.parent).as_posix() if f.resolve().is_relative_to(C_DIR.parent) else f.name
        raw = f.read_text(encoding="utf-8")
        code = strip_c(raw)
        lines = code.splitlines()
        for ln, line in enumerate(lines, 1):
            where = f"{rel}:{ln}"
            if line.lstrip().startswith("#") and "include" in line:
                hdr = raw.splitlines()[ln - 1]
                if any(h in hdr for h in FORBIDDEN_HEADERS):
                    v["forbidden_header"].append(where)
                continue
            if re.search(r"\bdouble\b", line):
                v["double"].append(where)
            if _CHAR.search(line) and not re.search(r"const\s+char\s*\*\s*gm_version\s*\(\s*void\s*\)", line):
                v["char_data"].append(where)
            if _PLAIN_INT.search(line):
                v["plain_int"].append(where)
            for name in FORBIDDEN_CALLS:
                if re.search(rf"\b{name}\s*\(", line):
                    v["forbidden_call"].append(f"{where} {name}")
            for m in _FLOAT_LIT.finditer(line):
                if m.group(2) not in ("f", "F"):
                    v["float_literal_without_f"].append(f"{where} {m.group(0)}")
            if re.search(r"\bunion\b", line) or re.search(r"\(\s*(?:const\s+)?u?int(?:8|16|32|64)_t\s*\*\s*\)\s*&", line):
                v["type_punning"].append(where)
            if "sizeof" in line and not (rel.endswith("gm_api.c") and "sizeof(gm_ctx_t)" in line):
                v["sizeof"].append(where)
            m = re.search(r"\bstatic\b(?!\s+(const|inline)\b)", line)
            if m and "(" not in line[m.end():]:
                v["static_mutable"].append(where)
        for st in _top_level_statements(code):
            s = " ".join(st.split())
            if s.startswith("#") or s.startswith(("typedef", "extern")) or "(" in s:
                continue
            if re.match(r"^(static\s+)?const\b", s) or re.search(r"\bconst\s+[\w\s]*\w\s*(\[|=)", s):
                continue
            if s.startswith(("enum", "struct")) and s.endswith("};"):
                continue
            if s in (";", "};"):
                continue
            v["global_mutable"].append(f"{rel}: {s[:80]}")
        if f.suffix == ".c":
            for m in _FUNC_DEF.finditer(code):
                name = m.group(1)
                if name in ("if", "for", "while", "switch", "return", "sizeof"):
                    continue
                start = m.end() - 1
                depth, j = 0, start
                while j < len(code):
                    depth += code[j] == "{"
                    depth -= code[j] == "}"
                    if depth == 0:
                        break
                    j += 1
                defs[name] = code[start:j + 1]
    graph = {fn: {c for c in _CALL.findall(body) if c in defs} for fn, body in defs.items()}
    v["recursion"] = _cycles(graph)
    return v


def _cycles(graph: dict[str, set[str]]) -> list[str]:
    color: dict[str, int] = {}
    found: list[str] = []

    def visit(u: str, path: list[str]) -> None:
        color[u] = 1
        for w in sorted(graph[u]):
            if color.get(w) == 1:
                found.append(" -> ".join(path[path.index(w):] + [w]) if w in path else f"{u} -> {w}")
            elif w not in color:
                visit(w, path + [w])
        color[u] = 2

    for u in sorted(graph):
        if u not in color:
            visit(u, [u])
    return found


STUB_MATH = """/* stub <math.h> for freestanding cross-compile checks: C99 prototypes only */
#ifndef GM_STUB_MATH_H
#define GM_STUB_MATH_H
float sinf(float); float cosf(float); float atan2f(float, float); float sqrtf(float);
float floorf(float); float expf(float); float fabsf(float);
#define NAN (__builtin_nanf(""))
#endif
"""


def _compile_all(cmd_prefix: list[str], extra: list[str]) -> tuple[bool, str]:
    tmp = BUILD_DIR / "xcheck"
    tmp.mkdir(parents=True, exist_ok=True)
    for s in CORE_SOURCES:
        for defs in ([], ["-DGM_ENABLE_GBM=0"]):
            cmd = cmd_prefix + WARN_FLAGS + ["-O2"] + extra + defs + INCLUDES + ["-c", str(s), "-o", str(tmp / (s.stem + ".o"))]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                return False, f"{s.name}: {(r.stderr or r.stdout).strip()[:600]}"
    return True, "ok"


def _zig() -> list[str] | None:
    zig = [sys.executable, "-m", "ziglang", "cc"]
    try:
        return zig if subprocess.run(zig + ["--version"], capture_output=True).returncode == 0 else None
    except OSError:
        return None


def cross_compile_checks() -> dict[str, dict[str, str]]:
    """{'m32': {...}, 'int16': {...}} with status 'pass' / 'fail' / 'skipped' and details."""
    res: dict[str, dict[str, str]] = {}
    stub = BUILD_DIR / "stub_include"
    stub.mkdir(parents=True, exist_ok=True)
    (stub / "math.h").write_text(STUB_MATH)
    cc = find_compiler(install_zig=False)
    ok, msg = _compile_all(cc, ["-m32"])
    if ok:
        res["m32"] = {"status": "pass", "toolchain": " ".join(cc) + " -m32"}
    else:
        zig = _zig()
        if zig:
            ok2, msg2 = _compile_all(zig, ["-target", "x86-linux-musl"])
            res["m32"] = {"status": "pass" if ok2 else "fail", "toolchain": "zig cc -target x86-linux-musl (ILP32)",
                          "note": "host -m32 unavailable: " + msg.splitlines()[-1][:160], "detail": msg2[:600]}
        elif "No such file" in msg or "not found" in msg:
            res["m32"] = {"status": "skipped", "note": "host -m32 libc headers missing and no ziglang: " + msg[:200]}
        else:
            res["m32"] = {"status": "fail", "detail": msg[:600]}
    zig = _zig()
    tgt = (zig, ["-target", "msp430-freestanding-none"]) if zig else (["clang"], ["--target=msp430-elf"])
    if tgt[0] is None:
        res["int16"] = {"status": "skipped", "note": "no msp430-capable compiler"}
    else:
        ok3, msg3 = _compile_all(tgt[0], tgt[1] + ["-ffreestanding", "-I" + str(stub)])
        res["int16"] = {"status": "pass" if ok3 else "fail",
                        "toolchain": " ".join(tgt[0][-2:]) + " " + " ".join(tgt[1]) + " (16-bit int, freestanding, stub math.h)",
                        "detail": msg3[:600]}
    return res
