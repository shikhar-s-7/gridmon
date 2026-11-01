"""M6: build the portable C core for the host (shared library for the ctypes SIL, gm_cli, sanitizer CLI,
compile checks of every configuration). Output in build/ (git-ignored).

Usage: python scripts/build_c.py [--force] [--no-sanitize]
Finds $CC / cc / gcc / clang, else installs and uses `python -m ziglang cc` (Windows without a compiler).
"""

from __future__ import annotations

import argparse
import json
import sys

from gridmon.c_build import BuildError, build_all


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild even if sources are unchanged")
    ap.add_argument("--no-sanitize", action="store_true", help="skip the -fsanitize build")
    ap.add_argument("--fast", action="store_true", help="accepted for symmetry with the other scripts (no effect)")
    args = ap.parse_args(argv)
    try:
        info = build_all(force=args.force, sanitize=not args.no_sanitize)
    except BuildError as e:
        print(str(e))
        return 1
    checks = json.loads(info.pop("checks"))
    for k, v in info.items():
        print(f"{k}: {v}")
    print("compile checks: " + ", ".join(f"{k} {v}" for k, v in checks.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
