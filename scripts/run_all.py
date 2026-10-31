"""Run every implemented stage in order: M1 dataset -> M2 baseline -> M2 golden vectors.

Usage: python scripts/run_all.py [--fast]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ["make_dataset", "run_baseline", "make_golden"]


def _load(name: str):
    path = HERE / f"{name}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args(argv)
    times = {}
    t_all = time.perf_counter()
    for name in STAGES:
        mod = _load(name)
        if mod is None:
            print(f"[run_all] {name}: not implemented yet, skipped")
            continue
        print(f"[run_all] === {name} ===")
        t0 = time.perf_counter()
        rc = mod.main(["--fast"] if args.fast else [])
        times[name] = time.perf_counter() - t0
        if rc:
            print(f"[run_all] {name} failed with code {rc}")
            return rc
    total = time.perf_counter() - t_all
    print("[run_all] runtimes: " + ", ".join(f"{k} {v:.1f} s" for k, v in times.items()) + f"; total {total:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
