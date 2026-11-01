"""M4 freeze (PROMPT_02 A1.4): configs/frozen_methods.yaml lists everything that defines the methods
(rule thresholds, GBM configurations, seeds, model files with SHA-256, SHA-256 of every
method-defining source/config file). scripts/evaluate_test.py refuses to load a test split unless
`verify()` returns no problems: file hashes match, model hashes match, no unlisted method file
exists, and (in a git checkout) the working tree equals the M4-freeze commit for those files.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .io import ROOT

FROZEN_PATH = "configs/frozen_methods.yaml"
# Method-defining files: everything between the int16 waveform and a label (DSP, features, rules,
# model input mapping, exported model walker, configurations, generated C tables).
METHOD_GLOBS = ("src/gridmon/dsp/*.py", "src/gridmon/features.py", "src/gridmon/baseline_rules.py",
                "src/gridmon/units.py", "src/gridmon/ml_gbm.py", "src/gridmon/export_gbm.py",
                "configs/baseline.yaml", "configs/ml.yaml", "c/generated/gm_model_gbm.c",
                "c/generated/gm_model_gbm.h")


class FreezeError(RuntimeError):
    """Raised when the working tree differs from the frozen methods."""


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def method_files(root: Path = ROOT) -> list[str]:
    """Method-defining files present under root (posix paths relative to root, sorted)."""
    out: set[str] = set()
    for g in METHOD_GLOBS:
        out |= {p.relative_to(root).as_posix() for p in root.glob(g) if p.is_file()}
    return sorted(out)


def file_hashes(root: Path = ROOT) -> dict[str, str]:
    return {f: sha256(root / f) for f in method_files(root)}


def load_frozen(root: Path = ROOT) -> dict[str, Any]:
    p = root / FROZEN_PATH
    if not p.exists():
        raise FreezeError(f"{FROZEN_PATH} not found: methods are not frozen (run scripts/freeze_methods.py)")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def freeze_commit(root: Path = ROOT) -> str | None:
    """Last commit that touched configs/frozen_methods.yaml (the M4-freeze commit), or None."""
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%H", "--", FROZEN_PATH], cwd=root, capture_output=True,
                             text=True, check=True).stdout.strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def verify(root: Path = ROOT, frozen: dict[str, Any] | None = None, check_models: bool = True,
           check_git: bool = True) -> list[str]:
    """List of problems (empty = the working tree matches the freeze)."""
    root = Path(root)
    try:
        frozen = frozen or load_frozen(root)
    except FreezeError as e:
        return [str(e)]
    problems: list[str] = []
    listed = frozen.get("method_files", {})
    present = file_hashes(root)
    for f, h in listed.items():
        if f not in present:
            problems.append(f"missing method file {f}")
        elif present[f] != h:
            problems.append(f"method file changed since the freeze: {f}")
    for f in sorted(set(present) - set(listed)):
        problems.append(f"unlisted method file (added after the freeze): {f}")
    if check_models:
        for kind in ("gbm_embedded", "gbm_unconstrained"):
            for m in frozen.get(kind, {}).get("models", []):
                p = root / m["path"]
                if not p.exists():
                    problems.append(f"missing model file {m['path']}")
                elif sha256(p) != m["sha256"]:
                    problems.append(f"model file changed since the freeze: {m['path']}")
    if check_git:
        fc = freeze_commit(root)
        if fc is None:
            problems.append("no M4-freeze commit found (configs/frozen_methods.yaml is not committed)")
        else:
            paths = list(listed) + [FROZEN_PATH] + [m["path"] for k in ("gbm_embedded", "gbm_unconstrained")
                                                    for m in frozen.get(k, {}).get("models", [])]
            r = subprocess.run(["git", "diff", "--name-only", fc, "--", *paths], cwd=root, capture_output=True, text=True)
            changed = [x for x in r.stdout.split() if x]
            if r.returncode != 0:
                problems.append(f"git diff against the freeze commit failed: {r.stderr.strip()[:200]}")
            for x in changed:
                problems.append(f"working tree differs from the M4-freeze commit {fc[:7]}: {x}")
    return problems


def require_frozen(root: Path = ROOT, **kw) -> dict[str, Any]:
    """Raise FreezeError unless verify() passes; return the frozen configuration."""
    problems = verify(root, **kw)
    if problems:
        raise FreezeError("refusing to evaluate the sealed test sets:\n  " + "\n  ".join(problems))
    return load_frozen(root)
