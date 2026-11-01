"""Paths, configuration loading, hashing and split I/O (with the sealed-test-set guard, rule A1.4)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT: Path = Path(__file__).resolve().parents[2]
SEALED_SPLITS: frozenset[str] = frozenset({"test_iid", "test_shift"})


class SealedSplitError(RuntimeError):
    """Raised when a sealed test split is loaded without allow_test=True."""


def data_root(fast: bool = False) -> Path:
    """data/ for full runs, data/fast/ for --fast runs (both git-ignored)."""
    return ROOT / "data" / ("fast" if fast else "")


def reports_root(fast: bool = False) -> Path:
    """reports/ for full runs, reports/fast/ for --fast runs (fast outputs are git-ignored)."""
    return ROOT / "reports" / ("fast" if fast else "")


def models_root(fast: bool = False) -> Path:
    """models/ (committed exported models) or models/fast/ (git-ignored)."""
    return ROOT / "models" / ("fast" if fast else "")


def c_generated_root(fast: bool = False) -> Path:
    """c/generated/ (committed C tables) or c/generated/fast/ (git-ignored)."""
    return ROOT / "c" / "generated" / ("fast" if fast else "")


def load_ml_config() -> dict[str, Any]:
    return load_yaml(ROOT / "configs" / "ml.yaml")


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_dataset_config() -> dict[str, Any]:
    return load_yaml(ROOT / "configs" / "dataset.yaml")


def load_baseline_config() -> dict[str, Any]:
    return load_yaml(ROOT / "configs" / "baseline.yaml")


def config_hash(*objs: Any) -> str:
    """sha256 (first 16 hex) of the canonical JSON of the given objects."""
    blob = json.dumps(objs, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def git_commit() -> str:
    """Current commit hash, with '+dirty' if the work tree has changes; 'uncommitted' if none."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs", "scripts"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def _check_sealed(split: str, allow_test: bool) -> None:
    if split in SEALED_SPLITS and not allow_test:
        raise SealedSplitError(
            f"split '{split}' is sealed; only the final evaluation (M4) and the C-equivalence "
            f"check (M6) may pass allow_test=True")


def load_split(split: str, fast: bool = False, allow_test: bool = False,
               mmap: bool = True) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Load (X int16 [N, 3, 5000], manifest, meta) for a split. Raises for sealed splits."""
    _check_sealed(split, allow_test)
    d = data_root(fast) / split
    X = np.load(d / "X.npy", mmap_mode="r" if mmap else None)
    man = pd.read_csv(d / "manifest.csv")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return X, man, meta


def load_waveforms(split: str, fast: bool = False, allow_test: bool = False,
                   mmap: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Load only the waveform array and meta (what methods may see: no manifest)."""
    _check_sealed(split, allow_test)
    d = data_root(fast) / split
    X = np.load(d / "X.npy", mmap_mode="r" if mmap else None)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return X, meta


def load_labels(split: str, fast: bool = False, allow_test: bool = False) -> np.ndarray:
    """Labels for evaluation only (never passed to feature extraction or rules)."""
    _check_sealed(split, allow_test)
    return pd.read_csv(data_root(fast) / split / "manifest.csv", usecols=["label"])["label"].to_numpy()


def load_manifest(split: str, fast: bool = False, allow_test_manifest: bool = False) -> pd.DataFrame:
    """Manifest only. Sealed-split manifests need allow_test_manifest=True (tests: counts, seeds, margins)."""
    if split in SEALED_SPLITS and not allow_test_manifest:
        raise SealedSplitError(f"manifest of sealed split '{split}' requires allow_test_manifest=True")
    return pd.read_csv(data_root(fast) / split / "manifest.csv")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
