"""Shared fixtures. Data-dependent tests use data/ (full) if present, else data/fast/, else skip.

Set GRIDMON_TEST_FAST=1 to force data/fast/.
"""

from __future__ import annotations

import os

import pytest

from gridmon.io import data_root, load_dataset_config

ALL_SPLITS = ("train", "val", "test_iid", "test_shift", "preview_iid", "preview_shift")


@pytest.fixture(scope="session")
def cfg():
    return load_dataset_config()


@pytest.fixture(scope="session")
def fast_flag() -> bool:
    force_fast = os.environ.get("GRIDMON_TEST_FAST", "") == "1"
    if not force_fast and all((data_root(False) / s / "manifest.csv").exists() for s in ALL_SPLITS):
        return False
    if all((data_root(True) / s / "manifest.csv").exists() for s in ALL_SPLITS):
        return True
    pytest.skip("no generated dataset (run scripts/make_dataset.py [--fast])")
