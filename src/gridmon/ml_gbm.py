"""M3 gradient boosting on the 37 M2 features: training, grid evaluation, selection, prediction.

GBM-embedded: sklearn GradientBoostingClassifier (exportable, docs/MODEL_FORMAT.md); grid over
max_leaf_nodes x learning_rate x subsample with n_estimators evaluated as stages of one fit (the
first n stages of a fit ARE the model trained with n_estimators = n: the per-stage random draws do
not depend on n_estimators; checked in tests). Selection: highest mean val macro-F1 over seeds among
configurations whose exported size is <= budget for every seed; mean F1 within tie_tol of the best
-> the smaller model (max bytes over seeds), then the higher F1.
GBM-unconstrained (reference): HistGradientBoostingClassifier with early stopping on an internal
15 % split of TRAIN. Features enter as float64 and are cast to float32 by sklearn (and by the C walker).
Labels are encoded as gridmon.CLASSES indices so class k of every model is CLASSES[k].
"""

from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

from . import CLASSES
from .export_gbm import gbc_size, hgb_size
from .features import FEATURE_NAMES
from .io import data_root, load_labels
from .metrics import macro_f1

K = len(CLASSES)


def encode(y: np.ndarray) -> np.ndarray:
    """String labels -> CLASSES indices."""
    return np.array([CLASSES.index(c) for c in y], dtype=np.int64)


def decode(idx: np.ndarray) -> np.ndarray:
    return np.array([CLASSES[int(i)] for i in idx])


NAN_FILL = 0.0  # D-040: model input for a feature the reference marks "not computed" (NaN)


def model_input(F: pd.DataFrame) -> np.ndarray:
    """ML input [N, 37] (FEATURE_NAMES order, float64): non-finite values -> NAN_FILL (D-040).

    Only rocof_end can be NaN in a full record (jump mask covering the whole END window). The same
    mapping is used by every model and by the C walker (x = isfinite(x) ? x : 0.0f)."""
    X = F[FEATURE_NAMES].to_numpy(dtype=np.float64).copy()
    X[~np.isfinite(X)] = NAN_FILL
    return X


def nonfinite_counts(F: pd.DataFrame) -> dict[str, int]:
    return {c: int((~np.isfinite(F[c].to_numpy(dtype=float))).sum()) for c in FEATURE_NAMES
            if (~np.isfinite(F[c].to_numpy(dtype=float))).any()}


def load_features(split: str, fast: bool = False, allow_test: bool = False) -> np.ndarray:
    """ML input matrix [N, 37] from data/<split>/features.csv (written by run_baseline / evaluate_test)."""
    from .io import _check_sealed

    _check_sealed(split, allow_test)
    return model_input(pd.read_csv(data_root(fast) / split / "features.csv"))


def load_xy(split: str, fast: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(features, encoded labels) of a non-sealed split."""
    return load_features(split, fast), encode(load_labels(split, fast=fast))


def make_embedded(params: dict[str, Any], seed: int, fixed: dict[str, Any]) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(**fixed, **params, random_state=seed)


def make_unconstrained(params: dict[str, Any], seed: int, fixed: dict[str, Any]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**fixed, **params, random_state=seed)


def _f1(y: np.ndarray, p: np.ndarray) -> float:
    return macro_f1(np.asarray(y), np.asarray(p), K)


def _fit_embedded_staged(base: dict[str, Any], seed: int, n_list: list[int], fixed: dict[str, Any],
                         Xtr, ytr, Xva, yva) -> list[dict[str, Any]]:
    """One fit with max(n_list) stages; metrics and size for each n in n_list."""
    t0 = time.perf_counter()
    clf = make_embedded({**base, "n_estimators": max(n_list)}, seed, fixed).fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    want = set(n_list)
    out = []
    staged_va = clf.staged_predict(Xva)
    staged_tr = clf.staged_predict(Xtr)
    for n, (pva, ptr) in enumerate(zip(staged_va, staged_tr), start=1):
        if n in want:
            nodes, trees, nbytes = gbc_size(clf, n)
            out.append({**base, "n_estimators": n, "seed": seed, "val_macro_f1": _f1(yva, pva),
                        "train_macro_f1": _f1(ytr, ptr), "n_nodes": nodes, "n_trees": trees, "bytes": nbytes,
                        "fit_s_100": fit_s})
    return out


def grid_embedded(mcfg: dict[str, Any], Xtr, ytr, Xva, yva) -> pd.DataFrame:
    """Per-fit table (config x seed) of the GBM-embedded grid."""
    g = mcfg["embedded"]["grid"]
    fixed = mcfg["embedded"]["fixed"]
    bases = [dict(zip(("max_leaf_nodes", "learning_rate", "subsample"), c))
             for c in itertools.product(g["max_leaf_nodes"], g["learning_rate"], g["subsample"])]
    jobs = [(b, s) for b in bases for s in mcfg["seeds"]]
    res = Parallel(n_jobs=mcfg["n_jobs"])(
        delayed(_fit_embedded_staged)(b, s, list(g["n_estimators"]), fixed, Xtr, ytr, Xva, yva) for b, s in jobs)
    return pd.DataFrame([r for rr in res for r in rr])


def summarise(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean/sd over seeds per configuration (+ max bytes over seeds)."""
    agg = df.groupby(keys).agg(val_f1_mean=("val_macro_f1", "mean"), val_f1_sd=("val_macro_f1", "std"),
                               train_f1_mean=("train_macro_f1", "mean"), bytes_max=("bytes", "max"),
                               bytes_mean=("bytes", "mean"), n_nodes_max=("n_nodes", "max"),
                               n_seeds=("seed", "count")).reset_index()
    agg["val_f1_sd"] = agg["val_f1_sd"].fillna(0.0)
    return agg


def select_embedded(summary: pd.DataFrame, budget: int, tie_tol: float) -> dict[str, Any]:
    """Selection rule (PROMPT_02 M3) on the per-config summary."""
    feas = summary[summary["bytes_max"] <= budget]
    if feas.empty:
        raise ValueError("no configuration fits the size budget")
    best = float(feas["val_f1_mean"].max())
    cand = feas[feas["val_f1_mean"] >= best - tie_tol].sort_values(
        ["bytes_max", "val_f1_mean", "n_estimators", "max_leaf_nodes"], ascending=[True, False, True, True])
    row = cand.iloc[0]
    return {"config": {"n_estimators": int(row["n_estimators"]), "max_leaf_nodes": int(row["max_leaf_nodes"]),
                       "learning_rate": float(row["learning_rate"]), "subsample": float(row["subsample"])},
            "val_f1_mean": float(row["val_f1_mean"]), "val_f1_sd": float(row["val_f1_sd"]),
            "bytes_max": int(row["bytes_max"]), "best_feasible_f1": best, "n_feasible": int(len(feas)),
            "n_tied": int(len(cand)),
            "tied_configs": cand[["n_estimators", "max_leaf_nodes", "learning_rate", "subsample", "val_f1_mean",
                                  "bytes_max"]].to_dict(orient="records")}


def _fit_unconstrained(params: dict[str, Any], seed: int, fixed: dict[str, Any], Xtr, ytr, Xva, yva) -> dict[str, Any]:
    t0 = time.perf_counter()
    clf = make_unconstrained(params, seed, fixed).fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    nodes, trees, nbytes = hgb_size(clf)
    return {**params, "seed": seed, "val_macro_f1": _f1(yva, clf.predict(Xva)), "train_macro_f1": _f1(ytr, clf.predict(Xtr)),
            "n_iter": int(clf.n_iter_), "n_nodes": nodes, "n_trees": trees, "bytes": nbytes, "fit_s": fit_s}


def grid_unconstrained(mcfg: dict[str, Any], Xtr, ytr, Xva, yva) -> pd.DataFrame:
    g = mcfg["unconstrained"]["grid"]
    fixed = mcfg["unconstrained"]["fixed"]
    params = [dict(zip(("max_leaf_nodes", "learning_rate"), c)) for c in itertools.product(g["max_leaf_nodes"], g["learning_rate"])]
    res = Parallel(n_jobs=mcfg["n_jobs"])(
        delayed(_fit_unconstrained)(p, s, fixed, Xtr, ytr, Xva, yva) for p in params for s in mcfg["seeds"])
    return pd.DataFrame(res)


def select_unconstrained(summary: pd.DataFrame) -> dict[str, Any]:
    """Highest mean val macro-F1 (no size limit); ties -> smaller model."""
    row = summary.sort_values(["val_f1_mean", "bytes_max"], ascending=[False, True]).iloc[0]
    return {"config": {"max_leaf_nodes": int(row["max_leaf_nodes"]), "learning_rate": float(row["learning_rate"])},
            "val_f1_mean": float(row["val_f1_mean"]), "val_f1_sd": float(row["val_f1_sd"]),
            "bytes_max": int(row["bytes_max"])}


def fit_seeds(kind: str, config: dict[str, Any], mcfg: dict[str, Any], Xtr, ytr) -> list:
    """Fit the selected configuration for every seed (kind 'embedded' or 'unconstrained')."""
    return Parallel(n_jobs=mcfg["n_jobs"])(delayed(_fit_one)(kind, config, s, mcfg[kind]["fixed"], Xtr, ytr)
                                           for s in mcfg["seeds"])


def _fit_one(kind: str, config: dict[str, Any], seed: int, fixed: dict[str, Any], Xtr, ytr):
    make = make_embedded if kind == "embedded" else make_unconstrained
    return make(config, seed, fixed).fit(Xtr, ytr)
