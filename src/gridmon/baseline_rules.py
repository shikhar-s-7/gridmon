"""Priority-ordered rule classifier (brief B7.3). Input: feature table only.

1. Dip:   urms_min_1 < 0.90 [STD]            -> rho > rho_th ? sag_unbal : sag_bal
2. Swell: urms_max_3 > 1.10 [STD]            -> swell
3. Osc.:  env_mag_amp_pct > osc_mag_pct and env_mag_prom >= osc_prom, or
          env_ang_amp_deg > osc_ang_deg and env_ang_prom >= osc_prom [ASSUMED] -> oscillation
4. Jump:  j_max_deg > jump_deg and max(|v1_min_rel - 1|, |v1_max_rel - 1|) < jump_max_dv1 [ASSUMED] -> phase_jump
5. Freq:  |df_end| > df_hz (jump-masked) [ASSUMED] -> freq_event
6. Otherwise normal.
The IEC hysteresis end conditions (all >= 0.92 / all <= 1.08) only affect event duration, not the
label. Returns label, reason code and onset estimate (first detection time of the firing detector).
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from . import CLASSES
from .metrics import macro_f1

REASONS: dict[int, str] = {0: "none", 1: "dip_balanced", 2: "dip_unbalanced", 3: "swell",
                           4: "osc_magnitude", 5: "osc_angle", 6: "phase_jump", 7: "freq_offset"}
REASON_LABEL: dict[int, str] = {0: "normal", 1: "sag_bal", 2: "sag_unbal", 3: "swell", 4: "oscillation",
                                5: "oscillation", 6: "phase_jump", 7: "freq_event"}
TUNABLE = ("rho_th", "osc_mag_pct", "osc_ang_deg", "osc_prom", "jump_deg", "df_hz")


def _arrays(F: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c: F[c].to_numpy(dtype=float) for c in F.columns}


def classify_reasons(F: pd.DataFrame | dict[str, np.ndarray], th: dict[str, Any]) -> np.ndarray:
    """Reason code per record (priority order)."""
    f = _arrays(F) if isinstance(F, pd.DataFrame) else F
    n = len(f["urms_min_1"])
    reason = np.zeros(n, dtype=np.int8)
    open_ = np.ones(n, dtype=bool)

    def take(cond, code):
        nonlocal open_
        sel = open_ & cond
        reason[sel] = code
        open_ = open_ & ~sel

    with np.errstate(invalid="ignore"):
        dip = f["urms_min_1"] < th["dip_pu"]
        take(dip & (f["rho"] > th["rho_th"]), 2)
        take(dip, 1)
        take(f["urms_max_3"] > th["swell_pu"], 3)
        take((f["env_mag_amp_pct"] > th["osc_mag_pct"]) & (f["env_mag_prom"] >= th["osc_prom"]), 4)
        take((f["env_ang_amp_deg"] > th["osc_ang_deg"]) & (f["env_ang_prom"] >= th["osc_prom"]), 5)
        dv1 = np.maximum(np.abs(f["v1_min_rel"] - 1.0), np.abs(f["v1_max_rel"] - 1.0))
        take((f["j_max_deg"] > th["jump_deg"]) & (dv1 < th["jump_max_dv1"]), 6)
        take(np.abs(f["df_end"]) > th["df_hz"], 7)
    return reason


def classify(F: pd.DataFrame, th: dict[str, Any]) -> pd.DataFrame:
    """Rule label, reason code/name and onset estimate (s; NaN for normal) for every record."""
    reason = classify_reasons(F, th)
    onset_col = {1: "aux_t_dip", 2: "aux_t_dip", 3: "aux_t_swell", 4: "aux_t_osc_mag", 5: "aux_t_osc_ang",
                 6: "aux_t_jump", 7: "aux_t_freq"}
    onset = np.full(len(reason), np.nan)
    for code, col in onset_col.items():
        sel = reason == code
        if col in F.columns:
            onset[sel] = F[col].to_numpy()[sel]
    return pd.DataFrame({"rule_label": [REASON_LABEL[int(r)] for r in reason], "rule_reason": reason,
                         "rule_reason_name": [REASONS[int(r)] for r in reason], "rule_onset": onset})


def tune_thresholds(F: pd.DataFrame, y: np.ndarray, base: dict[str, Any], grid: dict[str, list]) -> tuple[dict[str, Any], float, int]:
    """Exhaustive grid search of the tunable thresholds maximising macro-F1 on (F, y) (TRAIN only).

    Ties (equal F1) are broken towards the defaults: smallest sum over thresholds of |log(th/default)|
    (deterministic; avoids drifting to grid edges on plateaus). Returns (best thresholds, F1, n_combos).
    """
    f = _arrays(F)
    keys = [k for k in TUNABLE if k in grid]
    best_th, best, best_dist = dict(base), -1.0, np.inf
    count = 0
    y_idx = np.array([CLASSES.index(c) for c in y])
    lab_of_reason = np.array([CLASSES.index(REASON_LABEL[i]) for i in range(8)])
    for combo in itertools.product(*(grid[k] for k in keys)):
        th = dict(base)
        th.update(dict(zip(keys, combo)))
        pred = lab_of_reason[classify_reasons(f, th)]
        score = macro_f1(y_idx, pred, len(CLASSES))
        dist = sum(abs(np.log(th[k] / base[k])) for k in keys)
        count += 1
        if score > best + 1e-12 or (abs(score - best) <= 1e-12 and dist < best_dist):
            best, best_th, best_dist = score, th, dist
    return best_th, best, count
