"""Golden vectors for the C port (brief B7.5): reference traces, features and rule outputs.

Each golden/<id>.npz holds the int16 input and every intermediate the C core must reproduce,
computed by the float64 Python reference. Tolerances (float32 C vs this reference) are derived
from the observed float32-numpy error on the same records and written to golden/README.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .baseline_rules import REASONS, classify
from .features import AUX_NAMES, FEATURE_NAMES, compute_traces, features_from_traces
from .generator.events import quiet_normal
from .generator.nuisance import clean_pre_event
from .generator.synth import synth_record
from .units import LSB_PU

TRACE_KEYS = ("urms", "v1", "v2", "f_dft", "rocof", "J", "f_pll", "e_mag", "e_ang", "r_mag", "r_ang",
              "S_mag", "S_ang", "thd")


def unit_signals() -> list[tuple[str, str, np.ndarray]]:
    """Six noise-free unit signals (nuisances off, quantisation on): (name, description, int16 [3, 5000])."""
    pre = clean_pre_event(theta0=0.4)
    sag = {"t0": 0.3013, "psi": 0.0, "dur": 0.3, "rec_abrupt": True, "rec_level": 1.0, "rec_tau": 0.1, "special": "a"}
    sigs = [
        ("u1_clean_nominal", "clean nominal 50 Hz, 1.0 pu", quiet_normal()),
        ("u2_sag_bal_050", "balanced (type A) dip to 0.5 pu, t0 = 0.3013 s, 0.3 s, abrupt recovery",
         {"label": "sag_bal", "V": 0.5, "sag_type": "A", **sag}),
        ("u3_sag_typeC_050", "type C dip V = 0.5 (phase a special), t0 = 0.3013 s, 0.3 s",
         {"label": "sag_unbal", "V": 0.5, "sag_type": "C", **sag}),
        ("u4_jump_20deg", "+20 deg positive-sequence phase jump at t0 = 0.3013 s",
         {"label": "phase_jump", "t0": 0.3013, "dphi": np.radians(20.0), "dv1": 0.0}),
        ("u5_ramp_1hz_s", "frequency ramp +1 Hz/s from t0 = 0.2 s to the record end",
         {"label": "freq_event", "t0": 0.2, "rocof": 1.0, "df_target": 5.0, "dv1": 0.0}),
        ("u6_mod_10hz", "10 Hz modulation: V1 magnitude depth 5 %, angle 2 deg, sigma 0, from t0 = 0.3 s",
         {"label": "oscillation", "t0": 0.3, "depth": 0.05, "ang_amp": np.radians(2.0), "f_m": 10.0,
          "sigma": 0.0, "phi_m": 0.0, "phi_rel": np.pi / 2, "ramp": 0.05}),
    ]
    return [(n, d, synth_record(pre, ev)) for n, d, ev in sigs]


def reference_outputs(x: np.ndarray, fcfg: dict[str, Any], th: dict[str, Any], dtype=np.float64) -> list[dict[str, Any]]:
    """Traces, features, aux and rule outputs for records x (int16 [R, 3, 5000])."""
    tr = compute_traces(x, LSB_PU, dtype)
    feats, extra = features_from_traces(tr, fcfg)
    import pandas as pd
    F = pd.DataFrame({k: np.asarray(v, dtype=np.float64) for k, v in feats.items()})
    rules = classify(F, th)
    out = []
    for i in range(x.shape[0]):
        d = {
            "x": x[i].astype(np.int16),
            "urms": tr["urms"][i], "v1": tr["v1"][i], "v2": tr["v2"][i], "f_dft": tr["f_dft"][i],
            "rocof": tr["rocof_raw"][i], "J": tr["J"][i], "f_pll": tr["f_pll"][i],
            "fmask": extra["fmask"][i], "rmask": extra["rmask"][i],
            "e_mag": extra["e_mag"][i], "e_ang": extra["e_ang"][i], "r_mag": extra["r_mag"][i],
            "r_ang": extra["r_ang"][i], "S_mag": extra["S_mag"][i], "S_ang": extra["S_ang"][i], "thd": tr["thd"][i],
            "features": F.loc[i, FEATURE_NAMES].to_numpy(dtype=np.float64),
            "aux": F.loc[i, AUX_NAMES].to_numpy(dtype=np.float64),
            "rule_label": rules.loc[i, "rule_label"], "rule_reason": int(rules.loc[i, "rule_reason"]),
            "rule_onset": float(rules.loc[i, "rule_onset"]),
        }
        out.append(d)
    return out


def write_npz(path: Path, d: dict[str, Any], meta: dict[str, Any]) -> None:
    arrays = {k: (np.asarray(v, dtype=np.complex128) if np.iscomplexobj(v) else
                  (np.asarray(v, dtype=np.float64) if isinstance(v, np.ndarray) and v.dtype.kind == "f" else np.asarray(v)))
              for k, v in d.items()}
    arrays["x"] = d["x"].astype(np.int16)
    arrays["feature_names"] = np.array(FEATURE_NAMES)
    arrays["aux_names"] = np.array(AUX_NAMES)
    arrays["rule_reason_name"] = np.array(REASONS[d["rule_reason"]])
    arrays["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)


def float32_errors(ref: list[dict[str, Any]], f32: list[dict[str, Any]]) -> dict[str, float]:
    """Max |float32 - float64| per array over all golden records (NaN-aware; complex by modulus)."""
    errs: dict[str, float] = {}
    for key in TRACE_KEYS + ("features",):
        worst = 0.0
        for a, b in zip(ref, f32):
            x = np.asarray(a[key], dtype=np.complex128 if np.iscomplexobj(a[key]) else np.float64)
            y = np.asarray(b[key], dtype=x.dtype)
            if key == "J":  # angles: compare wrapped
                d = np.abs(np.angle(np.exp(1j * (x - y))))
            else:
                d = np.abs(x - y)
            ok = np.isfinite(d)
            if ok.any():
                worst = max(worst, float(d[ok].max()))
        errs[key] = worst
    return errs
