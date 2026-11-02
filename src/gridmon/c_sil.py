"""ctypes wrapper of the M6 C core (software-in-the-loop, brief B13.1).

`CCore` loads build/libgridmon_sil.* (built by gridmon.c_build: GM_TRACES, GM_REPORT_NAN=1) and runs
records through the C core. Outputs mirror the Python reference: feature tables with the reference
column names (invalid features are NaN, exactly where the reference yields non-finite values),
rule labels/reasons/onsets, GBM labels/probabilities, and (per record) every golden-vector trace.
The C core sees only the int16 waveform (rule A1.5). Simulation-only.
"""

from __future__ import annotations

import ctypes as ct
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import CLASSES
from .baseline_rules import REASONS
from .features import AUX_NAMES, FEATURE_NAMES
from .units import F_NOM, LSB_PU

N_CH, N_REC, M, P, NWIN, NF, NTHD, NURMS = 3, 5000, 1000, 100, 5, 28, 10, 99
NFEAT, NAUX, NCLS = len(FEATURE_NAMES), len(AUX_NAMES), len(CLASSES)


class GmResult(ct.Structure):
    _fields_ = [("status", ct.c_int16), ("flags", ct.c_uint16),
                ("features", ct.c_float * NFEAT), ("feature_valid", ct.c_uint16 * NFEAT),
                ("aux", ct.c_float * NAUX), ("aux_valid", ct.c_uint16 * NAUX),
                ("rule_label", ct.c_int16), ("rule_reason", ct.c_int16),
                ("onset_estimate_s", ct.c_float), ("onset_valid", ct.c_uint16),
                ("gbm_enabled", ct.c_int16), ("gbm_label", ct.c_int16),
                ("gbm_proba", ct.c_float * NCLS), ("gbm_raw", ct.c_float * NCLS)]


class GmTraces(ct.Structure):
    _fields_ = [("urms", (ct.c_float * NURMS) * N_CH),
                ("v1_re", ct.c_float * M), ("v1_im", ct.c_float * M), ("v2_re", ct.c_float * M), ("v2_im", ct.c_float * M),
                ("df_dft", ct.c_float * M), ("rocof", ct.c_float * M), ("J", ct.c_float * M), ("f_pll_dev", ct.c_float * M),
                ("fmask", ct.c_int16 * M), ("rmask", ct.c_int16 * M),
                ("e_mag", ct.c_float * P), ("e_ang", ct.c_float * P), ("r_mag", ct.c_float * P), ("r_ang", ct.c_float * P),
                ("S_mag", (ct.c_float * NF) * NWIN), ("S_ang", (ct.c_float * NF) * NWIN),
                ("thd", ct.c_float * NTHD)]


def _ptr(a: np.ndarray, ctype):
    return a.ctypes.data_as(ct.POINTER(ctype))


class CCore:
    """Host build of the C core."""

    def __init__(self, lib_path: Path | None = None, build: bool = True):
        if lib_path is None:
            from .c_build import build_sil
            lib_path = build_sil() if build else None
        if lib_path is None:
            raise FileNotFoundError("no SIL library")
        self.path = Path(lib_path)
        lib = ct.CDLL(str(self.path))
        self.lib = lib
        for name in ("gm_sil_sizeof_ctx", "gm_sil_sizeof_result", "gm_sil_sizeof_traces", "gm_sil_ctx_words"):
            getattr(lib, name).restype = ct.c_uint32
        lib.gm_sil_version.restype = ct.c_char_p
        lib.gm_sil_batch.restype = ct.c_int32
        lib.gm_sil_batch.argtypes = [ct.POINTER(ct.c_int16), ct.c_int32, ct.c_float,
                                     ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint16), ct.POINTER(ct.c_float),
                                     ct.POINTER(ct.c_uint16), ct.POINTER(ct.c_int16), ct.POINTER(ct.c_int16),
                                     ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint16), ct.POINTER(ct.c_int16),
                                     ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint16),
                                     ct.POINTER(ct.c_int16)]
        lib.gm_sil_traces.restype = ct.c_int32
        lib.gm_sil_traces.argtypes = [ct.POINTER(ct.c_int16), ct.c_int32, ct.c_float, ct.POINTER(GmResult),
                                      ct.POINTER(GmTraces)]
        lib.gm_sil_gbm.restype = ct.c_int32
        lib.gm_sil_gbm.argtypes = [ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint16), ct.c_int32, ct.POINTER(ct.c_int16),
                                   ct.POINTER(ct.c_float), ct.POINTER(ct.c_float)]
        lib.gm_sil_error_probe.restype = ct.c_int32
        lib.gm_sil_error_probe.argtypes = [ct.c_int32, ct.POINTER(ct.c_int16)]
        if lib.gm_sil_sizeof_result() != ct.sizeof(GmResult) or lib.gm_sil_sizeof_traces() != ct.sizeof(GmTraces):
            raise RuntimeError("ctypes layout does not match the C structs")

    # ------------------------------------------------------------------ info
    @property
    def version(self) -> str:
        return self.lib.gm_sil_version().decode()

    @property
    def sizeof_ctx(self) -> int:
        return int(self.lib.gm_sil_sizeof_ctx())

    @property
    def ctx_words(self) -> int:
        return int(self.lib.gm_sil_ctx_words())

    def error_probe(self, what: int) -> int:
        x = np.zeros(N_CH * N_REC, dtype=np.int16)
        return int(self.lib.gm_sil_error_probe(what, _ptr(x, ct.c_int16)))

    def gbm(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """C walker on feature rows F [N, 37] (NaN = invalid, D-051). Returns (label, proba, raw)."""
        F = np.asarray(F, dtype=np.float64)
        valid = np.ascontiguousarray(np.isfinite(F).astype(np.uint16))
        f32 = np.ascontiguousarray(np.where(np.isfinite(F), F, 0.0).astype(np.float32))
        n = F.shape[0]
        lab = np.zeros(n, np.int16)
        proba = np.zeros((n, NCLS), np.float32)
        raw = np.zeros((n, NCLS), np.float32)
        rc = self.lib.gm_sil_gbm(_ptr(f32, ct.c_float), _ptr(valid, ct.c_uint16), n, _ptr(lab, ct.c_int16),
                                 _ptr(proba, ct.c_float), _ptr(raw, ct.c_float))
        if rc != 0:
            raise RuntimeError("GBM disabled in this build")
        return lab, proba, raw

    # ------------------------------------------------------------------ batch
    def process(self, X: np.ndarray, lsb: float = LSB_PU, chunk: int = 500) -> dict[str, Any]:
        """Run records X (int16 [N, 3, 5000]) through the core. Returns arrays and `features` (DataFrame:
        37 features + 8 aux columns, NaN where the core flags a value invalid) and `seconds` (C time)."""
        N = int(X.shape[0])
        out = {"features_raw": np.zeros((N, NFEAT), np.float32), "feature_valid": np.zeros((N, NFEAT), np.uint16),
               "aux_raw": np.zeros((N, NAUX), np.float32), "aux_valid": np.zeros((N, NAUX), np.uint16),
               "rule_label": np.zeros(N, np.int16), "rule_reason": np.zeros(N, np.int16),
               "onset": np.zeros(N, np.float32), "onset_valid": np.zeros(N, np.uint16),
               "gbm_label": np.zeros(N, np.int16), "gbm_proba": np.zeros((N, NCLS), np.float32),
               "gbm_raw": np.zeros((N, NCLS), np.float32), "flags": np.zeros(N, np.uint16), "status": np.zeros(N, np.int16)}
        secs = 0.0
        for c0 in range(0, N, chunk):
            x = np.ascontiguousarray(np.asarray(X[c0:c0 + chunk], dtype=np.int16))
            n = x.shape[0]
            if x.shape[1:] != (N_CH, N_REC):
                raise ValueError("records must be int16 [N, 3, 5000]")
            v = {k: np.ascontiguousarray(a[c0:c0 + n]) for k, a in out.items()}
            t0 = time.perf_counter()
            self.lib.gm_sil_batch(_ptr(x, ct.c_int16), n, ct.c_float(lsb),
                                  _ptr(v["features_raw"], ct.c_float), _ptr(v["feature_valid"], ct.c_uint16),
                                  _ptr(v["aux_raw"], ct.c_float), _ptr(v["aux_valid"], ct.c_uint16),
                                  _ptr(v["rule_label"], ct.c_int16), _ptr(v["rule_reason"], ct.c_int16),
                                  _ptr(v["onset"], ct.c_float), _ptr(v["onset_valid"], ct.c_uint16),
                                  _ptr(v["gbm_label"], ct.c_int16), _ptr(v["gbm_proba"], ct.c_float),
                                  _ptr(v["gbm_raw"], ct.c_float), _ptr(v["flags"], ct.c_uint16),
                                  _ptr(v["status"], ct.c_int16))
            secs += time.perf_counter() - t0
            for k, a in v.items():
                out[k][c0:c0 + n] = a
        F = np.where(out["feature_valid"] > 0, out["features_raw"].astype(np.float64), np.nan)
        A = np.where(out["aux_valid"] > 0, out["aux_raw"].astype(np.float64), np.nan)
        df = pd.DataFrame(np.concatenate([F, A], axis=1), columns=FEATURE_NAMES + AUX_NAMES)
        out["features"] = df
        out["rule_label_name"] = np.array([CLASSES[i] for i in out["rule_label"]])
        out["rule_reason_name"] = np.array([REASONS[int(i)] for i in out["rule_reason"]])
        out["rule_onset"] = np.where(out["onset_valid"] > 0, out["onset"].astype(np.float64), np.nan)
        out["seconds"] = secs
        return out

    # ------------------------------------------------------------------ traces
    def traces(self, x: np.ndarray, lsb: float = LSB_PU) -> dict[str, Any]:
        """One record (int16 [3, 5000]) -> golden-vector arrays (same keys/units as golden/*.npz)."""
        x = np.ascontiguousarray(np.asarray(x, dtype=np.int16))
        res, tr = GmResult(), GmTraces()
        st = self.lib.gm_sil_traces(_ptr(x, ct.c_int16), x.shape[-1], ct.c_float(lsb), ct.byref(res), ct.byref(tr))

        def arr(field) -> np.ndarray:
            return np.ctypeslib.as_array(field).astype(np.float64)

        feats = np.array([res.features[i] if res.feature_valid[i] else np.nan for i in range(NFEAT)], dtype=np.float64)
        aux = np.array([res.aux[i] if res.aux_valid[i] else np.nan for i in range(NAUX)], dtype=np.float64)
        return {
            "status": int(st), "flags": int(res.flags),
            "urms": arr(tr.urms), "v1": arr(tr.v1_re) + 1j * arr(tr.v1_im), "v2": arr(tr.v2_re) + 1j * arr(tr.v2_im),
            "f_dft": F_NOM + arr(tr.df_dft), "rocof": arr(tr.rocof), "J": arr(tr.J), "f_pll": F_NOM + arr(tr.f_pll_dev),
            "fmask": np.ctypeslib.as_array(tr.fmask).astype(bool), "rmask": np.ctypeslib.as_array(tr.rmask).astype(bool),
            "e_mag": arr(tr.e_mag), "e_ang": arr(tr.e_ang), "r_mag": arr(tr.r_mag), "r_ang": arr(tr.r_ang),
            "S_mag": arr(tr.S_mag), "S_ang": arr(tr.S_ang), "thd": arr(tr.thd),
            "features": feats, "aux": aux, "feature_valid": np.array(res.feature_valid[:], dtype=bool),
            "rule_label": CLASSES[res.rule_label], "rule_reason": int(res.rule_reason),
            "rule_onset": float(res.onset_estimate_s) if res.onset_valid else float("nan"),
            "gbm_label": int(res.gbm_label), "gbm_proba": np.array(res.gbm_proba[:], dtype=np.float32),
        }


# ---------------------------------------------------------------------- SIL comparison gates
# D-069 (documentation-only, no method change): on top of the FEATURES.md gates (features.TOL_GATES, "reference
# column >= value"), the SIL compares rho only where the rules use it: reference urms_min_1 < 0.90, exactly the
# rules' dip condition (dip_pu; the gate was 0.95 until D-079). Outside that, rho depends on the first argmin of a
# flat |V1|, where float32 ties are legitimate (D-060); such differences are still counted and reported as "ungated".
SIL_GATES_BELOW: dict[str, list[tuple[str, float]]] = {"rho": [("urms_min_1", 0.90)]}


def sil_gate(F_ref: pd.DataFrame, name: str) -> np.ndarray:
    """Records on which feature `name` is compared (finite reference, FEATURES.md gates, SIL gates)."""
    from .features import TOL_GATES
    gate = np.isfinite(F_ref[name].to_numpy(dtype=float))
    for col, lo in TOL_GATES.get(name, []):
        gate &= F_ref[col].to_numpy(dtype=float) >= lo
    for col, hi in SIL_GATES_BELOW.get(name, []):
        gate &= F_ref[col].to_numpy(dtype=float) < hi
    return gate


def sil_tolerance_check(F_ref: pd.DataFrame, F_test: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """features.tolerance_check with the SIL gates of D-069; adds n_over_tol_ungated for SIL-gated features."""
    from .features import FEATURE_SPEC, REL_TOL, tolerance_check
    out = tolerance_check(F_ref, F_test)
    for name, unit, tol, _ in FEATURE_SPEC:
        if name not in SIL_GATES_BELOW:
            continue
        a = F_ref[name].to_numpy(dtype=float)
        d = np.abs(a - F_test[name].to_numpy(dtype=float))
        gate = sil_gate(F_ref, name)
        lim = tol + REL_TOL.get(name, 0.0) * np.abs(a)
        out[name].update({"n_over_tol_ungated": out[name]["n_over_tol"], "n_compared_ungated": out[name]["n_compared"],
                          "sil_gate": " and ".join(f"{c} < {h}" for c, h in SIL_GATES_BELOW[name]),
                          "n_compared": int(gate.sum()), "n_over_tol": int((gate & ~(d <= lim)).sum()),
                          "max_abs_diff": float(d[gate].max()) if gate.any() else 0.0})
    return out


# ---------------------------------------------------------------------- golden comparison
GOLDEN_TOL: dict[str, float] = {"urms": 1e-5, "v1": 1e-5, "v2": 1e-5, "f_dft": 1e-4, "rocof": 1e-3, "J": 1e-4,
                                "f_pll": 3e-3, "e_mag": 1e-5, "e_ang": 1.8e-4, "r_mag": 1e-5, "r_ang": 1e-4,
                                "S_mag": 1e-3, "S_ang": 1e-3, "thd": 1e-6}  # golden/README.md


def compare_golden(g: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    """Compare C traces with one golden file (tolerances of golden/README.md, features of FEATURES.md)."""
    from .features import FEATURE_SPEC
    res: dict[str, Any] = {"traces": {}, "ok": True}
    for key, tol in GOLDEN_TOL.items():
        a = np.asarray(g[key])
        b = np.asarray(c[key])
        fin_a = np.isfinite(a)
        same_nan = bool(np.array_equal(fin_a, np.isfinite(b)))
        if key == "J":
            d = np.abs(np.angle(np.exp(1j * (a - b))))
        else:
            d = np.abs(a - b)
        dm = float(np.max(d[fin_a])) if fin_a.any() else 0.0
        ok = same_nan and dm <= tol
        res["traces"][key] = {"max_abs_diff": dm, "tol": tol, "nan_pattern_equal": same_nan, "ok": ok}
        res["ok"] &= ok
    for key in ("fmask", "rmask"):
        eq = bool(np.array_equal(np.asarray(g[key], bool), c[key]))
        res["traces"][key] = {"equal": eq, "ok": eq}
        res["ok"] &= eq
    names = [f[0] for f in FEATURE_SPEC]
    Fg = pd.DataFrame([np.concatenate([g["features"], g["aux"]])], columns=names + AUX_NAMES)
    Fc = pd.DataFrame([np.concatenate([c["features"], c["aux"]])], columns=names + AUX_NAMES)
    tc = sil_tolerance_check(Fg, Fc)
    nan_eq = bool(np.array_equal(np.isfinite(g["features"]), np.isfinite(c["features"])))
    res["features_over_tol"] = [k for k, v in tc.items() if v["n_over_tol"]]
    res["features_nan_pattern_equal"] = nan_eq
    res["rule_label_equal"] = str(g["rule_label"]) == c["rule_label"]
    res["rule_reason_equal"] = int(g["rule_reason"]) == c["rule_reason"]
    res["ok"] &= not res["features_over_tol"] and nan_eq and res["rule_label_equal"] and res["rule_reason_equal"]
    return res
