"""ctypes wrapper of the M7 port-layer host simulation (firmware/c2000/gm_port.c + host/gm_port_sim.c).

`PortSim` feeds control-ISR-rate samples (float32 pu, [n, 3]) through gm_port_push sample by sample and
runs gm_port_background every `bg_every` inputs (the background loop), optionally blocked for
`proc_delay` inputs after each classified record (target record-processing time; overrun check).
Simulation-only.
"""

from __future__ import annotations

import ctypes as ct
from pathlib import Path
from typing import Any

import numpy as np

from . import CLASSES
from .baseline_rules import REASONS

TRIG_NAMES = {0x1: "urms", 0x2: "jump", 0x4: "freq", 0x8: "env"}


class GmTelemetry(ct.Structure):
    _fields_ = [("seq", ct.c_uint32), ("t_trigger", ct.c_uint32), ("t_record", ct.c_uint32), ("cause", ct.c_uint16),
                ("status", ct.c_int16), ("flags", ct.c_uint16), ("rule_label", ct.c_int16), ("rule_reason", ct.c_int16),
                ("onset_s", ct.c_float), ("onset_valid", ct.c_uint16), ("onset_est_s", ct.c_float),
                ("gbm_label", ct.c_int16), ("gbm_max_proba", ct.c_float), ("overruns", ct.c_uint32),
                ("triggers", ct.c_uint32), ("resyncs", ct.c_uint32), ("clipped", ct.c_uint32)]


def cause_names(c: int) -> list[str]:
    return [n for b, n in TRIG_NAMES.items() if c & b]


class PortSim:
    def __init__(self, f_in: int = 50000, input_ln: bool = False, gain=(1.0, 1.0, 1.0), offset=(0.0, 0.0, 0.0),
                 bg_every: int = 50, proc_delay: int = 0, lib_path: Path | None = None):
        if lib_path is None:
            from .c_build import build_port_sim
            lib_path = build_port_sim()
        lib = ct.CDLL(str(lib_path))
        self.lib = lib
        lib.gm_port_sim_create.restype = ct.c_void_p
        lib.gm_port_sim_create.argtypes = [ct.c_int32, ct.c_int16, ct.POINTER(ct.c_float), ct.POINTER(ct.c_float),
                                           ct.c_int32, ct.c_int32]
        lib.gm_port_sim_feed.restype = ct.c_int32
        lib.gm_port_sim_feed.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_int32, ct.POINTER(GmTelemetry), ct.c_int32]
        lib.gm_port_sim_decimate.restype = ct.c_int32
        lib.gm_port_sim_decimate.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_int32, ct.POINTER(ct.c_int16), ct.c_int32]
        lib.gm_port_sim_destroy.argtypes = [ct.c_void_p]
        for name in ("gm_port_sim_head", "gm_port_sim_overruns", "gm_port_sim_triggers"):
            getattr(lib, name).restype = ct.c_uint32
            getattr(lib, name).argtypes = [ct.c_void_p]
        for name in ("gm_port_sim_sizeof_port", "gm_port_sim_sizeof_telemetry", "gm_port_sim_port_words"):
            getattr(lib, name).restype = ct.c_uint32
        if lib.gm_port_sim_sizeof_telemetry() != ct.sizeof(GmTelemetry):
            raise RuntimeError("telemetry layout mismatch")
        g = (ct.c_float * 3)(*gain)
        o = (ct.c_float * 3)(*offset)
        self.h = lib.gm_port_sim_create(int(f_in), int(bool(input_ln)), g, o, int(bg_every), int(proc_delay))
        if not self.h:
            raise ValueError(f"unsupported f_in {f_in}")
        self.f_in = int(f_in)

    def __del__(self):
        if getattr(self, "h", None):
            self.lib.gm_port_sim_destroy(self.h)
            self.h = None

    def feed(self, x: np.ndarray, max_tel: int = 64) -> list[dict[str, Any]]:
        x = np.ascontiguousarray(x, dtype=np.float32)
        buf = (GmTelemetry * max_tel)()
        k = self.lib.gm_port_sim_feed(self.h, x.ctypes.data_as(ct.POINTER(ct.c_float)), x.shape[0], buf, max_tel)
        out = []
        for t in buf[:k]:
            d = {f: getattr(t, f) for f, _ in GmTelemetry._fields_}
            d["rule_label_name"] = CLASSES[t.rule_label]
            d["rule_reason_name"] = REASONS[int(t.rule_reason)]
            d["gbm_label_name"] = CLASSES[t.gbm_label] if t.gbm_label >= 0 else None
            d["cause_names"] = cause_names(t.cause)
            out.append(d)
        return out

    def decimate(self, x: np.ndarray) -> np.ndarray:
        """Decimated ring samples (int16 [m, 3]) for inputs x (float32 [n, 3]); trigger not run."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        m = x.shape[0] // 4 + 8
        y = np.zeros((m, 3), dtype=np.int16)
        k = self.lib.gm_port_sim_decimate(self.h, x.ctypes.data_as(ct.POINTER(ct.c_float)), x.shape[0],
                                          y.ctypes.data_as(ct.POINTER(ct.c_int16)), m)
        return y[:k]

    @property
    def head(self) -> int:
        return int(self.lib.gm_port_sim_head(self.h))

    @property
    def overruns(self) -> int:
        return int(self.lib.gm_port_sim_overruns(self.h))

    @property
    def triggers(self) -> int:
        return int(self.lib.gm_port_sim_triggers(self.h))

    @property
    def sizeof_port(self) -> int:
        return int(self.lib.gm_port_sim_sizeof_port())

    @property
    def port_words(self) -> int:
        return int(self.lib.gm_port_sim_port_words())
