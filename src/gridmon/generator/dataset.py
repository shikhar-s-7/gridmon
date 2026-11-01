"""Split generation: seeding, label assignment, margin rejection, chunked synthesis, storage (B6.5).

Seeding (rule A1.3): ss = SeedSequence(split_seed); record i uses child ss.spawn(N)[i], from which
a 63-bit integer record seed is derived (stored in the manifest). Each record then has three
independent streams SeedSequence(record_seed).spawn(3) = (pre, event, noise). The label order is
shuffled with the next child of ss. No global RNG state is used anywhere.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .. import CLASSES, GENERATOR_VERSION
from ..io import config_hash, git_commit, write_json
from ..units import FS, FS_SYNTH, LSB_PU, N_RECORD
from . import events as ev_mod
from .events import Event, draw_event, event_to_record, margin_metrics
from .nuisance import PreEvent, draw_pre_event
from .synth import load_aa_fir, measure, synth_analog_ll, synth_time


@dataclass
class RecordSpec:
    """Everything needed to synthesise one record deterministically."""

    record_id: int
    label: str
    seed: int
    pre: PreEvent
    ev: Event
    metrics: dict[str, float]
    n_reject: int


def split_seeds(split_seed: int, n: int) -> tuple[list[int], np.random.Generator]:
    """63-bit record seeds from SeedSequence(split_seed).spawn(n), plus the label-shuffle generator."""
    ss = np.random.SeedSequence(split_seed)
    children = ss.spawn(n)
    seeds = [int(c.generate_state(1, np.uint64)[0] >> np.uint64(1)) for c in children]
    shuffle_rng = np.random.default_rng(ss.spawn(1)[0])
    return seeds, shuffle_rng


def record_streams(record_seed: int) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    """(pre, event, noise) generators of one record."""
    pre_ss, ev_ss, noise_ss = np.random.SeedSequence(record_seed).spawn(3)
    return np.random.default_rng(pre_ss), np.random.default_rng(ev_ss), np.random.default_rng(noise_ss)


def make_record_spec(record_id: int, label: str, seed: int, cfg: dict[str, Any], nuisance: str,
                     margins_ok: Callable[..., bool] | None = None) -> RecordSpec:
    """Pre-event draw once (common path), then event draw with margin rejection (event stream only)."""
    ok_fn = margins_ok or ev_mod.margins_ok
    rng_pre, rng_ev, _ = record_streams(seed)
    pre = draw_pre_event(rng_pre, cfg["nuisance"][nuisance])  # same code path for every class
    mc = cfg["margins"]
    onset = tuple(cfg["record"]["onset_s"])
    best: tuple[float, Event, dict[str, float]] | None = None
    for attempt in range(int(cfg["max_redraws"]) + 1):
        ev = draw_event(label, rng_ev, cfg["events"], onset)  # type: ignore[arg-type]
        m = margin_metrics(pre, ev, grid_hz=mc["grid_hz"], duration=cfg["record"]["duration_s"])
        if ok_fn(label, pre, ev, m, mc):
            return RecordSpec(record_id, label, seed, pre, ev, m, attempt)
        if label == "oscillation":
            exc = ev_mod.osc_band_excess(m, mc)
            if best is None or exc < best[0]:
                best = (exc, ev, m)
    if best is not None:
        # D-039: the D-034 band is infeasible for this pre-event (high U_pre x high background unbalance,
        # shift config): keep the least-violating draw (pre-event never redrawn); n_reject = max_redraws + 1.
        return RecordSpec(record_id, label, seed, pre, best[1], best[2], attempt + 1)
    raise RuntimeError(f"record {record_id} ({label}): label margins not met after {attempt} redraws")


def split_labels(scfg: dict[str, Any], classes: tuple[str, ...], fast_fraction: float | None) -> list[str]:
    """Balanced labels (per_class each) or round-robin over classes (preview 'total')."""
    if "per_class" in scfg:
        n = scfg["per_class"]
        if fast_fraction:
            n = max(1, int(round(n * fast_fraction)))
        return [c for c in classes for _ in range(n)]
    n = scfg["total"]
    if fast_fraction:
        n = max(len(classes), int(round(n * fast_fraction)))
    return [classes[i % len(classes)] for i in range(n)]


def generate_split(name: str, cfg: dict[str, Any], out_dir: Path, fast: bool = False,
                   progress: bool = True) -> dict[str, Any]:
    """Generate one split: X.npy (int16 [N, 3, 5000]), manifest.csv, meta.json in out_dir."""
    t_start = time.perf_counter()
    scfg = cfg["splits"][name]
    classes = tuple(cfg["classes"])
    assert classes == CLASSES, "class list in configs/dataset.yaml must match gridmon.CLASSES"
    labels = split_labels(scfg, classes, cfg["fast_fraction"] if fast else None)
    n = len(labels)
    seeds, shuffle_rng = split_seeds(scfg["seed"], n)
    labels = [labels[i] for i in shuffle_rng.permutation(n)]

    fir = load_aa_fir()
    h = fir["h"]
    t = synth_time(len(h))
    out_dir.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(out_dir / "X.npy", mode="w+", dtype=np.int16, shape=(n, 3, N_RECORD))
    rows: list[dict[str, Any]] = []
    chunk = int(cfg["chunk_records"])
    iterator = range(0, n, chunk)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"{name:>13s}", unit="chunk", leave=False)
    for c0 in iterator:
        ids = range(c0, min(c0 + chunk, n))
        specs = [make_record_spec(i, labels[i], seeds[i], cfg, scfg["nuisance"]) for i in ids]
        analog = np.stack([synth_analog_ll(s.pre, s.ev, t) for s in specs])
        noise_rngs = [record_streams(s.seed)[2] for s in specs]
        x, nsat = measure(analog, np.array([s.pre.snr_db for s in specs]), noise_rngs, h)
        X[c0:c0 + len(specs)] = x
        maxabs = np.abs(x.astype(np.int32)).reshape(len(specs), -1).max(axis=1)
        for s, ns, mx in zip(specs, nsat, maxabs):
            row: dict[str, Any] = {"record_id": s.record_id, "label": s.label, "seed": s.seed,
                                   "n_reject": s.n_reject}
            row.update(s.pre.to_record())
            row.update(event_to_record(s.ev))
            row.update(s.metrics)
            row.update({"max_abs_count": int(mx), "n_sat": int(ns)})
            rows.append(row)
    X.flush()
    del X
    man = pd.DataFrame(rows)
    man.to_csv(out_dir / "manifest.csv", index=False, float_format="%.10g")
    runtime = time.perf_counter() - t_start

    counts = Counter(labels)
    rej = man.groupby("label")["n_reject"].agg(["sum", "count"])
    rej_rate = {k: float(r["sum"] / (r["sum"] + r["count"])) for k, r in rej.iterrows()}
    fir_meta = {k: v for k, v in fir.items() if k not in ("h", "coefficients")}
    meta = {
        "split": name, "n_records": n, "counts": {c: counts.get(c, 0) for c in classes},
        "split_seed": scfg["seed"], "nuisance_config": scfg["nuisance"], "fast": fast,
        "sealed": bool(scfg.get("sealed", False)),
        "generator_version": GENERATOR_VERSION,
        "config_hash": config_hash(cfg, fir["coefficients"]),
        "git_commit": git_commit(),
        "fs_hz": FS, "fs_synth_hz": FS_SYNTH, "n_samples": N_RECORD, "channels": ["ab", "bc", "ca"],
        "dtype": "int16", "adc_bits": cfg["adc"]["bits"], "lsb_pu": LSB_PU,
        "units": "counts; v_pu = lsb_pu * count (pu of nominal LL peak)",
        "aa_fir": fir_meta,
        "timeline": "group-delay compensated: sample n <-> analog time n/fs; onset ev_t0 on this timeline",
        "margin_rejection_rate": rej_rate,
        "max_abs_count": int(man["max_abs_count"].max()), "n_records_saturated": int((man["n_sat"] > 0).sum()),
        "osc_band_fallback": int(((man["label"] == "oscillation") & (man["n_reject"] > int(cfg["max_redraws"]))).sum()),
        "runtime_s": round(runtime, 2),
    }
    write_json(out_dir / "meta.json", meta)
    return meta
