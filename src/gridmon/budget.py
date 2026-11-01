"""Embedded budget estimate for one 1 s record (brief B7.6): operations and working memory per block.

Counts are analytic, from the block parameters (no profiling). 'mac' = real multiply-accumulates
(a real x complex product counts 2); 'ops' = other adds/compares/multiplies; 'tf' = transcendental
or root calls (sin, cos, atan2, sqrt, division). Memory is float32 (4 B) / complex float32 (8 B);
'ram' = mutable working state in a streaming C design, 'const' = tables that can live in flash.
The 30 KB int16 record buffer is excluded (brief: budget is besides it).
"""

from __future__ import annotations

from typing import Any

from .dsp import envelope as env
from .dsp.freq import LAG, MA, ROCOF_WIN
from .dsp.phasor import M_FIRST
from .dsp.thd import H_MAX, WIN as THD_WIN
from .units import N_CYCLE, N_RECORD

F32, C64 = 4, 8


def estimate(n: int = N_RECORD, n_ch: int = 3) -> dict[str, Any]:
    M = n // 5
    m_valid = M - M_FIRST                         # 981 phasor samples
    k_urms = (n - N_CYCLE) // (N_CYCLE // 2) + 1  # 99
    n_ref = (749 - 4) // 5 + 1 - M_FIRST          # 131 phasor samples in REF
    P = M // env.BLOCK
    W = n // THD_WIN
    blocks: dict[str, dict[str, Any]] = {
        "scale_to_pu": {"mac": 0, "ops": n * n_ch, "tf": 0, "ram": 0, "const": 0,
                        "note": "int16 -> float, x LSB (done on the fly by each consumer)"},
        "urms_half": {"mac": k_urms * n_ch * N_CYCLE, "ops": k_urms * n_ch * 4, "tf": k_urms * n_ch,
                      "ram": n_ch * 4 * F32 + 16, "const": 0,
                      "note": "direct 100-sample sums per window; per-channel min/max, threshold counters"},
        "dft_phasors": {"mac": m_valid * n_ch * N_CYCLE * 2, "ops": m_valid * n_ch * 2, "tf": 0,
                        "ram": n_ch * C64, "const": N_CYCLE * C64,
                        "note": "non-recursive one-cycle DFT at 1 kHz, 100-entry twiddle table"},
        "sequence_mag_angle": {"mac": m_valid * 2 * 8, "ops": m_valid * 10, "tf": m_valid * 3,
                               "ram": m_valid * 3 * F32, "const": 0,
                               "note": "V1, V2; store |V1|, angV1, |V2|/|V1| (3 x 981 floats); V2 kept only for REF and at running argmin"},
        "f_dft_J": {"mac": (m_valid - LAG) + (m_valid - LAG - MA + 1) * MA, "ops": m_valid * 6, "tf": 0,
                    "ram": m_valid * F32, "const": 0,
                    "note": "lag-20 angle difference, 20-tap moving average (direct), J from the angle buffer; f_dft buffer kept for the jump-masked second pass"},
        "rocof": {"mac": (m_valid - LAG - MA + 1 - ROCOF_WIN + 1) * ROCOF_WIN, "ops": m_valid * 2, "tf": 0,
                  "ram": 2 * F32, "const": ROCOF_WIN * F32,
                  "note": "100-tap least-squares slope kernel (D-016), second pass after the jump mask"},
        "dsogi_pll": {"mac": n * (4 * 5 + 4 + 2 + 4 + 2), "ops": n * 8, "tf": n * 4,
                      "ram": 12 * F32, "const": 0,
                      "note": "4 DF2T biquads, pos.-seq., sin/cos/sqrt/div per sample; f_pll features accumulated on the fly after REF"},
        "envelope": {"mac": 2 * m_valid + 2 * env.N_WIN * len(env.FREQS) * env.WIN * 2,
                     "ops": 2 * (P - env.P_FIRST) * 110 + 2 * env.N_WIN * len(env.FREQS) * 3,
                     "tf": 2 * env.N_WIN * len(env.FREQS),
                     "ram": 4 * P * F32 + 2 * env.N_WIN * len(env.FREQS) * F32 + 21 * F32,
                     "const": env.FS_ENV * C64 + env.WIN * F32,
                     "note": "block means, 21-sample running median (insertion sort), 5 Hann windows x 28 bins"},
        "thd": {"mac": W * n_ch * H_MAX * THD_WIN * 2, "ops": W * n_ch * H_MAX * 3, "tf": W * n_ch * 2,
                "ram": W * F32 + H_MAX * C64, "const": THD_WIN * C64,
                "note": "direct DFT at bins 5h, h = 1..15, 500-entry twiddle table"},
        "features_and_rules": {"mac": 0, "ops": 12 * m_valid + 200, "tf": 50,
                               "ram": (2 * n_ref + 3 * n_ref) * F32 + 64 * F32, "const": 0,
                               "note": "reductions; median buffers for REF (|V1|, u2, f_dft, Re/Im V2)"},
    }
    tot = {k: sum(b[k] for b in blocks.values()) for k in ("mac", "ops", "tf", "ram", "const")}
    total_ops = tot["mac"] + tot["ops"] + tot["tf"]
    flags = []
    if tot["ram"] > 32 * 1024:
        flags.append(f"working RAM {tot['ram'] / 1024:.1f} KB exceeds 32 KB")
    if total_ops > 3e6:
        flags.append(f"{total_ops / 1e6:.2f} M operations is above ~2x the 1.5 M expectation")
    return {
        "record": {"n_samples": n, "channels": n_ch, "fs_hz": 5000, "int16_record_bytes": n * n_ch * 2},
        "blocks": blocks,
        "totals": {**tot, "total_ops": total_ops, "ram_kb": round(tot["ram"] / 1024, 2),
                   "const_kb": round(tot["const"] / 1024, 2)},
        "c2000_notes": [
            "C28x has 16-bit chars: sizeof(float) = 2 words; byte figures here are 8-bit bytes.",
            "Use float only (EABI double is 64-bit; F2837x has no FPU64): build with -Wdouble-promotion.",
            "TMU (F2837xD/F28004x/F28P65x) accelerates sin/cos/atan2/sqrt/div; without it ~5000 x 4 + ~3000 libm calls per record.",
            "Estimated 1.5 M ops x ~2-4 cycles -> ~3-6 M cycles (~15-30 ms at 200 MHz) per record in background context.",
            "Running median (21 samples, 196 times) uses an insertion sort on a 21-float scratch buffer: bounded, no malloc.",
            "Second pass over the phasor-rate buffers (jump mask before ROCOF/df features) needs the 4 KB f_dft buffer.",
        ],
        "flags": flags,
    }
