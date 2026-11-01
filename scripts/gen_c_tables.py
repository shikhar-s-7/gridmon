"""M6 (D-054): generate the C core's constant tables from the frozen Python reference.

Every DSP constant of the C core (DFT and THD twiddles, ROCOF kernel, Hann window, SOGI/PLL
coefficients, sequence operators, scale factors, feature/rule thresholds, index windows) is computed
here in float64 from the SAME Python functions the reference uses (gridmon.dsp.*, configs/
baseline.yaml, configs/frozen_methods.yaml) and written as float32 hexadecimal literals, so every
target (host, C28x, WebAssembly) uses bit-identical constants and nothing is computed with libm at
run time.

Comparison thresholds are rounded in the direction that keeps the comparison exact: for a float32
value x, `x < t` (float64 t) equals `x < up32(t)`, `x > t` equals `x > down32(t)`, `x >= t` equals
`x >= up32(t)` (up32/down32 = nearest float32 above/below t). Other constants round to nearest.

Outputs: c/tables/gm_tables.{h,c} (portable core) and c/host/gm_names.h (host-only name strings).
Usage: python scripts/gen_c_tables.py [--check]   (--check: exit 1 if the committed files differ)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from gridmon import CLASSES
from gridmon.baseline_rules import REASONS
from gridmon.dsp import envelope as env
from gridmon.dsp.freq import DT_PHASOR, LAG, MA, ROCOF_WIN, rocof_kernel
from gridmon.dsp.phasor import M_FIRST, STEP
from gridmon.dsp.pll import pi_coeffs, sogi_coeffs
from gridmon.dsp.thd import H_MAX
from gridmon.dsp.thd import WIN as THD_WIN
from gridmon.export_gbm import sha256_file
from gridmon.features import AUX_NAMES, FEATURE_NAMES
from gridmon.io import ROOT, load_baseline_config, load_yaml
from gridmon.units import A_OP, F_NOM, FS, N_CYCLE, N_RECORD, SQRT3, TS

OUT_DIR = ROOT / "c" / "tables"
NAMES_H = ROOT / "c" / "host" / "gm_names.h"
GEN_VERSION = 1


# ---------------------------------------------------------------------------------------------- float32 literals
def f32_near(v: float) -> np.float32:
    return np.float32(v)


def f32_up(v: float) -> np.float32:
    """Smallest float32 >= v."""
    t = np.float32(v)
    if float(t) < v:
        t = np.nextafter(t, np.float32(np.inf))
    return t


def f32_down(v: float) -> np.float32:
    """Largest float32 <= v."""
    t = np.float32(v)
    if float(t) > v:
        t = np.nextafter(t, np.float32(-np.inf))
    return t


def hexf(v: np.float32 | float) -> str:
    """C99 hexadecimal float literal (with f suffix) of a float32 value; parses back exactly."""
    x = float(np.float32(v))
    if x == 0.0:
        return "-0x0p+0f" if np.signbit(x) else "0x0p+0f"
    s = x.hex()  # e.g. '0x1.921fb60000000p+1' (exact: float32 -> float64 is exact)
    sign = "-" if s.startswith("-") else ""
    mant, exp = s.lstrip("-")[2:].split("p")
    if "." in mant:
        ip, fp = mant.split(".")
        fp = fp.rstrip("0")
        mant = ip + ("." + fp if fp else "")
    lit = f"{sign}0x{mant}p{exp}f"
    if float.fromhex(lit[:-1]) != x:
        raise AssertionError(f"hex literal round trip failed: {v!r} -> {lit}")
    return lit


def _rows(vals: list[str], per_line: int) -> str:
    return ",\n".join("    " + ", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line))


# ---------------------------------------------------------------------------------------------- content
def build() -> tuple[str, str, str]:
    """(gm_tables.h, gm_tables.c, gm_names.h) as text."""
    bcfg = load_baseline_config()
    fc = bcfg["features"]
    frozen = load_yaml(ROOT / "configs" / "frozen_methods.yaml")
    th = frozen["rules"]["thresholds"]  # frozen (TRAIN-tuned = defaults, D-036)
    model_sha = sha256_file(ROOT / "models" / "gbm_embedded.json")

    M = N_RECORD // STEP                                   # 1000 phasors
    ref_last = (fc["ref_end_sample"] - (STEP - 1)) // STEP  # n_m <= 749 -> m <= 149
    obs_first = -(-(fc["obs_start_sample"] - (STEP - 1)) // STEP)  # n_m >= 750 -> m >= 150
    end_first = M - fc["end_window_phasors"]
    m_fdft = M_FIRST + LAG + MA - 1                         # 58
    m_j = M_FIRST + 2 * LAG                                 # 59
    m_rocof = m_fdft + ROCOF_WIN - 1                        # 157
    n_urms = (N_RECORD - N_CYCLE) // (N_CYCLE // 2) + 1     # 99
    n_env = M // env.BLOCK                                  # 100
    assert (ref_last, obs_first, end_first, m_fdft, m_j, m_rocof, n_urms) == (149, 150, 900, 58, 59, 157, 99)

    sc = sogi_coeffs(F_NOM, ts=TS)
    pc = pi_coeffs(2 * np.pi * 20.0, 0.707, TS)
    rk = rocof_kernel(np.float64, ROCOF_WIN, DT_PHASOR)
    hann = env.hann(env.WIN, np.float64)
    tw100 = np.exp(-2j * np.pi * np.arange(N_CYCLE) / N_CYCLE)
    tw500 = np.exp(-2j * np.pi * np.arange(THD_WIN) / THD_WIN)

    ints: list[tuple[str, int, str]] = [
        ("GM_N_CH", 3, "channels ab, bc, ca (planar input)"),
        ("GM_N_RECORD", N_RECORD, "samples per channel per record (1 s at 5 kHz)"),
        ("GM_N_CYCLE", N_CYCLE, "samples per nominal cycle (DFT length)"),
        ("GM_STEP", STEP, "phasor decimation: phasor m at sample n = 5m + 4"),
        ("GM_M", M, "phasors per record"),
        ("GM_M_FIRST", M_FIRST, "first valid phasor"),
        ("GM_M_REF_LAST", ref_last, "REF = m 19..149 (n_m <= 749)"),
        ("GM_N_REF", ref_last - M_FIRST + 1, "phasors in REF"),
        ("GM_M_OBS_FIRST", obs_first, "OBS = m 150..999 (n_m >= 750)"),
        ("GM_M_END_FIRST", end_first, "END = m 900..999"),
        ("GM_N_END", fc["end_window_phasors"], "phasors in END"),
        ("GM_LAG", LAG, "angle lag for df_raw and J (one cycle at 1 kHz)"),
        ("GM_MA", MA, "moving-average length of df_raw"),
        ("GM_M_DFRAW_FIRST", M_FIRST + LAG, "first valid df_raw"),
        ("GM_M_FDFT_FIRST", m_fdft, "first valid f_dft"),
        ("GM_M_J_FIRST", m_j, "first valid J"),
        ("GM_ROCOF_WIN", ROCOF_WIN, "ROCOF least-squares window (phasors)"),
        ("GM_M_ROCOF_FIRST", m_rocof, "first valid ROCOF"),
        ("GM_N_FREF", ref_last - m_fdft + 1, "valid f_dft samples in REF (median for f_ref)"),
        ("GM_JUMP_MASK_HALF", int(fc["jump_mask_half"]), "f-mask = mJ +/- 40 phasors"),
        ("GM_N_URMS", n_urms, "Urms(1/2) values per channel (window 100, hop 50)"),
        ("GM_URMS_HOP", N_CYCLE // 2, "Urms(1/2) hop"),
        ("GM_P", n_env, "envelope samples (10-phasor blocks)"),
        ("GM_ENV_BLOCK", env.BLOCK, "phasors per envelope block"),
        ("GM_P_FIRST", env.P_FIRST, "first valid envelope sample"),
        ("GM_MED_HALF", env.MED_HALF, "detrending running median half width"),
        ("GM_ENV_WIN", env.WIN, "envelope spectrum window length"),
        ("GM_ENV_HOP", env.HOP, "envelope spectrum hop"),
        ("GM_ENV_NWIN", env.N_WIN, "envelope spectrum windows"),
        ("GM_ENV_F_FIRST", int(env.FREQS[0]), "first spectrum bin (Hz)"),
        ("GM_ENV_NF", len(env.FREQS), "spectrum bins (3..30 Hz)"),
        ("GM_ENV_FS", env.FS_ENV, "envelope rate (Hz) = twiddle table length"),
        ("GM_THD_WIN", THD_WIN, "THD window (samples)"),
        ("GM_THD_NWIN", N_RECORD // THD_WIN, "THD windows per record"),
        ("GM_THD_HMAX", H_MAX, "highest harmonic"),
        ("GM_FS_HZ", FS, "sample rate"),
    ]
    near: list[tuple[str, float, str]] = [
        ("GM_K_PI", np.pi, "pi"),
        ("GM_K_TWO_PI", 2 * np.pi, "2 pi"),
        ("GM_K_INV_2PI", 1 / (2 * np.pi), "1 / (2 pi)"),
        ("GM_K_RAD2DEG", 180 / np.pi, "degrees per radian"),
        ("GM_K_F_NOM", F_NOM, "nominal frequency (Hz)"),
        ("GM_K_FS", float(FS), "sample rate (Hz) for t = n / fs"),
        ("GM_K_DFT_SCALE", 2.0 / N_CYCLE, "one-cycle DFT scale 2/N"),
        ("GM_K_A_RE", A_OP.real, "Fortescue a = exp(j 2 pi/3), real"),
        ("GM_K_A_IM", A_OP.imag, "Fortescue a, imaginary"),
        ("GM_K_A2_RE", (A_OP * A_OP).real, "a^2, real"),
        ("GM_K_A2_IM", (A_OP * A_OP).imag, "a^2, imaginary"),
        ("GM_K_THIRD", 1.0 / 3.0, "1/3"),
        ("GM_K_DF_SCALE", 1.0 / (2.0 * np.pi * LAG * DT_PHASOR), "df_raw = wrap(dang) / (2 pi 0.02)"),
        ("GM_K_MA_INV", 1.0 / MA, "1 / 20"),
        ("GM_K_END_INV", 1.0 / fc["end_window_phasors"], "1 / 100"),
        ("GM_K_BLOCK_INV", 1.0 / env.BLOCK, "1 / 10"),
        ("GM_K_URMS_SCALE", 2.0 / N_CYCLE, "Urms = sqrt(2 acc / 100)"),
        ("GM_K_SPEC_SCALE", 2.0 / float(hann.sum()), "envelope spectrum scale 2 / sum(hann) = 2/24.5"),
        ("GM_K_INV_SQRT3", 1.0 / SQRT3, "alpha = (v_ab - v_ca)/sqrt3"),
        ("GM_K_HALF", 0.5, "0.5"),
        ("GM_K_TS", TS, "PLL sample time (s)"),
        ("GM_K_W0", 2 * np.pi * F_NOM, "PLL nominal angular frequency (rad/s)"),
        ("GM_K_PLL_VMIN", 0.05, "PLL error normalisation floor (pu)"),
        ("GM_K_SOGI_A1", sc["a"][1], "SOGI denominator a1 (D and Q)"),
        ("GM_K_SOGI_A2", sc["a"][2], "SOGI denominator a2"),
        ("GM_K_SOGI_BD0", sc["bd"][0], "SOGI D numerator b0 (b1 = 0)"),
        ("GM_K_SOGI_BD2", sc["bd"][2], "SOGI D numerator b2"),
        ("GM_K_SOGI_BQ0", sc["bq"][0], "SOGI Q numerator b0"),
        ("GM_K_SOGI_BQ1", sc["bq"][1], "SOGI Q numerator b1"),
        ("GM_K_SOGI_BQ2", sc["bq"][2], "SOGI Q numerator b2"),
        ("GM_K_PI_B0", pc["b0"], "PI Tustin b0 = Kp + Ki Ts/2"),
        ("GM_K_PI_B1", pc["b1"], "PI Tustin b1 = -Kp + Ki Ts/2"),
        ("GM_K_V1_FLOOR", fc["v1_floor"], "|V1| floor in |V2|/|V1| (pu)"),
        ("GM_K_RHO_DEN_FLOOR", fc["rho_den_floor"], "Bollen denominator floor"),
        ("GM_K_URMS_T_STEP", 0.01, "t_below/t_above: 0.01 s per Urms(1/2) value"),
        ("GM_K_PHASOR_T_STEP", 0.001, "t_rocof/t_fpll: 0.001 s per phasor sample"),
        ("GM_K_PCT", 100.0, "fraction -> percent"),
        ("GM_K_TINY", 1e-12, "floor of spectral median / peak in prominence and gap"),
    ]
    # thresholds: (name, float64 value, comparison the C code uses, source)
    thr: list[tuple[str, float, str, str]] = [
        ("GM_TH_URMS_LOW_LT", 0.90, "<", "features: Urms(1/2) < 0.90 (t_below_090, aux_t_dip)"),
        ("GM_TH_URMS_HIGH_GT", 1.10, ">", "features: Urms(1/2) > 1.10 (t_above_110, aux_t_swell)"),
        ("GM_TH_JUMP_DETECT_DEG_GT", float(fc["jump_detect_deg"]), ">", "features: j_max_deg > 3 deg -> jump mask"),
        ("GM_TH_JUMP_ONSET_RAD_GT", float(np.radians(fc["jump_detect_deg"])), ">", "aux_t_jump: |J| > radians(3)"),
        ("GM_TH_FREQ_ONSET_GT", float(fc["freq_onset_hz"]), ">", "aux_t_freq: |f_dft - f_ref| > 0.1 Hz"),
        ("GM_TH_ROCOF_FLAG_GT", float(fc["rocof_flag_hz_s"]), ">", "t_rocof_02: |ROCOF| > 0.2 Hz/s"),
        ("GM_TH_FPLL_FLAG_GT", float(fc["fpll_flag_hz"]), ">", "t_fpll_05: |f_pll - f_ref| > 0.5 Hz"),
        ("GM_TH_DIP_LT", float(th["dip_pu"]), "<", "rule 1: urms_min_1 < dip_pu"),
        ("GM_TH_RHO_GT", float(th["rho_th"]), ">", "rule 1: rho > rho_th -> sag_unbal"),
        ("GM_TH_SWELL_GT", float(th["swell_pu"]), ">", "rule 2: urms_max_3 > swell_pu"),
        ("GM_TH_OSC_MAG_GT", float(th["osc_mag_pct"]), ">", "rule 3: env_mag_amp_pct > osc_mag_pct"),
        ("GM_TH_OSC_ANG_GT", float(th["osc_ang_deg"]), ">", "rule 3: env_ang_amp_deg > osc_ang_deg"),
        ("GM_TH_OSC_PROM_GE", float(th["osc_prom"]), ">=", "rule 3: env_*_prom >= osc_prom"),
        ("GM_TH_JUMP_GT", float(th["jump_deg"]), ">", "rule 4: j_max_deg > jump_deg"),
        ("GM_TH_JUMP_DV1_LT", float(th["jump_max_dv1"]), "<", "rule 4: max |v1_rel - 1| < jump_max_dv1"),
        ("GM_TH_DF_GT", float(th["df_hz"]), ">", "rule 5: |df_end| > df_hz"),
    ]

    def thr_lit(v: float, op: str) -> np.float32:
        return f32_up(v) if op in ("<", ">=") else f32_down(v)

    arrays: list[tuple[str, int, list[str], str]] = [
        ("gm_tw100_re", N_CYCLE, [hexf(f32_near(v)) for v in tw100.real], "Re exp(-j 2 pi i/100): DFT and envelope twiddles"),
        ("gm_tw100_im", N_CYCLE, [hexf(f32_near(v)) for v in tw100.imag], "Im exp(-j 2 pi i/100)"),
        ("gm_tw500_re", THD_WIN, [hexf(f32_near(v)) for v in tw500.real], "Re exp(-j 2 pi i/500): THD twiddles"),
        ("gm_tw500_im", THD_WIN, [hexf(f32_near(v)) for v in tw500.imag], "Im exp(-j 2 pi i/500)"),
        ("gm_rocof_c", ROCOF_WIN, [hexf(f32_near(v)) for v in rk], "ROCOF kernel c_j = (j - 49.5)/(83325 * 0.001) (Hz/s per Hz)"),
        ("gm_hann50", env.WIN, [hexf(f32_near(v)) for v in hann], "symmetric Hann 0.5 - 0.5 cos(2 pi i/49)"),
    ]

    # ---------------------------------------------------------------- gm_tables.c
    c_lines = ["/* GENERATED by scripts/gen_c_tables.py (D-054) from the frozen Python reference. Do not edit.",
               " * float32 values as C99 hexadecimal literals (bit-exact on every target). Simulation-only. */",
               "", '#include "gm_tables.h"', ""]
    for name, n, vals, doc in arrays:
        c_lines.append(f"/* {doc} */")
        c_lines.append(f"const float {name}[{n}] = {{\n{_rows(vals, 5)}\n}};\n")
    c_text = "\n".join(c_lines)

    # ---------------------------------------------------------------- gm_tables.h (hash inserted last)
    h_lines = ["/* GENERATED by scripts/gen_c_tables.py (D-054) from the frozen Python reference. Do not edit.",
               " * Dimensions, float32 constants (hex literals, nearest rounding), comparison thresholds",
               " * (directed rounding: x < t -> up32(t), x > t -> down32(t), x >= t -> up32(t), so a float32",
               " * comparison equals the comparison with the float64 threshold) and table declarations.",
               " * Simulation-only. */",
               "#ifndef GM_TABLES_H", "#define GM_TABLES_H", ""]
    h_lines.append("/* ---- dimensions and index windows (5 kHz samples n, 1 kHz phasors m, 100 Hz envelope p) */")
    for name, v, doc in ints:
        h_lines.append(f"#define {name} {v}  /* {doc} */")
    h_lines.append("")
    h_lines.append("/* ---- constants (float32, nearest) */")
    for name, v, doc in near:
        h_lines.append(f"#define {name} ({hexf(f32_near(v))})  /* {doc}: {float(v):.10g} */")
    h_lines.append("")
    h_lines.append("/* ---- comparison thresholds (frozen: configs/frozen_methods.yaml rules, configs/baseline.yaml features) */")
    for name, v, op, doc in thr:
        h_lines.append(f"#define {name} ({hexf(thr_lit(v, op))})  /* {doc}; {op} {v:g} */")
    h_lines.append("")
    h_lines.append("/* ---- tables (gm_tables.c) */")
    for name, n, _, doc in arrays:
        h_lines.append(f"extern const float {name}[{n}];  /* {doc} */")
    h_lines.append("")
    h_lines.append(f'#define GM_MODEL_ID "gbm_embedded seed0 sha256 {model_sha[:16]}"')
    h_body = "\n".join(h_lines)
    digest = hashlib.sha256((h_body + "\n" + c_text).encode("utf-8")).hexdigest()
    h_text = (h_body + f'\n#define GM_TABLES_SHA256 "{digest}"\n#define GM_TABLES_ID "{digest[:16]}"\n'
              f"#define GM_TABLES_GEN_VERSION {GEN_VERSION}\n\n#endif /* GM_TABLES_H */\n")

    # ---------------------------------------------------------------- host-only names
    def cstrs(xs: list[str]) -> str:
        return _rows([f'"{x}"' for x in xs], 4)
    n_text = ("/* GENERATED by scripts/gen_c_tables.py. Host-only name strings (never part of the portable core). */\n"
              "#ifndef GM_NAMES_H\n#define GM_NAMES_H\n\n"
              f"static const char *const gm_feature_names[{len(FEATURE_NAMES)}] = {{\n{cstrs(FEATURE_NAMES)}\n}};\n\n"
              f"static const char *const gm_aux_names[{len(AUX_NAMES)}] = {{\n{cstrs(AUX_NAMES)}\n}};\n\n"
              f"static const char *const gm_class_names[{len(CLASSES)}] = {{\n{cstrs(list(CLASSES))}\n}};\n\n"
              f"static const char *const gm_reason_names[{len(REASONS)}] = {{\n{cstrs([REASONS[i] for i in range(len(REASONS))])}\n}};\n\n"
              "#endif /* GM_NAMES_H */\n")
    return h_text, c_text, n_text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="compare with the committed files, write nothing")
    args = ap.parse_args(argv)
    h_text, c_text, n_text = build()
    targets = [(OUT_DIR / "gm_tables.h", h_text), (OUT_DIR / "gm_tables.c", c_text), (NAMES_H, n_text)]
    if args.check:
        bad = [p for p, t in targets if not p.exists() or p.read_text(encoding="utf-8") != t]
        for p in bad:
            print(f"differs: {p.relative_to(ROOT).as_posix()}")
        return 1 if bad else 0
    for p, t in targets:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(t, encoding="utf-8", newline="\n")
    sha = h_text.split('GM_TABLES_ID "')[1][:16]
    print(f"wrote c/tables/gm_tables.{{h,c}} (tables id {sha}) and c/host/gm_names.h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
