"""M6 portability checks of the portable C core (c/src, c/include, c/tables): static rules for C28x
(16-bit char/int, float-only, no allocation, no mutable static state) and cross compiles (-m32, 16-bit int)."""

from __future__ import annotations

import pytest

from gridmon.c_build import BUILD_DIR, BuildError, build_checks, find_compiler
from gridmon.c_portability import cross_compile_checks, portable_files, static_checks


def _cc_or_skip():
    try:
        return find_compiler(install_zig=False)
    except BuildError as e:
        pytest.skip(str(e))


def test_portable_sources_exist():
    names = {p.name for p in portable_files()}
    for n in ("gridmon.h", "gm_internal.h", "gm_api.c", "gm_rms.c", "gm_phasor.c", "gm_freq.c", "gm_pll.c",
              "gm_envelope.c", "gm_thd.c", "gm_features.c", "gm_rules.c", "gm_gbm.c", "gm_tables.c", "gm_tables.h"):
        assert n in names


def test_static_rules_hold():
    v = static_checks()
    assert {k: x for k, x in v.items() if x} == {}


def test_static_checker_detects_violations():
    """The checker itself flags each category on a deliberately bad file."""
    bad = BUILD_DIR / "portability_selftest" / "bad.c"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(
        "#include <stdio.h>\nstatic float s_buf[4];\nfloat g_x = 1.0f;\nconst float k_ok[1] = {1.0f};\ndouble d;\n"
        "static int f(int x) { return x > 0 ? f(x - 1) : (int)(0.5 + sizeof(int)); }\n"
        "void g(void) { char b[2]; union { float f; } u; float y = 1e-3; (void)b; (void)u; (void)y; malloc(1); }\n")
    v = static_checks([bad])
    for key in ("double", "char_data", "plain_int", "forbidden_call", "forbidden_header", "static_mutable",
                "global_mutable", "float_literal_without_f", "type_punning", "sizeof", "recursion"):
        assert v[key], key
    assert not any("k_ok" in x for x in v["global_mutable"])


def test_build_variants_compile_cleanly():
    """GM_ENABLE_GBM=0, no traces / GM_REPORT_NAN=0 and traces / NaN reporting all compile with -Werror."""
    cc = _cc_or_skip()
    res = build_checks(cc)
    assert all(v == "ok" for v in res.values()), res


@pytest.fixture(scope="module")
def cross():
    _cc_or_skip()
    return cross_compile_checks()


def test_cross_compile_m32(cross):
    r = cross["m32"]
    if r["status"] == "skipped":
        pytest.skip(r.get("note", "no 32-bit toolchain"))
    assert r["status"] == "pass", r


def test_cross_compile_16bit_int(cross):
    r = cross["int16"]
    if r["status"] == "skipped":
        pytest.skip(r.get("note", "no 16-bit-int toolchain"))
    assert r["status"] == "pass", r


def _script(name: str):
    import importlib.util

    from gridmon.io import ROOT
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_c2000_compile_check_dry_run(capsys):
    """L2 script (run locally with TI cl2000): every core/port source in both GBM configurations, C28x flags."""
    mod = _script("c2000_compile_check")
    assert mod.main(["--dry-run"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 * len(mod.SOURCES)
    for flag in ("-v28", "--float_support=fpu32", "--tmu_support=tmu0", "--abi=eabi", "--float_operations_allowed=32"):
        assert all(flag in ln for ln in lines)
    names = {s.name for s in mod.SOURCES}
    assert {"gm_api.c", "gm_port.c", "gm_port_fir.c", "gm_model_gbm.c", "gm_tables.c"} <= names


def test_pil_records_current():
    _cc_or_skip()
    assert _script("make_pil_records").main(["--check"]) == 0
