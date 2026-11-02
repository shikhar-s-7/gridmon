"""Render reports/S2E1_RESULTS.md from reports/s2e1_results.json (+ hand-written reports/s2e1_findings.md).

Every number comes from the JSON; the findings file carries the interpretation (findings with [SIM] tags,
plain-language takeaways, limitations). Target length <= 150 lines.
"""

from __future__ import annotations

from typing import Any

from . import CLASSES

ARMS = ("R", "G0", "GW", "GP", "HW", "H0")
NAMES = {"R": "R (Stage 1 rules)", "G0": "G0 (Stage 1 GBM)", "GW": "GW (GBM, wide training)",
         "GP": "GP (GBM, physics subset)", "HW": "HW (hybrid GW/R)", "H0": "H0 (hybrid G0/R)"}
SETS = ("test2_shift", "test2_novel")


def _f(x: float | None, d: int = 4) -> str:
    return "-" if x is None else f"{x:.{d}f}"


def _s(x: float, d: int = 4) -> str:
    """Signed fixed-point without a negative zero."""
    v = round(float(x), d)
    return f"{(v if v != 0 else 0.0):+.{d}f}"


def _ci(c: list[float], d: int = 4) -> str:
    return f"[{_s(c[0], d)}, {_s(c[1], d)}]"


def _ms(m: dict[str, Any]) -> str:
    return _f(m["mean"]) + (f" +/- {m['sd']:.4f}" if len(m["per_seed"]) > 1 else "")


def _pct(m: dict[str, Any]) -> str:
    return f"{100 * m['mean']:.1f} %"


def render(res: dict[str, Any], findings: str | None = None) -> str:
    ps = res["per_set"]
    fr = res["frozen"]
    L: list[str] = []
    a = L.append
    a("# S2E1 RESULTS — does ML catch up with representative training data? [SIM]")
    a("")
    a("> Simulation-only: synthetic records from the project's own generator; not validated on measured data; not for")
    a("> protection or control. Every number is a [SIM] result. Pre-registration: docs/S2E1_PREREG.md, configs/s2e1.yaml.")
    a("")
    a("## 1. Setup")
    dry = " **DRY RUN (stand-in splits, not the sealed sets)**" if res.get("dry_run") else ""
    a(f"- Commits: prereg `{str(res.get('prereg_commit'))[:7]}`, S2E1-freeze `{str(res.get('s2e1_freeze_commit'))[:7]}`, "
      f"evaluated at `{res['git_commit'][:12]}` (one run); Stage 1 freeze `{str(res.get('stage1_freeze_commit'))[:7]}`.{dry}")
    d = fr["data"]
    a(f"- Sealed sets ({ps['test2_shift']['n']} + {ps['test2_novel']['n']} records, 250/class): test2_shift (seed "
      f"{d['test2_shift']['split_seed']}, Stage 1 `shift` config incl. the unseen transient in "
      f"{ps['test2_shift']['breakdowns']['transient']['n_with']} records) and test2_novel (seed {d['test2_novel']['split_seed']}, "
      f"`wide` + an unseen interharmonic in {ps['test2_novel']['breakdowns'].get('interharmonic', {}).get('n_with', 0)} records).")
    for arm in ("GW", "GP"):
        x = fr["arms"][arm]
        c = x["config"]
        a(f"- {arm}: {x['n_features_used']} features ({x['feature_subset']}), selected on val_wide: {c['n_estimators']} stages x "
          f"{c['max_leaf_nodes']} leaves, lr {c['learning_rate']}, subsample {c['subsample']}; {x['models'][0]['size_bytes']} B; "
          f"val_wide macro-F1 {x['val_wide_macro_f1_mean']:.4f}; pre-event-only control {x['pre_event_only_control_mean']:.3f}.")
    a("- R, G0: Stage 1 frozen rules / GBM-embedded (seeds 0-2), unchanged. H0/HW: ML label if max probability >= "
      f"{fr['hybrid']['threshold']}, else the rules' label (per seed).")
    a("- Statistics: macro-F1 primary; ML mean +/- sd over 3 seeds; paired bootstrap (2,000 class-stratified resamples per set, "
      "shared by all arms) of the seed-mean difference; exact McNemar per seed; confusions = seed 0.")
    a("")
    a("## 2. Headline (macro-F1; false alarms = normal called an event; misses = event called normal)")
    a("")
    a("| arm | test2_shift | test2_novel | false alarms shift / novel | misses shift / novel | ML share (hybrids) |")
    a("|---|---|---|---|---|---|")
    for arm in ARMS:
        m1, m2 = ps["test2_shift"]["arms"][arm], ps["test2_novel"]["arms"][arm]
        share = ""
        if arm in ("H0", "HW"):
            share = " / ".join(f"{100 * sum(ps[s]['hybrid_ml_share'][arm]) / len(ps[s]['hybrid_ml_share'][arm]):.1f} %" for s in SETS)
        a(f"| {NAMES[arm]} | {_ms(m1['macro_f1'])} | {_ms(m2['macro_f1'])} | {_pct(m1['false_alarm_rate'])} / "
          f"{_pct(m2['false_alarm_rate'])} | {_pct(m1['miss_rate'])} / {_pct(m2['miss_rate'])} | {share} |")
    a("")
    a("## 3. Comparisons (paired bootstrap 95 % CI of the seed-mean macro-F1 difference; exact McNemar)")
    a("")
    a("| comparison | role | set | diff | 95 % CI | CI vs 0 | McNemar A-only / B-only correct, p (seeds) |")
    a("|---|---|---|---|---|---|---|")
    for c in ps["test2_shift"]["comparisons"]:
        for s in SETS:
            x = ps[s]["comparisons"][c]
            mc = "; ".join(f"{m['a_correct_b_wrong']}/{m['a_wrong_b_correct']}, p={m['p_value']:.2g}" for m in x["mcnemar_per_seed"])
            role = x["role"].split(":")[0]
            a(f"| {c} | {role} | {s} | {_s(x['diff_seed_mean'])} | {_ci(x['ci95_seed_mean'])} | "
              f"{x['reading']} | {mc} |")
    a("")
    a("## 4. Pre-registered decision rule and hypotheses")
    a("")
    a("| arm | role | CI (arm - R) test2_shift | CI test2_novel | verdict |")
    a("|---|---|---|---|---|")
    for arm, v in res["decision"].items():
        ci = v["ci95_by_set"]
        a(f"| {arm} | {v['role']} | {_ci(ci['test2_shift'])} | {_ci(ci['test2_novel'])} | **{v['verdict']}** |")
    a("")
    for h, v in res["hypotheses"].items():
        a(f"- {h} ({v['text']}): **{'supported' if v['supported'] else 'not supported'}**")
    if "survival_test2_novel" in res:
        a("")
        a("Unseen-nuisance survival on test2_novel (delta = macro-F1 with - without interharmonic; survives if CI lower >= -0.03):")
        a("")
        a("| arm | without | with | delta | 95 % CI | verdict |")
        a("|---|---|---|---|---|---|")
        for arm, v in res["survival_test2_novel"].items():
            a(f"| {arm} | {v['f1_without']:.4f} | {v['f1_with']:.4f} | {_s(v['delta'])} | {_ci(v['ci95'])} | {v['verdict']} |")
    a("")
    a("## 5. Per-class F1 and critical confusions (seed mean)")
    a("")
    a("| class | " + " | ".join(f"{arm} shift / novel" for arm in ("R", "G0", "GW", "GP", "HW")) + " |")
    a("|---|" + "---|" * 5)
    for c in CLASSES:
        a(f"| {c} | " + " | ".join(f"{ps['test2_shift']['arms'][arm]['per_class_f1_mean'][c]:.3f} / "
                                   f"{ps['test2_novel']['arms'][arm]['per_class_f1_mean'][c]:.3f}"
                                   for arm in ("R", "G0", "GW", "GP", "HW")) + " |")
    for k in ("phase_jump<->freq_event", "sag_*->normal", "sag_bal<->sag_unbal"):
        a(f"| {k} (count) | " + " | ".join(f"{ps['test2_shift']['arms'][arm]['critical_confusions_mean'][k]:.3g} / "
                                           f"{ps['test2_novel']['arms'][arm]['critical_confusions_mean'][k]:.3g}"
                                           for arm in ("R", "G0", "GW", "GP", "HW")) + " |")
    a("")
    a("## 6. Breakdowns (macro-F1 per bin unless stated; seed mean)")
    a("")
    a("| set | subset | n | " + " | ".join(ARMS) + " |")
    a("|---|---|---|" + "---|" * len(ARMS))
    for s in SETS:
        bd = ps[s]["breakdowns"]
        for b in bd["snr"]["bins"]:
            if bd["snr"]["R"][b]["n"]:
                a(f"| {s} | SNR {b} dB | {bd['snr']['R'][b]['n']} | " + " | ".join(_f(bd["snr"][arm][b]["value"]) for arm in ARMS) + " |")
    tr = ps["test2_shift"]["breakdowns"]["transient"]
    for k in ("without_transient", "with_transient"):
        a(f"| test2_shift | {k.replace('_', ' ')} | {tr['R'][k]['n']} | " + " | ".join(_f(tr[arm][k]['value']) for arm in ARMS) + " |")
    bd = ps["test2_novel"]["breakdowns"]
    if "interharmonic" in bd:
        for k in ("without_interharmonic", "with_interharmonic"):
            a(f"| test2_novel | {k.replace('_', ' ')} | {bd['interharmonic']['R'][k]['n']} | "
              + " | ".join(_f(bd["interharmonic"][arm][k]["value"]) for arm in ARMS) + " |")
        for key, lab in (("interharmonic_beat_primary", "accuracy, beat"), ("interharmonic_beat_secondary", "accuracy, beat")):
            x = bd[key]
            for b in x["bins"]:
                if x["R"][b]["n"]:
                    a(f"| test2_novel | {lab} {b.replace('beat_', '').replace('lt_', '< ').replace('ge_', '>= ')} Hz | {x['R'][b]['n']} | "
                      + " | ".join(_f(x[arm][b]["value"]) for arm in ARMS) + " |")
        err = bd["interharmonic_errors_primary"]
        a("")
        a("Errors on interharmonic records (seed 0, true->predicted: count): " + "; ".join(
            f"{arm}: " + (", ".join(f"{k} {v}" for k, v in sorted(err[arm].items(), key=lambda kv: -kv[1])[:4]) or "none")
            for arm in ("R", "G0", "GW", "GP")))
    a("")
    if findings:
        a(findings.strip())
        a("")
    a("Figures: reports/figures/s2e1_confusions.png, s2e1_f1_by_snr.png, s2e1_novel_breakdown.png; all numbers: "
      "reports/s2e1_results.json.")
    return "\n".join(L) + "\n"
