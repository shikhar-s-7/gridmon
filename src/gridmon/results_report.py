"""Render reports/RESULTS.md from results.json (tables) plus the hand-written findings file
(reports/results_findings.md: Findings, Limitations, What would change the conclusion)."""

from __future__ import annotations

from typing import Any

from . import CLASSES

M = ("rules", "gbm_embedded", "gbm_unconstrained")
MN = {"rules": "rules", "gbm_embedded": "GBM-embedded", "gbm_unconstrained": "GBM-unconstrained"}
AB = {"normal": "nor", "sag_bal": "sgB", "sag_unbal": "sgU", "swell": "swl", "phase_jump": "pjp",
      "freq_event": "frq", "oscillation": "osc"}
S = ("test_iid", "test_shift")


def _f(x: float | None, nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _ms(d: dict[str, Any], nd: int = 3) -> str:
    return f"{d['mean']:.{nd}f}" + (f" +/- {d['sd']:.{nd}f}" if d.get("sd") else "")


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def _conf_block(res: dict[str, Any], split: str) -> list[str]:
    mats = [res["per_split"][split]["methods"][m]["primary"]["confusion"]["matrix"] for m in M]
    w = 4
    head = "true\\pred " + " | ".join("".join(f"{AB[c]:>{w}s}" for c in CLASSES) for _ in M)
    title = "          " + " | ".join(f"{MN[m] + (' (seed 0)' if m != 'rules' else ''):^{w * 7}s}" for m in M)
    lines = ["```", f"{split}", title, head]
    for i, c in enumerate(CLASSES):
        lines.append(f"{AB[c]:<10s}" + " | ".join("".join(f"{v:>{w}d}" for v in mat[i]) for mat in mats))
    lines.append("```")
    return lines


def render_results_md(res: dict[str, Any], findings: str | None) -> str:
    fz = res["frozen"]
    emb, unc = fz["gbm_embedded"], fz["gbm_unconstrained"]
    ps = res["per_split"]
    ec = res["embedded_cost"]
    c = emb["config"]
    L: list[str] = []
    L += ["# RESULTS — gridmon Stage 1, sealed test evaluation (M4) [SIM]", "",
          "> Simulation-only: synthetic records from the project's own generator; not validated on measured data;",
          "> not for protection or control. Every number below is a [SIM] result.", ""]
    tag = " (DRY RUN on val)" if res.get("dry_run") else (" (FAST subset)" if res.get("fast") else "")
    L += [f"## 1. Setup{tag}",
          f"- Test sets (sealed until this run): test_iid (seed 3003, iid nuisances) and test_shift (seed 4004: SNR 25-35 dB,",
          f"  f_pre 49.5-50.5 Hz, more harmonics/unbalance, +/-2 % gain/offset, unseen damped 300-900 Hz transient in "
          f"{ps['test_shift']['breakdowns']['shift_transient']['n_with']} of {ps['test_shift']['n']} records); "
          f"250 records per class each; data generated at commit {ps['test_iid']['data_meta']['git_commit'][:7]}.",
          f"- Frozen methods (M4-freeze commit {str(res['freeze_commit'])[:7]}; evaluated at {res['git_commit'][:7]}; one run):",
          f"  rules with the TRAIN-tuned thresholds (D-036: {', '.join(f'{k} {v}' for k, v in fz['rules']['thresholds'].items())});",
          f"  GBM-embedded = sklearn GradientBoostingClassifier, {c['n_estimators']} stages x {c['max_leaf_nodes']} leaves, lr "
          f"{c['learning_rate']}, subsample {c['subsample']}, {emb['models'][0]['size_bytes']} B exported, seeds {emb['seeds']}, "
          "run through the float32 reference walker on the exported JSON (= the C model);",
          f"  GBM-unconstrained = HistGradientBoostingClassifier (max_leaf_nodes {unc['config']['max_leaf_nodes']}, lr "
          f"{unc['config']['learning_rate']}, early stopping), seeds {unc['seeds']}; reference for the cost of the 64 KB budget.",
          "- All methods use the same 37 M2 features computed from the int16 waveforms only; labels and manifests are",
          "  used for scoring and breakdowns only. Selection used TRAIN/VAL only (M2, M3).",
          f"- Statistics: macro-F1 primary; ML = mean +/- sd over 3 seeds; confusions and figures use seed 0 (the exported",
          f"  model); paired bootstrap {fz['evaluation_plan']['bootstrap_resamples']} resamples stratified by class (seed-mean"
          " difference); exact McNemar on per-record correctness (per seed).", ""]
    # headline
    L += ["## 2. Headline results (macro-F1; ML mean +/- sd over seeds)", "",
          "| method | test_iid | test_shift | iid -> shift (95 % CI) | false alarms iid / shift | misses iid / shift |",
          "|---|---|---|---|---|---|"]
    for m in M:
        a, b = ps["test_iid"]["methods"][m], ps["test_shift"]["methods"][m]
        d = res["iid_to_shift"][m]
        L.append(f"| {MN[m]} | {_ms(a['macro_f1'], 4)} | {_ms(b['macro_f1'], 4)} | {d['change']:+.4f} [{d['ci95'][0]:+.4f}, "
                 f"{d['ci95'][1]:+.4f}] | {_pct(a['false_alarm_rate']['mean'])} / {_pct(b['false_alarm_rate']['mean'])} | "
                 f"{_pct(a['miss_rate']['mean'])} / {_pct(b['miss_rate']['mean'])} |")
    L += ["", "## 3. Method differences (paired bootstrap 95 % CI of the macro-F1 difference; exact McNemar)", "",
          "| comparison | split | diff | 95 % CI | McNemar A-only / B-only correct, p (seeds 0/1/2) | reading |", "|---|---|---|---|---|---|"]
    for key, name in (("gbm_embedded_minus_rules", "GBM-emb - rules"), ("gbm_unconstrained_minus_rules", "GBM-unc - rules"),
                      ("gbm_unconstrained_minus_gbm_embedded", "GBM-unc - GBM-emb (seed 0)")):
        for s in S:
            cc = ps[s]["comparisons"][key]
            bs = cc["bootstrap"]
            mc = "; ".join(f"{x['a_correct_b_wrong']}/{x['a_wrong_b_correct']}, p={x['p_value']:.2g}" for x in cc["mcnemar_per_seed"])
            lo, hi = bs["ci95_seed_mean"]
            verdict = "A better" if lo > 0 else ("B better" if hi < 0 else "no significant difference")
            L.append(f"| {name} | {s} | {bs['diff_seed_mean']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {mc} | {verdict} |")
    L += ["", "## 4. Confusion matrices (rows true, columns predicted)", ""]
    for s in S:
        L += _conf_block(res, s)
    L += ["", "## 5. Per-class F1 and critical confusions (ML: seed mean)", "",
          "| class | " + " | ".join(f"{MN[m]} {s[5:]}" for m in M for s in S) + " |", "|---" * (1 + 2 * len(M)) + "|"]
    for cl in CLASSES:
        L.append(f"| {cl} | " + " | ".join(_f(ps[s]["methods"][m]["per_class_mean"][cl]["f1"], 4) for m in M for s in S) + " |")
    L += ["", "| critical confusion (count) | " + " | ".join(f"{MN[m]} {s[5:]}" for m in M for s in S) + " |",
          "|---" * (1 + 2 * len(M)) + "|"]
    for k in ("phase_jump->freq_event", "freq_event->phase_jump", "sag_*->normal", "sag_bal->sag_unbal", "sag_unbal->sag_bal"):
        L.append(f"| {k} | " + " | ".join(f"{ps[s]['methods'][m]['critical_confusions_mean'][k]:.3g}" for m in M for s in S) + " |")
    # breakdowns
    L += ["", "## 6. Breakdowns", "", "SNR (macro-F1 per bin; bins below 35 dB exist only in test_shift, above only in test_iid):", "",
          "| SNR bin (dB) | n | " + " | ".join(MN[m] for m in M) + " |", "|---|---|---|---|---|"]
    sb = res["snr_by_bin_pooled"]
    for i, g in enumerate(sb["bins"]):
        L.append(f"| {g} | {sb['rules']['n'][i]} | " + " | ".join(_f(sb[m]["f1"][i], 4) for m in M) + " |")
    L += ["", "Severity (recall per bin, ML seed mean; iid / shift). Listed: bins where some method is below 0.99 on",
          "either set; every unlisted bin (bins: sag depth, swell height, abs(dphi), abs(ROCOF), oscillation depth)",
          "is >= 0.99 for every method on both sets (all bins in results.json):", "",
          "| class | variable | bin | n iid / shift | " + " | ".join(MN[m] for m in M) + " |", "|---|---|---|---|---|---|---|"]
    for cl in CLASSES[1:]:
        si, ss = ps["test_iid"]["breakdowns"]["severity_recall"][cl], ps["test_shift"]["breakdowns"]["severity_recall"][cl]
        for g in si["bins"]:
            def v(src, m):
                x = src[m][g]["value"]
                return "-" if x is None else _f(x, 2)
            vals = [src[m][g]["value"] for src in (si, ss) for m in M if src[m][g]["value"] is not None]
            if vals and min(vals) >= 0.99:
                continue
            L.append(f"| {cl} | {si['variable']} | {g} | {si['rules'][g]['n']} / {ss['rules'][g]['n']} | "
                     + " | ".join(f"{v(si, m)} / {v(ss, m)}" for m in M) + " |")
    tb = ps["test_shift"]["breakdowns"]["shift_transient"]
    L += ["", "Unseen shift transient (test_shift, macro-F1):", "", "| subset | n | " + " | ".join(MN[m] for m in M) + " |",
          "|---|---|---|---|---|"]
    for g in ("without_transient", "with_transient"):
        L.append(f"| {g.replace('_', ' ')} | {tb['rules'][g]['n']} | " + " | ".join(_f(tb[m][g]["value"], 4) for m in M) + " |")
    L += ["", "## 7. Naive-PLL diagnostic (flag 'frequency event' if |f_pll - f_ref| > 1 Hz at any instant)", "",
          "| class | test_iid flagged | test_shift flagged |", "|---|---|---|"]
    for cl in CLASSES:
        a, b = ps["test_iid"]["naive_pll"]["by_class"][cl], ps["test_shift"]["naive_pll"]["by_class"][cl]
        L.append(f"| {cl} | {a['flagged']}/{a['n']} ({_pct(a['rate'])}) | {b['flagged']}/{b['n']} ({_pct(b['rate'])}) |")
    ji, js = ps["test_iid"]["naive_pll"]["by_jump_deg"], ps["test_shift"]["naive_pll"]["by_jump_deg"]
    L.append("| phase_jump by abs(dphi) | " + ", ".join(f"{g} deg {ji[g]['flagged']}/{ji[g]['n']}" for g in ji) + " | "
             + ", ".join(f"{g} deg {js[g]['flagged']}/{js[g]['n']}" for g in js) + " |")
    t = ec["python_timing_ms_per_record"]
    L += ["", "## 8. Embedded cost (per 1 s record)", "",
          "| block | operations | RAM | const/flash |", "|---|---|---|---|",
          f"| DSP + 37 features (shared by all methods; M2 estimate) | {ec['dsp_features_ops_per_record'] / 1e6:.2f} M ops | "
          f"{ec['dsp_ram_kb']} KB (+30 KB int16 record) | {ec['dsp_const_kb']} KB |",
          f"| rules | {ec['rules_comparisons_per_record']} | negligible | thresholds |",
          f"| GBM-embedded | {ec['gbm_comparisons_per_record_mean']['test_iid']:.0f} comparisons mean (worst {ec['gbm_comparisons_worst_case']}),"
          f" {emb['models'][0]['n_trees']} adds, 7 expf | {ec['gbm_ram_bytes']} B | {ec['gbm_model_bytes'] / 1024:.1f} KB |",
          f"| GBM-unconstrained (reference, not exported) | - | - | ~{ec['gbm_unconstrained_model_bytes_equivalent'] / 1024:.0f} KB equivalent |",
          "", f"Python timings (indicative, this VM): features {t.get('features_test_iid_ms_per_record', 0):.2f} ms/record; rules "
          f"{t.get('rules_ms_per_record_test_iid', 0):.4f}; GBM-embedded float32 walker {t.get('gbm_embedded_walker_ms_per_record_test_iid', 0):.3f}; "
          f"HGB {t.get('gbm_unconstrained_ms_per_record_test_iid', 0):.3f} ms/record. C2000 cycles need Stage 2 PIL.", ""]
    if findings:
        L += [findings.strip(), ""]
    else:
        L += ["## 9. Findings", "", "(to be written)", ""]
    L += ["Figures: reports/figures/m4_confusions.png, m4_f1_by_snr.png, m4_shift_degradation.png, m4_disagreements.png; "
          "all numbers: reports/results.json."]
    return "\n".join(L) + "\n"
