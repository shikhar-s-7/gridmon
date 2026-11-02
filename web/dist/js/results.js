/* Results page: tables and charts rendered from window.GM_SITE (built from reports/results.json). */
"use strict";
document.addEventListener("DOMContentLoaded", () => {
  const { $, fmt, pct, signed, ci, int, METHOD_LABELS, METHOD_VARS } = GMU;
  const R = GM_SITE.results, S = R.splits, methods = R.methods, classes = GM_SITE.classes;
  const pm = (m) => (m.sd && m.sd > 0 ? `${fmt(m.mean, 4)} &plusmn; ${fmt(m.sd, 4)}` : fmt(m.mean, 4));

  // headline
  let h = `<div class="table-wrap"><table><caption>Macro-F1, false-alarm and miss rates on the sealed test sets</caption><thead><tr>
    <th scope="col">method</th><th scope="col" class="num">val (selection)</th><th scope="col" class="num">test_iid</th><th scope="col" class="num">test_shift</th>
    <th scope="col" class="num">iid &rarr; shift (95 % CI)</th><th scope="col" class="num">false alarms iid / shift</th><th scope="col" class="num">misses iid / shift</th></tr></thead><tbody>`;
  for (const m of methods) {
    const a = S.test_iid.methods[m], b = S.test_shift.methods[m], d = R.iid_to_shift[m];
    h += `<tr><th scope="row"><span class="swatch" style="background:${METHOD_VARS[m]}"></span>${METHOD_LABELS[m]}</th>
      <td class="num">${fmt(R.val_macro_f1[m], 4)}</td><td class="num">${pm(a.macro_f1)}</td><td class="num">${pm(b.macro_f1)}</td>
      <td class="num">${signed(d.change, 3)} ${ci(d.ci95, 3)}</td>
      <td class="num">${pct(a.false_alarm_rate.mean)} / ${pct(b.false_alarm_rate.mean)}</td><td class="num">${pct(a.miss_rate.mean)} / ${pct(b.miss_rate.mean)}</td></tr>`;
  }
  $("#headline").innerHTML = h + "</tbody></table></div>";

  // comparisons
  const names = { gbm_embedded_minus_rules: "GBM-embedded − rules", gbm_unconstrained_minus_rules: "GBM-unconstrained − rules",
    gbm_unconstrained_minus_gbm_embedded: "GBM-unconstrained − GBM-embedded (seed 0)" };
  let c = `<div class="table-wrap"><table><caption>Differences in macro-F1 (positive = first method better)</caption><thead><tr><th scope="col">comparison</th>
    <th scope="col">split</th><th scope="col" class="num">difference</th><th scope="col" class="num">95 % CI</th><th scope="col">significant?</th>
    <th scope="col" class="num">McNemar discordant (per seed)</th><th scope="col" class="num">p (per seed)</th></tr></thead><tbody>`;
  for (const [k, label] of Object.entries(names)) {
    for (const sp of ["test_iid", "test_shift"]) {
      const x = S[sp].comparisons[k];
      c += `<tr><td>${label}</td><td>${sp}</td><td class="num">${signed(x.diff, 4)}</td><td class="num">${ci(x.ci95, 4)}</td>
        <td class="${x.excludes_zero ? "bad" : ""}">${x.excludes_zero ? (x.diff < 0 ? "yes: first is worse" : "yes: first is better") : "no"}</td>
        <td class="num">${x.mcnemar_counts.map((p) => p.join(" vs ")).join("; ")}</td><td class="num">${x.mcnemar_p.map((p) => p < 1e-4 ? p.toExponential(1) : p.toFixed(3)).join("; ")}</td></tr>`;
    }
  }
  $("#comparisons").innerHTML = c + "</tbody></table></div>";

  // confusion matrices
  const drawCM = () => {
    const m = $("#cm-method").value;
    GMC.matrix($("#cm-iid"), S.test_iid.methods[m].confusion.matrix, classes, `test_iid — ${METHOD_LABELS[m]}`);
    GMC.matrix($("#cm-shift"), S.test_shift.methods[m].confusion.matrix, classes, `test_shift — ${METHOD_LABELS[m]}`);
  };
  $("#cm-method").addEventListener("change", drawCM);
  drawCM();

  // SNR (pooled over both test sets)
  const snr = R.snr_by_bin_pooled;
  const mid = snr.bins.map((b) => { const [lo, hi] = b.split("-").map(Number); return (lo + hi) / 2; });
  GMC.line($("#snr-chart"), methods.map((m) => ({ name: METHOD_LABELS[m], x: mid, y: snr[m].f1, color: METHOD_VARS[m], width: 2 })),
    { title: "Macro-F1 by SNR bin (test_iid and test_shift pooled)", xlabel: "SNR (dB, bin centre)", ylabel: "macro-F1", height: 240,
      ymax: 1, desc: "Line chart: macro-F1 against SNR for the three methods; the GBM models drop below 0.8 at 25-30 dB while the rules stay near 0.99." });
  let st = `<div class="table-wrap"><table><caption>Macro-F1 by SNR bin (table view)</caption><thead><tr><th scope="col">method</th>${snr.bins.map((b) => `<th scope="col" class="num">${b} dB</th>`).join("")}</tr></thead><tbody>`;
  for (const m of methods) st += `<tr><th scope="row">${METHOD_LABELS[m]}</th>${snr[m].f1.map((v, i) => `<td class="num">${fmt(v, 3)} <span class="muted small">(n ${int(snr[m].n[i])})</span></td>`).join("")}</tr>`;
  st += `</tbody></table></div>`;
  $("#snr-table").innerHTML = st;

  // severity (test_shift, recall per bin; rules vs GBM-embedded)
  const sev = S.test_shift.breakdowns.severity_recall;
  let sv = `<div class="table-wrap"><table><caption>Recall by event severity on test_shift (seed 0 for ML)</caption><thead><tr><th scope="col">class</th><th scope="col">severity variable</th><th scope="col">bin</th>
    ${methods.map((m) => `<th scope="col" class="num">${METHOD_LABELS[m]}</th>`).join("")}</tr></thead><tbody>`;
  for (const [cls, v] of Object.entries(sev)) {
    v.bins.forEach((b, i) => {
      sv += `<tr>${i === 0 ? `<td rowspan="${v.bins.length}">${cls}</td><td rowspan="${v.bins.length}">${v.variable}</td>` : ""}<td>${b}</td>
        ${methods.map((m) => { const x = v[m] && v[m][b]; return `<td class="num">${x && x.value !== null ? fmt(x.value, 3) : "—"} <span class="muted small">(n ${x ? x.n : 0})</span></td>`; }).join("")}</tr>`;
    });
  }
  $("#severity").innerHTML = sv + "</tbody></table></div>";
  const tr = S.test_shift.breakdowns.shift_transient;
  let tt = `<div class="table-wrap"><table><caption>test_shift with and without the unseen transient (macro-F1; ${int(tr.n_with)} records with it)</caption><thead><tr><th scope="col">method</th><th scope="col" class="num">without</th><th scope="col" class="num">with</th></tr></thead><tbody>`;
  for (const m of methods) tt += `<tr><th scope="row">${METHOD_LABELS[m]}</th><td class="num">${fmt(tr[m].without_transient.value, 3)}</td><td class="num">${fmt(tr[m].with_transient.value, 3)}</td></tr>`;
  $("#transient").innerHTML = tt + "</tbody></table></div>";

  // naive PLL
  let p = `<div class="table-wrap"><table><caption>Records flagged as frequency events by the naive PLL rule</caption><thead><tr><th scope="col">true class</th><th scope="col" class="num">test_iid</th><th scope="col" class="num">test_shift</th></tr></thead><tbody>`;
  for (const cl of classes) {
    const a = S.test_iid.naive_pll.by_class[cl], b = S.test_shift.naive_pll.by_class[cl];
    p += `<tr><td>${cl}</td><td class="num">${pct(a.rate)} (${a.flagged}/${a.n})</td><td class="num">${pct(b.rate)} (${b.flagged}/${b.n})</td></tr>`;
  }
  $("#pll").innerHTML = p + "</tbody></table></div>";

  // size vs F1
  const sz = GM_SITE.size_vs_f1;
  const pts = sz.grid.map((g) => ({ x: g.bytes, y: g.val_f1, color: g.bytes <= sz.budget_bytes ? "var(--series-2)" : "var(--text-3)", r: 4,
    label: `${g.stages} stages × ${g.leaves} leaves, lr ${g.lr}, subsample ${g.subsample}: ${int(g.bytes)} B, val F1 ${fmt(g.val_f1, 4)}` }));
  pts.push({ x: sz.selected.bytes, y: sz.selected.val_f1, color: "var(--series-1)", r: 7, label: `selected GBM-embedded: ${int(sz.selected.bytes)} B, val F1 ${fmt(sz.selected.val_f1, 4)}` });
  pts.push({ x: sz.unconstrained_bytes, y: sz.unconstrained_val_f1, color: "var(--series-3)", r: 7, label: `GBM-unconstrained: ~${int(sz.unconstrained_bytes)} B, val F1 ${fmt(sz.unconstrained_val_f1, 4)}` });
  GMC.scatter($("#size-chart"), pts, { xlog: true, xlabel: "exported model size (bytes, log scale)", ylabel: "validation macro-F1",
    vlines: [{ x: sz.budget_bytes, label: "64 KB budget" }], fmtX: (v) => (v >= 1000 ? (v / 1000) + "k" : v),
    desc: "Scatter plot of model size against validation macro-F1 for every grid configuration; all are near 1.0." });
  const leg = GMC.legend([{ name: "grid configuration within budget", color: "var(--series-2)" }, { name: "over budget", color: "var(--text-3)" },
    { name: "selected (smallest tied)", color: "var(--series-1)" }, { name: "GBM-unconstrained reference", color: "var(--series-3)" }]);
  $("#size-chart").prepend(leg);
});
