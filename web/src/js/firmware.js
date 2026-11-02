/* Firmware page: SIL table, ISR cost, stream events, C28x sizes, validation ladder (from window.GM_SITE). */
"use strict";
document.addEventListener("DOMContentLoaded", () => {
  const { $, fmt, int, sci } = GMU;
  const sil = GM_SITE.sil;
  let h = `<div class="table-wrap"><table><caption>C core vs Python reference (reports/m6_sil.json)</caption><thead><tr><th scope="col">set</th><th scope="col" class="num">records</th>
    <th scope="col" class="num">features within tolerance</th><th scope="col" class="num">values outside (gated / ungated)</th><th scope="col" class="num">rule labels equal</th>
    <th scope="col">GBM on same C features</th><th scope="col" class="num">GBM end to end</th></tr></thead><tbody>`;
  for (const [s, v] of Object.entries(sil.splits)) {
    h += `<tr><td>${s}${v.subset !== "all" ? ` <span class="small muted">(${v.subset})</span>` : ""}</td><td class="num">${int(v.n)}</td>
      <td class="num">${v.features_ok}/${v.n_features}</td><td class="num">${v.outside} / ${v.outside_ungated}</td>
      <td class="num ${v.rules_agree === v.n ? "ok" : "bad"}">${int(v.rules_agree)}/${int(v.n)}</td>
      <td>${v.gbm_a_bit_exact ? "labels bit-exact" : "MISMATCH"}, max |&Delta;p| ${sci(v.gbm_a_max_dp, 1)}</td><td class="num">${int(v.gbm_b_agree)}/${int(v.n)}</td></tr>`;
  }
  $("#sil").innerHTML = h + "</tbody></table></div>" +
    `<p class="small">Portability checks: static C28x rules ${sil.static_checks_pass ? '<span class="ok">pass</span>' : '<span class="bad">fail</span>'};
    ${Object.entries(sil.portability).map(([k, v]) => `${k === "m32" ? "32-bit (ILP32)" : "16-bit int"} build <span class="${v.status === "pass" ? "ok" : "bad"}">${v.status}</span>`).join("; ")}.</p>`;

  const P = GM_SITE.port;
  let c = `<div class="table-wrap"><table><caption>Interrupt cost of gm_port_push per input sample (all three channels)</caption><thead><tr><th scope="col" class="num">input rate</th>
    <th scope="col" class="num">FIR taps L</th><th scope="col" class="num">decimation D</th><th scope="col" class="num">MACs per sample</th><th scope="col" class="num">M MAC/s</th>
    <th scope="col" class="num">stopband</th><th scope="col" class="num">decimator vs float64</th></tr></thead><tbody>`;
  P.isr_cost.forEach((r, i) => {
    const d = P.decimator_check[i];
    c += `<tr><td class="num">${int(r.f_in_hz / 1000)} kHz</td><td class="num">${r.L}</td><td class="num">${r.D}</td><td class="num">${r.total_macs_per_input_sample}</td>
      <td class="num">${fmt(r.macs_per_s / 1e6, 2)}</td><td class="num">${fmt(r.stopband_atten_db, 1)} dB</td><td class="num">&le; ${fmt(d.max_abs_diff_lsb, 2)} LSB</td></tr>`;
  });
  $("#isr").innerHTML = c + "</tbody></table></div>";

  let e = `<details><summary>Per-event results of the stream test</summary><div class="table-wrap"><table><thead><tr><th scope="col">true class</th><th scope="col" class="num">onset (s)</th>
    <th scope="col">trigger</th><th scope="col" class="num">latency (ms)</th><th scope="col" class="num">onset in record (s)</th><th scope="col">rules</th><th scope="col">GBM (advisory)</th><th scope="col" class="num">extra captures</th></tr></thead><tbody>`;
  for (const r of P.per_event) {
    e += `<tr><td>${r.label}</td><td class="num">${fmt(r.onset_s, 2)}</td><td>${r.triggered ? r.cause.join(", ") : "none"}</td>
      <td class="num">${r.triggered ? fmt(1000 * r.latency_s, 0) : "—"}</td><td class="num">${r.triggered ? fmt(r.onset_in_record_s, 3) : "—"}</td>
      <td class="${r.triggered ? (r.rule_correct ? "ok" : "bad") : ""}">${r.triggered ? r.rule_label : "—"}</td><td>${r.triggered ? `${r.gbm_label} (${fmt(r.gbm_max_proba, 2)})` : "—"}</td>
      <td class="num">${r.n_secondary_captures}</td></tr>`;
  }
  $("#events").innerHTML = e + "</tbody></table></div></details>";

  const C = GM_SITE.c28x;
  if (C) {
    const on = C.variants.gbm_on, off = C.variants.gbm_off;
    let t = `<p>${C.all_compiled ? '<span class="ok">All files compile</span>' : '<span class="bad">Compile errors</span>'} with ${C.compiler}
      (<code>${C.flags.join(" ")}</code>), GBM on and off. Compile-only object sizes in 16-bit words (the linked image is smaller; the map file comes with L3):</p>
      <div class="table-wrap"><table><caption>C28x object sizes (reports/m7_c28x_size.json)</caption><thead><tr><th scope="col">file</th><th scope="col" class="num">code</th><th scope="col" class="num">const</th><th scope="col" class="num">code (GBM off)</th></tr></thead><tbody>`;
    for (const [f, v] of Object.entries(on.files)) t += `<tr><td>${f}</td><td class="num">${int(v.code_words)}</td><td class="num">${int(v.const_words)}</td><td class="num">${int(off.files[f] ? off.files[f].code_words : null)}</td></tr>`;
    t += `<tr><th scope="row">total</th><td class="num"><strong>${int(on.total_words.code)}</strong></td><td class="num"><strong>${int(on.total_words.const)}</strong></td><td class="num"><strong>${int(off.total_words.code)}</strong></td></tr>`;
    $("#c28x").innerHTML = t + "</tbody></table></div>";
  } else {
    $("#c28x").innerHTML = "<p>Script ready, run locally (docs/C2000_BUILD.md).</p>";
  }

  let l = `<div class="table-wrap"><table><caption>Validation ladder (docs/VALIDATION.md)</caption><thead><tr><th scope="col">level</th><th scope="col">what</th><th scope="col">evidence</th><th scope="col">status</th></tr></thead><tbody>`;
  for (const r of GM_SITE.ladder) l += `<tr><td><strong>${r.level}</strong></td><td>${r.what}</td><td>${r.evidence}</td><td class="small">${r.status}</td></tr>`;
  $("#ladder").innerHTML = l + "</tbody></table></div>";
});
