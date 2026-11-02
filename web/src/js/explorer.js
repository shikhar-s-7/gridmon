/* Explorer page: record list with filters, stored reference results, plots, rule trace, WASM re-classification. */
"use strict";
const GMShow = (() => {
  const { el, fmt, b64f32, CLASS_LABELS } = GMU;
  const tol = () => Object.fromEntries(GM_SITE.features.map((f) => [f.name, f.tol]));
  /* rule decision trace table */
  function ruleTable(trace) {
    let h = `<div class="table-wrap"><table><caption>Rule decision trace (priority order; the first rule whose tests all pass fires)</caption>
      <thead><tr><th scope="col">rule</th><th scope="col">tests (value vs threshold)</th><th scope="col">result</th></tr></thead><tbody>`;
    for (const s of trace.steps) {
      const tests = s.tests.map(([n, v, op, t], i) => `<span class="${s.results[i] ? "ok" : "muted"}">${n} = ${Number.isFinite(v) ? (+v.toPrecision(4)) : "n/a"} ${op.replace("<", "&lt;").replace(">", "&gt;")} ${t}</span>`).join("<br>");
      h += `<tr${s.fired ? ' style="outline:2px solid var(--accent)"' : ""}><td>${s.rule}</td><td>${tests}</td><td>${s.fired ? '<strong class="ok">fires</strong>' : s.pass ? "passes (lower priority)" : "no"}</td></tr>`;
    }
    h += `<tr><td>6 otherwise</td><td>—</td><td>${trace.label === "normal" ? '<strong class="ok">normal</strong>' : "—"}</td></tr>`;
    return h + "</tbody></table></div>";
  }
  function probaBars(p, highlight) {
    const order = GM_SITE.classes.map((c, i) => [c, p[i]]).sort((a, b) => b[1] - a[1]);
    let h = `<div class="bars" role="table" aria-label="GBM class probabilities">`;
    for (const [c, v] of order) {
      h += `<span role="cell">${c}</span><span role="cell"><span class="bar${c === highlight ? " top" : ""}" style="display:block;width:${(100 * v).toFixed(1)}%"></span></span><span role="cell" class="mono">${fmt(v, 3)}</span>`;
    }
    return h + "</div>";
  }
  /* plots from a traces object: {x (Int16Array planar), lsb, urms[3], v1mag, v2mag, J_deg, f_dft, f_pll, r_mag} */
  function plots(container, t, onset) {
    container.innerHTML = "";
    const tm = Float32Array.from({ length: 1000 }, (_, m) => (5 * m + 4) / 5000);
    const vl = Number.isFinite(onset) ? [{ x: onset, label: "onset est." }] : [];
    const mk = () => { const d = el("div", { class: "card" }); container.append(d); return d; };
    if (t.x) {
      const n = 5000, step = 2, xs = [], ys = [[], [], []];
      for (let i = 0; i < n; i += step) { xs.push(i / 5000); for (let c = 0; c < 3; c++) ys[c].push(t.x[c * n + i] * t.lsb); }
      GMC.line(mk(), ["v_ab", "v_bc", "v_ca"].map((nm, c) => ({ name: nm, x: xs, y: ys[c], color: `var(--series-${c + 1})`, width: 1 })),
        { title: "Line-to-line voltages (pu of nominal LL peak)", xlabel: "time (s)", height: 200, vlines: vl, desc: "Three-phase waveform of the record." });
    }
    const tk = Float32Array.from({ length: 99 }, (_, k) => (50 * k + 50) / 5000);
    GMC.line(mk(), ["ab", "bc", "ca"].map((nm, c) => ({ name: `Urms(1/2) ${nm}`, x: tk, y: t.urms[c], color: `var(--series-${c + 1})` })),
      { title: "Half-cycle RMS Urms(1/2) (pu)", xlabel: "time (s)", height: 180, hlines: [{ y: 0.9, label: "0.90 dip" }, { y: 1.1, label: "1.10 swell" }], vlines: vl });
    GMC.line(mk(), [{ name: "|V1|", x: tm, y: t.v1mag, color: "var(--series-1)" }, { name: "|V2|", x: tm, y: t.v2mag, color: "var(--series-2)" }],
      { title: "Positive- and negative-sequence magnitude (pu)", xlabel: "time (s)", height: 180, vlines: vl });
    GMC.line(mk(), [{ name: "J", x: tm, y: t.J_deg, color: "var(--series-1)" }],
      { title: "Phase-jump metric J (deg)", xlabel: "time (s)", height: 160, hlines: [{ y: 5, label: "5 deg rule" }, { y: -5 }], vlines: vl });
    GMC.line(mk(), [{ name: "f_dft", x: tm, y: t.f_dft, color: "var(--series-1)" }, { name: "f_pll", x: tm, y: t.f_pll, color: "var(--series-2)" }],
      { title: "Frequency: DFT (used by the rules) and DSOGI-PLL (Hz)", xlabel: "time (s)", height: 180, vlines: vl });
    const tp = Float32Array.from({ length: 100 }, (_, p) => (5 * (10 * p + 5) + 4) / 5000);
    GMC.line(mk(), [{ name: "detrended |V1| envelope (%)", x: tp, y: Float32Array.from(t.r_mag, (v) => 100 * v), color: "var(--series-1)" }],
      { title: "Envelope of |V1| minus its running median (%)", xlabel: "time (s)", height: 160, vlines: vl });
  }
  function featureTable(feats, ref) {
    const t = tol();
    let h = `<details><summary>All 37 features${ref ? " (stored Python reference vs C core in this browser)" : ""}</summary><div class="table-wrap"><table><thead><tr><th scope="col">feature</th><th scope="col">unit</th><th scope="col" class="num">value</th>${ref ? '<th scope="col" class="num">browser (C/WASM)</th><th scope="col" class="num">|diff| / tolerance</th>' : ""}</tr></thead><tbody>`;
    GM_SITE.features.forEach((f, i) => {
      const a = feats[i];
      h += `<tr><td title="${f.def}">${f.name}</td><td>${f.unit}</td><td class="num mono">${a === null || !Number.isFinite(a) ? "n/a" : (+a.toPrecision(6))}</td>`;
      if (ref) {
        const b = ref[i]; const d = Math.abs(a - b);
        h += `<td class="num mono">${b === null || !Number.isFinite(b) ? "n/a" : (+b.toPrecision(6))}</td><td class="num">${Number.isFinite(d) && t[f.name] > 0 ? (d / t[f.name]).toFixed(2) : (Number.isFinite(d) ? (d === 0 ? "0" : "argmax-type") : "—")}</td>`;
      }
      h += "</tr>";
    });
    return h + "</tbody></table></div></details>";
  }
  return { ruleTable, probaBars, plots, featureTable, CLASS_LABELS, b64f32 };
})();

document.addEventListener("DOMContentLoaded", () => {
  if (!document.getElementById("rec-list")) return;
  const { $, el, fmt, b64f32, b64i16 } = GMU;
  const idx = window.GM_EXPLORER || [];
  const fc = $("#f-class");
  for (const c of GM_SITE.classes) fc.append(el("option", { value: c }, c));
  let current = null;
  const list = $("#rec-list");
  function filtered() {
    const c = fc.value, s = $("#f-split").value, a = $("#f-agree").value;
    return idx.filter((r) => (!c || r.true === c) && (!s || r.split === s) &&
      (!a || (a === "agree" && r.rule_label === r.gbm_label) || (a === "disagree" && r.rule_label !== r.gbm_label) ||
        (a === "rules_wrong" && r.rule_label !== r.true) || (a === "gbm_wrong" && r.gbm_label !== r.true)));
  }
  function renderList() {
    const rs = filtered();
    list.innerHTML = "";
    $("#f-count").textContent = `${rs.length} records`;
    for (const r of rs) {
      const mark = (lab) => (lab === r.true ? "✓" : "✗");
      const b = el("button", { type: "button", role: "option", "aria-selected": String(current === r.id), "data-id": r.id,
        onclick: () => select(r.id) },
        `${r.split} #${r.record} — ${r.true}`, el("br"),
        el("span", { class: "small muted" }, `rules ${r.rule_label} ${mark(r.rule_label)} · GBM ${r.gbm_label} ${mark(r.gbm_label)}`));
      list.append(b);
    }
  }
  async function select(id) {
    current = id;
    for (const b of list.querySelectorAll("button")) b.setAttribute("aria-selected", String(b.dataset.id === id));
    const view = $("#rec-view");
    view.innerHTML = '<p class="muted">Loading…</p>';
    let rec;
    try {
      rec = await (await fetch(`data/explorer/${id}.json`)).json();
    } catch (e) {
      view.innerHTML = `<p class="bad">Could not load the record (${e.message}). Open the site through a web server (python -m http.server -d web/dist), not as a file.</p>`;
      return;
    }
    const x = b64i16(rec.x);
    const T = rec.traces, tr = { x, lsb: rec.lsb, urms: [0, 1, 2].map((c) => b64f32(T.urms).subarray(99 * c, 99 * c + 99)),
      v1mag: b64f32(T.v1mag), v2mag: b64f32(T.v2mag), J_deg: b64f32(T.J_deg), f_dft: b64f32(T.f_dft), f_pll: b64f32(T.f_pll), r_mag: b64f32(T.r_mag) };
    const trace = GMCore.ruleTrace(rec.features);
    const ok = (lab) => (lab === rec.true ? '<span class="ok status-ok">correct</span>' : '<span class="bad status-bad">wrong</span>');
    const p = rec.params || {};
    const keys = Object.keys(p).filter((k) => k.startsWith("ev_") || (k.startsWith("hn_") && p[k]) || k === "tr_on");
    view.innerHTML = "";
    view.append(el("div", { class: "card", html: `<h2 style="margin-top:0">${rec.split} record ${rec.record}: true class <em>${rec.true}</em></h2>
      <p>Why shown: ${rec.why} · SNR ${fmt(rec.snr_db, 1)} dB · ${keys.map((k) => `${k.replace(/^ev_|^hn_/, "")} = ${typeof p[k] === "number" ? (+p[k].toPrecision(4)) : p[k]}`).join(" · ")}</p>
      <div class="grid2"><div><h3>Rules (primary)</h3><p><strong>${rec.rule_label}</strong> (reason: ${GM_SITE.reasons[rec.rule_reason]}) ${ok(rec.rule_label)}
      ${Number.isFinite(rec.rule_onset) ? ` · onset estimate ${fmt(rec.rule_onset, 3)} s` : ""}</p></div>
      <div><h3>GBM-embedded (advisory)</h3><p><strong>${rec.gbm_label}</strong> ${ok(rec.gbm_label)}</p>${GMShow.probaBars(rec.gbm_proba, rec.gbm_label)}</div></div>` }));
    const rc = el("div", { class: "card reclass" });
    const btn = el("button", { class: "primary", type: "button" }, "Re-classify in your browser");
    const out = el("div", { "aria-live": "polite" });
    rc.append(el("h3", { style: "margin-top:0" }, "Re-run the C core here"),
      el("p", { class: "small muted" }, "Runs the same C99 code as the PC software-in-the-loop tests and the C2000 build, compiled to WebAssembly, on the stored int16 waveform."), btn, out);
    view.append(rc);
    btn.addEventListener("click", async () => {
      btn.disabled = true; out.textContent = "Loading the WebAssembly core…";
      try {
        const ver = await GMCore.load();
        const r = GMCore.classify(x, rec.lsb, false);
        const tolm = Object.fromEntries(GM_SITE.features.map((f) => [f.name, f.tol]));
        let worst = 0, worstName = "";
        GM_SITE.features.forEach((f, i) => {
          const gated = f.name === "rho" && !(rec.features[0] < 0.95); // D-069: rho compared only if urms_min_1 < 0.95
          if (f.tol > 0 && !gated && Number.isFinite(rec.features[i]) && Number.isFinite(r.features[i])) {
            const q = Math.abs(rec.features[i] - r.features[i]) / tolm[f.name];
            if (q > worst) { worst = q; worstName = f.name; }
          }
        });
        const same = r.rule_label === rec.rule_label && r.rule_reason === rec.rule_reason && r.gbm_label === rec.gbm_label;
        out.innerHTML = `<p class="${same ? "ok" : "bad"}">${same ? "Reproduced" : "Different"}: rules <strong>${r.rule_label}</strong> (${GM_SITE.reasons[r.rule_reason]}),
          GBM <strong>${r.gbm_label}</strong> (max p ${fmt(Math.max(...r.gbm_proba), 3)}) in ${fmt(r.ms, 1)} ms.
          Largest feature difference: ${fmt(worst, 2)} &times; its tolerance (${worstName}).</p><p class="small muted">${ver}</p>` +
          GMShow.featureTable(rec.features, r.features);
      } catch (e) {
        out.innerHTML = `<p class="bad">${e.message}</p>`;
      }
      btn.disabled = false;
    });
    view.append(el("div", { class: "card", html: GMShow.ruleTable(trace) }));
    const pl = el("div");
    view.append(el("h3", {}, "Signal chain"), pl);
    GMShow.plots(pl, tr, rec.rule_onset);
    view.append(el("div", { class: "card", html: GMShow.featureTable(rec.features, null) }));
  }
  for (const s of ["#f-class", "#f-split", "#f-agree"]) $(s).addEventListener("change", renderList);
  renderList();
  const first = new URLSearchParams(location.search).get("id") || (idx[0] && idx[0].id);
  if (first) select(first);
  document.addEventListener("gm-theme", () => { if (current) select(current); });
});
