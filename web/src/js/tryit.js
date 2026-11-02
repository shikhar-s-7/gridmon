/* Try-it page: simplified in-browser event generator, CSV upload (LN->LL, resampling), WASM classification. */
"use strict";
const GMTry = (() => {
  const N = 5000, FS = 5000;
  function rng(seed) { // mulberry32 + Box-Muller (deterministic demo noise)
    let a = (seed >>> 0) + 0x6d2b79f5;
    const u = () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    return { u, n: () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u()) };
  }
  /* Simplified phasor model: V1(t), V2(t) relative to theta(t); returns planar Int16Array (12-bit counts, LSB 2^-10 pu). */
  function synth(p) {
    const r = rng(p.seed * 7919 + 13);
    const th0 = 2 * Math.PI * r.u(), unb = 0.005, unbAng = 2 * Math.PI * r.u();
    const h5 = 0.015, h7 = 0.01, ph5 = 2 * Math.PI * r.u(), ph7 = 2 * Math.PI * r.u();
    const sigma = Math.pow(10, -p.snr / 20) / Math.SQRT2;
    const x = new Int16Array(3 * N);
    const a = [0, -2 * Math.PI / 3, 2 * Math.PI / 3]; // phase offsets of r1 (1, a^2, a)
    let theta = th0, f = 50;
    for (let n = 0; n < N; n++) {
      const t = n / FS, on = t >= p.t0, dur = t >= p.t0 && t < p.t0 + p.dur;
      let m1 = 1, a1 = 0, m2 = unb, a2 = unbAng; // |V1|, angle V1, |V2|, angle V2
      if (p.cls === "normal" && t >= 0.5) m1 = 1.01;
      if (p.cls === "sag_bal" && dur) { m1 = p.depth; m2 = unb * p.depth; }
      if (p.cls === "sag_unbal" && dur) { m1 = (1 + p.depth) / 2; m2 = (1 - p.depth) / 2; a2 = 0; }
      if (p.cls === "swell" && dur) { m1 = p.depth; }
      if (p.cls === "phase_jump" && on) { a1 = p.jump * Math.PI / 180; }
      if (p.cls === "oscillation" && on) {
        const E = Math.min(1, (t - p.t0) / 0.05), s = Math.sin(2 * Math.PI * p.fm * (t - p.t0));
        m1 = 1 + (p.osc / 100) * E * s; a1 = 0.02 * E * Math.sin(2 * Math.PI * p.fm * (t - p.t0) + 1.0);
      }
      if (p.cls === "freq_event") {
        const Tr = Math.abs(p.df / p.rocof);
        f = 50 + (on ? Math.sign(p.rocof) * Math.min(Math.abs(p.rocof) * (t - p.t0), Math.abs(p.rocof) * Tr) : 0);
      }
      const ph = [0, 0, 0];
      for (let k = 0; k < 3; k++) {
        // phase voltage k: Re{(V1 r1_k + V2 r2_k) e^{j theta}} + harmonics (pu of nominal LN peak)
        ph[k] = m1 * Math.cos(theta + a1 + a[k]) + m2 * Math.cos(theta + a2 - a[k]) +
          h5 * Math.cos(5 * (theta + a[k]) + ph5) + h7 * Math.cos(7 * (theta + a[k]) + ph7);
      }
      const s3 = Math.sqrt(3), ll = [(ph[0] - ph[1]) / s3, (ph[1] - ph[2]) / s3, (ph[2] - ph[0]) / s3]; // LL = (v_a - v_b)/sqrt3
      for (let c = 0; c < 3; c++) {
        const v = ll[c] + sigma * r.n();
        x[c * N + n] = Math.max(-2048, Math.min(2047, Math.floor(v * 1024 + 0.5)));
      }
      theta += 2 * Math.PI * f / FS;
    }
    return x;
  }
  /* CSV -> {cols: [Float64Array x3], fs, hasTime} */
  function parseCSV(text) {
    const lines = text.split(/\r?\n/).filter((l) => l.trim() && !l.trim().startsWith("#"));
    const rows = [];
    for (const l of lines) {
      const v = l.split(/[,;\t ]+/).filter((s) => s !== "").map(Number);
      if (v.every(Number.isFinite)) rows.push(v);
    }
    if (rows.length < 100) throw new Error("fewer than 100 numeric rows");
    const w = rows[0].length;
    if (w !== 3 && w !== 4) throw new Error(`expected 3 or 4 numeric columns, found ${w}`);
    const off = w - 3, cols = [0, 1, 2].map((c) => Float64Array.from(rows, (r) => r[c + off]));
    let fs = null;
    if (off) { const dt = (rows[rows.length - 1][0] - rows[0][0]) / (rows.length - 1); if (dt > 0) fs = 1 / dt; }
    return { cols, fs, n: rows.length };
  }
  /* windowed-sinc resampling to 5 kHz (Blackman window, cut-off 0.45 x the lower rate, weights normalised) */
  function resample(x, fsIn) {
    const ratio = fsIn / FS, nOut = Math.floor((x.length - 1) / ratio) + 1;
    const fc = 0.45 * Math.min(1, 1 / ratio);              // cycles per input sample
    const W = Math.ceil(8 / fc);                             // half-width in input samples
    const y = new Float64Array(nOut);
    for (let k = 0; k < nOut; k++) {
      const tc = k * ratio, i0 = Math.ceil(tc - W), i1 = Math.floor(tc + W);
      let s = 0, ws = 0;
      for (let i = Math.max(0, i0); i <= Math.min(x.length - 1, i1); i++) {
        const u = i - tc, z = 2 * fc * u;
        const sinc = z === 0 ? 1 : Math.sin(Math.PI * z) / (Math.PI * z);
        const bw = 0.42 + 0.5 * Math.cos(Math.PI * u / W) + 0.08 * Math.cos(2 * Math.PI * u / W);
        const wgt = sinc * bw; s += wgt * x[i]; ws += wgt;
      }
      y[k] = ws !== 0 ? s / ws : 0;
    }
    return y;
  }
  return { synth, parseCSV, resample };
})();

document.addEventListener("DOMContentLoaded", () => {
  if (!document.getElementById("gen-form")) return;
  const { $, el, fmt } = GMU;
  const show = async (x, lsb, title, note) => {
    const out = $("#result");
    out.innerHTML = '<p class="muted">Classifying…</p>';
    try {
      const ver = await GMCore.load();
      const r = GMCore.classify(x, lsb, true);
      const trace = GMCore.ruleTrace(r.features);
      out.innerHTML = "";
      out.append(el("div", { class: "card", html: `<h3 style="margin-top:0">${title}</h3>${note ? `<p class="small muted">${note}</p>` : ""}
        <div class="grid2"><div><h3>Rules (primary)</h3><p class="kpi" data-testid="rule-label">${r.rule_label}</p><p>reason: ${GM_SITE.reasons[r.rule_reason]}
        ${Number.isFinite(r.onset) ? ` · onset estimate ${fmt(r.onset, 3)} s` : ""}</p></div>
        <div><h3>GBM-embedded (advisory)</h3><p class="kpi" data-testid="gbm-label">${r.gbm_label}</p>${GMShow.probaBars(r.gbm_proba, r.gbm_label)}</div></div>
        <p class="small muted">Computed in ${fmt(r.ms, 1)} ms by ${ver}.</p>` }));
      out.append(el("div", { class: "card", html: GMShow.ruleTable(trace) }));
      const pl = el("div");
      out.append(pl);
      GMShow.plots(pl, { x, lsb, ...r.traces }, r.onset);
      out.append(el("div", { class: "card", html: GMShow.featureTable(r.features, null) }));
    } catch (e) {
      out.innerHTML = `<p class="bad">Could not run the WebAssembly core: ${e.message}. Open the site through a web server (python -m http.server -d web/dist).</p>`;
    }
  };
  $("#g-run").addEventListener("click", () => {
    const v = (id) => Number($(id).value);
    const p = { cls: $("#g-class").value, depth: v("#g-depth"), dur: v("#g-dur"), jump: v("#g-jump"), rocof: v("#g-rocof") || 1,
      df: v("#g-df"), osc: v("#g-osc"), fm: v("#g-fm"), t0: v("#g-t0"), snr: v("#g-snr"), seed: v("#g-seed") | 0 };
    if (p.cls === "swell" && p.depth < 1) p.depth = 1.2;
    const x = GMTry.synth(p);
    show(x, 2 ** -10, `Synthesised: ${p.cls}`, "Simplified generator (demo), 12-bit counts at 5 kHz.");
  });
  let csv = null;
  $("#c-file").addEventListener("change", async (ev) => {
    const f = ev.target.files[0];
    if (!f) return;
    try {
      csv = GMTry.parseCSV(await f.text());
      if (csv.fs) $("#c-fs").value = Math.round(csv.fs);
      const dur = csv.n / Number($("#c-fs").value);
      $("#c-info").textContent = `${f.name}: ${csv.n} rows, ${csv.fs ? "sample rate from the time column" : "enter the sample rate"}, ${dur.toFixed(3)} s.`;
      $("#c-start").max = Math.max(0, dur - 1).toFixed(2);
      $("#c-run").disabled = false;
    } catch (e) {
      $("#c-info").textContent = `Could not read ${f.name}: ${e.message}`;
      $("#c-run").disabled = true;
    }
  });
  $("#c-run").addEventListener("click", () => {
    if (!csv) return;
    const fs = Number($("#c-fs").value), un = Number($("#c-un").value), start = Number($("#c-start").value);
    const pk = Math.SQRT2 * un; // nominal LL peak (V)
    let ll = csv.cols;
    if ($("#c-type").value === "LN") ll = [0, 1, 2].map((c) => Float64Array.from(csv.cols[c], (v, i) => v - csv.cols[(c + 1) % 3][i]));
    const pu = ll.map((col) => Float64Array.from(col, (v) => v / pk));
    const rs = fs === 5000 ? pu : pu.map((col) => GMTry.resample(col, fs));
    const i0 = Math.round(start * 5000);
    if (rs[0].length < i0 + 5000) { $("#c-info").textContent = "The window must lie inside the file: at least 1 s of data after the start."; return; }
    const lsb = 2 ** -14, x = new Int16Array(15000);
    let clipped = 0;
    for (let c = 0; c < 3; c++) for (let n = 0; n < 5000; n++) {
      const q = Math.round(rs[c][i0 + n] / lsb);
      if (Math.abs(q) > 32767) clipped++;
      x[c * 5000 + n] = Math.max(-32767, Math.min(32767, q));
    }
    show(x, lsb, `Uploaded: window ${start.toFixed(2)}–${(start + 1).toFixed(2)} s`,
      `${$("#c-type").value} input, ${fs} Hz${fs === 5000 ? "" : " resampled to 5 kHz"}, nominal ${un} V LL RMS${clipped ? `; ${clipped} samples clipped at ±2 pu` : ""}.`);
  });
});
