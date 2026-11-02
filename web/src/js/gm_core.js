/* gridmon site: the C core compiled to WebAssembly (wasm/gridmon.wasm, built from the same sources as the PC
 * SIL library with GM_ENABLE_GBM=1) + the rule decision trace for display. No network access: the module is
 * fetched from this site. Output layout: web/wasm/gm_wasm.c. */
"use strict";
const GMCore = (() => {
  let ex = null, loading = null;
  const N = 3 * 5000;
  function load(url = "wasm/gridmon.wasm") {
    if (loading) return loading;
    const wasi = new Proxy({}, { get: () => () => 0 }); // stub for any WASI import (the build has none)
    loading = fetch(url)
      .then((r) => { if (!r.ok) throw new Error(`cannot load ${url} (${r.status})`); return r.arrayBuffer(); })
      .then((b) => WebAssembly.instantiate(b, { wasi_snapshot_preview1: wasi }))
      .then(({ instance }) => { ex = instance.exports; if (ex._initialize) ex._initialize(); return version(); });
    return loading;
  }
  function version() {
    const p = ex.gm_wasm_version(), mem = new Uint8Array(ex.memory.buffer);
    let s = ""; for (let i = p; mem[i] !== 0 && i < p + 400; i++) s += String.fromCharCode(mem[i]);
    return s;
  }
  /* x: Int16Array(15000) planar ab|bc|ca, lsb: pu per count */
  function classify(x, lsb, withTraces = false) {
    if (!ex) throw new Error("WASM core not loaded");
    if (x.length !== N) throw new Error("record must be 3 x 5000 int16 samples");
    new Int16Array(ex.memory.buffer, ex.gm_wasm_input(), N).set(x);
    const t0 = performance.now();
    const st = ex.gm_wasm_run(lsb, withTraces ? 1 : 0);
    const ms = performance.now() - t0;
    const o = new Float32Array(ex.memory.buffer, ex.gm_wasm_output(), ex.gm_wasm_out_len());
    const res = { status: st, flags: o[1], rule_label: GM_SITE.classes[o[2]], rule_reason: o[3], onset: o[4],
      gbm_label: o[5] >= 0 ? GM_SITE.classes[o[5]] : null, gbm_proba: Array.from(o.slice(6, 13)),
      features: Array.from(o.slice(13, 50)), aux: Array.from(o.slice(50, 58)), ms };
    if (withTraces) {
      const t = new Float32Array(ex.memory.buffer, ex.gm_wasm_traces(), ex.gm_wasm_tr_len()).slice();
      let k = 0; const take = (n) => { const a = t.subarray(k, k + n); k += n; return a; };
      const urms = [take(99), take(99), take(99)];
      const v1re = take(1000), v1im = take(1000), v2re = take(1000), v2im = take(1000);
      const dfdft = take(1000), rocof = take(1000), J = take(1000), fpll = take(1000);
      take(2000); // masks
      const e_mag = take(100); take(100); const r_mag = take(100), r_ang = take(100);
      const mag = (re, im) => Float32Array.from(re, (v, i) => Math.hypot(v, im[i]));
      res.traces = { urms, v1mag: mag(v1re, v1im), v2mag: mag(v2re, v2im), J_deg: J.map((v) => v * 180 / Math.PI),
        f_dft: dfdft.map((v) => 50 + v), f_pll: fpll.map((v) => 50 + v), rocof, e_mag, r_mag, r_ang_deg: r_ang.map((v) => v * 180 / Math.PI) };
    }
    return res;
  }
  /* Rule decision trace (docs/DSP_SPEC.md section 10, priority order; NaN compares false). */
  function ruleTrace(features) {
    const names = GM_SITE.features.map((f) => f.name);
    const f = Object.fromEntries(names.map((n, i) => [n, features[i] === null ? NaN : features[i]]));
    const th = GM_SITE.results.thresholds;
    const dv1 = Math.max(Math.abs(f.v1_min_rel - 1), Math.abs(f.v1_max_rel - 1));
    const dip = f.urms_min_1 < th.dip_pu;
    const steps = [
      { rule: "1 dip, unbalanced", label: "sag_unbal", reason: 2, tests: [["urms_min_1", f.urms_min_1, "<", th.dip_pu], ["rho", f.rho, ">", th.rho_th]] },
      { rule: "1 dip, balanced", label: "sag_bal", reason: 1, tests: [["urms_min_1", f.urms_min_1, "<", th.dip_pu]] },
      { rule: "2 swell", label: "swell", reason: 3, tests: [["urms_max_3", f.urms_max_3, ">", th.swell_pu]] },
      { rule: "3 oscillation (magnitude)", label: "oscillation", reason: 4, tests: [["env_mag_amp_pct", f.env_mag_amp_pct, ">", th.osc_mag_pct], ["env_mag_prom", f.env_mag_prom, ">=", th.osc_prom]] },
      { rule: "3 oscillation (angle)", label: "oscillation", reason: 5, tests: [["env_ang_amp_deg", f.env_ang_amp_deg, ">", th.osc_ang_deg], ["env_ang_prom", f.env_ang_prom, ">=", th.osc_prom]] },
      { rule: "4 phase jump", label: "phase_jump", reason: 6, tests: [["j_max_deg", f.j_max_deg, ">", th.jump_deg], ["max|v1_rel - 1|", dv1, "<", th.jump_max_dv1]] },
      { rule: "5 frequency event", label: "freq_event", reason: 7, tests: [["|df_end|", Math.abs(f.df_end), ">", th.df_hz]] },
    ];
    const cmp = { "<": (a, b) => a < b, ">": (a, b) => a > b, ">=": (a, b) => a >= b };
    let fired = null;
    for (const s of steps) {
      s.results = s.tests.map(([n, v, op, t]) => cmp[op](v, t));
      s.pass = s.results.every(Boolean);
      if (s.pass && !fired) fired = s;
      s.fired = fired === s;
    }
    void dip;
    return { steps, label: fired ? fired.label : "normal", reason: fired ? fired.reason : 0 };
  }
  return { load, classify, ruleTrace, version: () => (ex ? version() : null) };
})();
