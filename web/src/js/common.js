/* gridmon site: shared helpers (no dependencies). */
"use strict";
const GMU = (() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const el = (tag, attrs = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") e.className = v;
      else if (k === "html") e.innerHTML = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid !== null && kid !== undefined) e.append(kid instanceof Node ? kid : String(kid));
    return e;
  };
  const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
  const pct = (v, d = 1) => (v === null || v === undefined ? "—" : (100 * v).toFixed(d) + " %");
  const signed = (v, d = 3) => (v === null || v === undefined ? "—" : (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d));
  const ci = (a, d = 3) => (a ? `[${signed(a[0], d)}, ${signed(a[1], d)}]` : "—");
  const sci = (v, d = 1) => (v === null || v === undefined ? "—" : Number(v).toExponential(d));
  const int = (v) => (v === null || v === undefined ? "—" : Number(v).toLocaleString("en-US"));
  const b64ToBytes = (s) => { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; };
  const b64f32 = (s) => { const u = b64ToBytes(s); return new Float32Array(u.buffer, u.byteOffset, u.byteLength / 4); };
  const b64i16 = (s) => { const u = b64ToBytes(s); return new Int16Array(u.buffer, u.byteOffset, u.byteLength / 2); };
  const CLASS_LABELS = { normal: "normal", sag_bal: "balanced sag", sag_unbal: "unbalanced sag", swell: "swell",
    phase_jump: "phase jump", freq_event: "frequency event", oscillation: "oscillation" };
  const METHOD_LABELS = { rules: "Rules (standards-based)", gbm_embedded: "GBM-embedded (13.6 KB)", gbm_unconstrained: "GBM-unconstrained (HGB)" };
  const METHOD_VARS = { rules: "var(--series-1)", gbm_embedded: "var(--series-2)", gbm_unconstrained: "var(--series-3)" };
  function initTheme() {
    const btn = $(".theme-toggle");
    if (!btn) return;
    let saved = null;
    try { saved = localStorage.getItem("gm-theme"); } catch (e) { /* storage unavailable */ }
    if (saved) document.documentElement.setAttribute("data-theme", saved);
    const label = () => {
      const t = document.documentElement.getAttribute("data-theme") ||
        (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      btn.textContent = t === "dark" ? "Light theme" : "Dark theme";
      btn.setAttribute("aria-label", `Switch to ${t === "dark" ? "light" : "dark"} theme`);
      return t;
    };
    label();
    btn.addEventListener("click", () => {
      const next = label() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("gm-theme", next); } catch (e) { /* ignore */ }
      label();
      document.dispatchEvent(new CustomEvent("gm-theme"));
    });
  }
  document.addEventListener("DOMContentLoaded", initTheme);
  return { $, el, fmt, pct, signed, ci, sci, int, b64f32, b64i16, CLASS_LABELS, METHOD_LABELS, METHOD_VARS };
})();
