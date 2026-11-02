/* gridmon site: minimal SVG charts (line chart with crosshair tooltip, scatter, heat-shaded matrix table).
 * One y-axis per chart (never dual axes); legends for >= 2 series; colours are CSS variables (theme-aware). */
"use strict";
const GMC = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const sv = (tag, attrs = {}) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; };
  function niceTicks(lo, hi, n = 5) {
    if (!(hi > lo)) { hi = lo + 1; }
    const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
    const t = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) t.push(+v.toPrecision(12));
    return t;
  }
  function extent(arrs) {
    let lo = Infinity, hi = -Infinity;
    for (const a of arrs) for (const v of a) if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!Number.isFinite(lo)) return [0, 1];
    if (lo === hi) { lo -= 0.5 * (Math.abs(lo) || 1); hi += 0.5 * (Math.abs(hi) || 1); }
    return [lo, hi];
  }
  function legend(series) {
    const d = document.createElement("div"); d.className = "legend";
    for (const s of series) {
      const it = document.createElement("span");
      it.innerHTML = `<span class="swatch" style="background:${s.color}"></span>`;
      it.append(s.name); d.append(it);
    }
    return d;
  }
  /* series: [{name, x: number[], y: number[], color: "var(--series-1)"}]; opts: {title, xlabel, ylabel, height, ymin, ymax, vlines: [{x, label}], hlines: [{y, label}], fmtY, fmtX, desc} */
  function line(container, series, opts = {}) {
    container.innerHTML = "";
    container.classList.add("chart");
    const W = 640, H = opts.height || 220, m = { l: 52, r: 12, t: opts.title ? 22 : 8, b: opts.xlabel ? 36 : 22 };
    const xs = extent(series.map((s) => s.x));
    let ys = extent(series.map((s) => s.y));
    if (opts.ymin !== undefined) ys[0] = Math.min(ys[0], opts.ymin);
    if (opts.ymax !== undefined) ys[1] = Math.max(ys[1], opts.ymax);
    for (const h of opts.hlines || []) { ys[0] = Math.min(ys[0], h.y); ys[1] = Math.max(ys[1], h.y); }
    const pad = 0.06 * (ys[1] - ys[0]); ys = [ys[0] - pad, ys[1] + pad];
    const X = (v) => m.l + (v - xs[0]) / (xs[1] - xs[0]) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - (v - ys[0]) / (ys[1] - ys[0])) * (H - m.t - m.b);
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.desc || opts.title || "chart" });
    if (opts.title) { const t = sv("text", { x: m.l, y: 14, class: "title" }); t.textContent = opts.title; svg.append(t); }
    const ax = sv("g", { class: "axis" });
    const fy = opts.fmtY || ((v) => +v.toPrecision(4));
    const fx = opts.fmtX || ((v) => +v.toPrecision(4));
    for (const v of niceTicks(ys[0], ys[1], 4)) {
      ax.append(sv("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "gridline" }));
      const t = sv("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }); t.textContent = fy(v); ax.append(t);
    }
    for (const v of niceTicks(xs[0], xs[1], 6)) {
      const t = sv("text", { x: X(v), y: H - m.b + 14, "text-anchor": "middle" }); t.textContent = fx(v); ax.append(t);
    }
    ax.append(sv("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    if (opts.xlabel) { const t = sv("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }); t.textContent = opts.xlabel; ax.append(t); }
    if (opts.ylabel) { const t = sv("text", { x: 12, y: (m.t + H - m.b) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` }); t.textContent = opts.ylabel; ax.append(t); }
    svg.append(ax);
    for (const h of opts.hlines || []) {
      svg.append(sv("line", { x1: m.l, x2: W - m.r, y1: Y(h.y), y2: Y(h.y), stroke: "var(--text-3)", "stroke-dasharray": "4 3", "stroke-width": 1 }));
      if (h.label) { const t = sv("text", { x: W - m.r - 2, y: Y(h.y) - 3, "text-anchor": "end", class: "axis" }); t.setAttribute("fill", "var(--text-2)"); t.setAttribute("font-size", "10"); t.textContent = h.label; svg.append(t); }
    }
    for (const v of opts.vlines || []) {
      svg.append(sv("line", { x1: X(v.x), x2: X(v.x), y1: m.t, y2: H - m.b, stroke: "var(--text-3)", "stroke-dasharray": "2 3", "stroke-width": 1 }));
      if (v.label) { const t = sv("text", { x: X(v.x) + 3, y: m.t + 10 }); t.setAttribute("fill", "var(--text-2)"); t.setAttribute("font-size", "10"); t.textContent = v.label; svg.append(t); }
    }
    for (const s of series) {
      let d = "", pen = false;
      const step = Math.max(1, Math.floor(s.x.length / 1600));
      for (let i = 0; i < s.x.length; i += step) {
        const yv = s.y[i];
        if (!Number.isFinite(yv)) { pen = false; continue; }
        d += (pen ? "L" : "M") + X(s.x[i]).toFixed(1) + " " + Y(yv).toFixed(1); pen = true;
      }
      svg.append(sv("path", { d, fill: "none", style: `stroke:${s.color}`, "stroke-width": s.width || 1.6, "stroke-linejoin": "round" }));
    }
    // crosshair + tooltip
    const cross = sv("line", { y1: m.t, y2: H - m.b, stroke: "var(--text-3)", "stroke-width": 1, visibility: "hidden" });
    svg.append(cross);
    const hit = sv("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
    svg.append(hit);
    const tip = document.createElement("div"); tip.className = "tooltip";
    hit.addEventListener("mousemove", (ev) => {
      const r = svg.getBoundingClientRect(); const px = (ev.clientX - r.left) * W / r.width;
      const xv = xs[0] + (px - m.l) / (W - m.l - m.r) * (xs[1] - xs[0]);
      cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
      const rows = series.map((s) => {
        let best = 0, bd = Infinity; for (let i = 0; i < s.x.length; i += Math.max(1, Math.floor(s.x.length / 2000))) { const dd = Math.abs(s.x[i] - xv); if (dd < bd) { bd = dd; best = i; } }
        return `<span class="swatch" style="background:${s.color}"></span>${s.name}: ${Number.isFinite(s.y[best]) ? fy(s.y[best]) : "—"}`;
      });
      tip.innerHTML = `${opts.xlabel || "x"} ${fx(xv)}<br>` + rows.join("<br>");
      tip.style.display = "block";
      const left = (ev.clientX - container.getBoundingClientRect().left) + 12;
      tip.style.left = Math.min(left, container.clientWidth - 160) + "px"; tip.style.top = "8px";
    });
    hit.addEventListener("mouseleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
    if (series.length >= 2) container.append(legend(series));
    container.append(svg, tip);
    return svg;
  }
  /* points: [{x, y, color, r, label}], opts: {xlog, xlabel, ylabel, title, vlines, desc} */
  function scatter(container, points, opts = {}) {
    container.innerHTML = ""; container.classList.add("chart");
    const W = 640, H = opts.height || 260, m = { l: 56, r: 12, t: 12, b: 40 };
    const tx = (v) => (opts.xlog ? Math.log10(v) : v);
    let xs = extent([points.map((p) => tx(p.x))]), ys = extent([points.map((p) => p.y)]);
    for (const v of opts.vlines || []) { xs[0] = Math.min(xs[0], tx(v.x)); xs[1] = Math.max(xs[1], tx(v.x)); }
    const px = 0.05 * (xs[1] - xs[0]), py = 0.08 * (ys[1] - ys[0]); xs = [xs[0] - px, xs[1] + px]; ys = [ys[0] - py, ys[1] + py];
    const X = (v) => m.l + (tx(v) - xs[0]) / (xs[1] - xs[0]) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - (v - ys[0]) / (ys[1] - ys[0])) * (H - m.t - m.b);
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.desc || "scatter plot" });
    const ax = sv("g", { class: "axis" });
    for (const v of niceTicks(ys[0], ys[1], 4)) {
      ax.append(sv("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "gridline" }));
      const t = sv("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }); t.textContent = +v.toPrecision(4); ax.append(t);
    }
    const xt = opts.xlog ? [1e4, 2e4, 5e4, 1e5, 2e5, 5e5].filter((v) => tx(v) >= xs[0] && tx(v) <= xs[1]) : niceTicks(xs[0], xs[1], 6);
    for (const v of xt) { const t = sv("text", { x: X(v), y: H - m.b + 14, "text-anchor": "middle" }); t.textContent = opts.fmtX ? opts.fmtX(v) : v; ax.append(t); }
    if (opts.xlabel) { const t = sv("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle" }); t.textContent = opts.xlabel; ax.append(t); }
    if (opts.ylabel) { const t = sv("text", { x: 12, y: (m.t + H - m.b) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` }); t.textContent = opts.ylabel; ax.append(t); }
    svg.append(ax);
    for (const v of opts.vlines || []) {
      svg.append(sv("line", { x1: X(v.x), x2: X(v.x), y1: m.t, y2: H - m.b, stroke: "var(--text-3)", "stroke-dasharray": "4 3" }));
      const t = sv("text", { x: X(v.x) + 4, y: m.t + 12 }); t.setAttribute("fill", "var(--text-2)"); t.setAttribute("font-size", "11"); t.textContent = v.label; svg.append(t);
    }
    const tip = document.createElement("div"); tip.className = "tooltip";
    for (const p of points) {
      const c = sv("circle", { cx: X(p.x), cy: Y(p.y), r: p.r || 5, style: `fill:${p.color};stroke:var(--surface-1);stroke-width:2`, tabindex: 0, "aria-label": p.label });
      const show = (ev) => { tip.innerHTML = p.label; tip.style.display = "block"; const r = container.getBoundingClientRect(); const b = c.getBoundingClientRect(); tip.style.left = Math.min(b.left - r.left + 10, container.clientWidth - 220) + "px"; tip.style.top = Math.max(0, b.top - r.top - 40) + "px"; };
      c.addEventListener("mouseenter", show); c.addEventListener("focus", show);
      c.addEventListener("mouseleave", () => (tip.style.display = "none")); c.addEventListener("blur", () => (tip.style.display = "none"));
      svg.append(c);
    }
    container.append(svg, tip);
  }
  /* confusion matrix as an accessible table with sequential (one-hue) shading */
  function matrix(container, mat, classes, caption) {
    const max = Math.max(...mat.flat());
    const short = { normal: "norm", sag_bal: "sag_b", sag_unbal: "sag_u", swell: "swell", phase_jump: "jump", freq_event: "freq", oscillation: "osc" };
    let h = `<div class="table-wrap"><table class="matrix"><caption>${caption}</caption><thead><tr><th scope="col">true \\ predicted</th>`;
    for (const c of classes) h += `<th scope="col" title="${c}">${short[c] || c}</th>`;
    h += "</tr></thead><tbody>";
    mat.forEach((row, i) => {
      h += `<tr><th scope="row">${classes[i]}</th>`;
      row.forEach((v, j) => {
        const a = v === 0 ? 0 : 0.12 + 0.75 * v / max;
        const dark = a > 0.55;
        h += `<td style="background:rgba(var(--heat),${a.toFixed(3)});${dark ? "color:#fff;" : ""}${i === j ? "font-weight:700;" : ""}">${v}</td>`;
      });
      h += "</tr>";
    });
    container.innerHTML = h + "</tbody></table></div>";
  }
  return { line, scatter, matrix, legend };
})();
