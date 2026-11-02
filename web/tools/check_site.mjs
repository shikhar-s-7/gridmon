// Development tool (not deployed): headless check of the built site (web/dist) with Playwright/Chromium.
// Usage: node web/tools/check_site.mjs <base-url> <out.json> <screenshot-dir>
// Checks: every page loads without console errors and shows the simulation-only banner; no horizontal scroll at
// phone width; explorer re-classification reproduces the stored result (several records); try-it classifies one
// synthesised sample per class and one example CSV. Saves screenshots m8_{overview,results,explorer,tryit}.png.
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright"); // pre-installed globally (NODE_PATH)
import { writeFileSync } from "node:fs";

const [base, outPath, shotDir] = process.argv.slice(2);
const BANNER = "Simulation-only research prototype. Not validated on measured data. Not for protection or control.";
const pages = ["index", "results", "explorer", "tryit", "firmware", "about"];
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1280, height: 1000 } });
const report = { pages: {}, explorer: [], tryit: {}, csv: null, mobile: {} };
const errors = [];
const page = await ctx.newPage();
page.on("console", (m) => { if (m.type() === "error") errors.push(`${page.url()}: ${m.text()}`); });
page.on("pageerror", (e) => errors.push(`${page.url()}: ${e.message}`));

for (const p of pages) {
  const before = errors.length;
  await page.goto(`${base}/${p}.html`, { waitUntil: "networkidle" });
  const banner = (await page.textContent(".banner")) || "";
  const unresolved = (await page.content()).includes("{{");
  report.pages[p] = { banner: banner.trim() === BANNER, unresolved_tokens: unresolved, console_errors: errors.length - before,
    title: await page.title() };
  if (p === "index") await page.screenshot({ path: `${shotDir}/m8_overview.png` });
  if (p === "results") { await page.locator("#cm-iid table").waitFor(); await page.screenshot({ path: `${shotDir}/m8_results.png` }); }
}

// explorer: re-classify several records (first, a disagreement, a test_shift record)
await page.goto(`${base}/explorer.html`, { waitUntil: "networkidle" });
const ids = await page.evaluate(() => {
  const idx = window.GM_EXPLORER;
  const pick = [idx[0], idx.find((r) => r.rule_label !== r.gbm_label), idx.find((r) => r.split === "test_shift" && r.true === "phase_jump"),
    idx.find((r) => r.true === "oscillation"), idx.find((r) => r.true === "freq_event")];
  return pick.filter(Boolean).map((r) => r.id);
});
for (const id of ids) {
  await page.goto(`${base}/explorer.html?id=${id}`, { waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Re-classify in your browser" }).click();
  const out = page.locator(".reclass [aria-live] p").first();
  await out.waitFor({ timeout: 20000 });
  const text = (await out.textContent()) || "";
  report.explorer.push({ id, reproduced: text.startsWith("Reproduced"), text: text.slice(0, 200) });
  if (id === ids[1]) {
    await page.evaluate(() => { document.querySelector(".reclass").scrollIntoView({ block: "start" }); window.scrollBy(0, -90); });
    await page.screenshot({ path: `${shotDir}/m8_explorer.png` });
  }
}

// try-it: one synthesised sample per class with the default parameters (swell: 1.2 pu)
await page.goto(`${base}/tryit.html`, { waitUntil: "networkidle" });
const classes = ["normal", "sag_bal", "sag_unbal", "swell", "phase_jump", "freq_event", "oscillation"];
for (const c of classes) {
  await page.selectOption("#g-class", c);
  await page.fill("#g-depth", c === "swell" ? "1.2" : "0.5");
  await page.click("#g-run");
  const lab = page.locator('[data-testid="rule-label"]');
  await lab.waitFor({ timeout: 20000 });
  report.tryit[c] = { rule: (await lab.textContent()).trim(), gbm: (await page.locator('[data-testid="gbm-label"]').textContent()).trim() };
  if (c === "phase_jump") {
    await page.locator("#result").scrollIntoViewIfNeeded();
    await page.evaluate(() => document.querySelector("#result").scrollIntoView({ block: "start" }));
    await page.evaluate(() => window.scrollBy(0, -60));
    await page.screenshot({ path: `${shotDir}/m8_tryit.png` });
  }
}
// CSV upload: the LN example
await page.setInputFiles("#c-file", "web/dist/examples/example_sag_unbal_LN_10kHz.csv");
await page.selectOption("#c-type", "LN");
await page.click("#c-run");
await page.locator('[data-testid="rule-label"]').waitFor();
report.csv = { file: "example_sag_unbal_LN_10kHz.csv", rule: (await page.locator('[data-testid="rule-label"]').textContent()).trim(),
  gbm: (await page.locator('[data-testid="gbm-label"]').textContent()).trim() };

// phone width: no horizontal page scroll
const mob = await browser.newPage({ viewport: { width: 390, height: 844 } });
for (const p of pages) {
  await mob.goto(`${base}/${p}.html`, { waitUntil: "networkidle" });
  report.mobile[p] = await mob.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1);
}
report.console_errors = errors;
report.ok = Object.values(report.pages).every((v) => v.banner && !v.unresolved_tokens && v.console_errors === 0) &&
  report.explorer.every((r) => r.reproduced) && errors.length === 0 && Object.values(report.mobile).every(Boolean);
writeFileSync(outPath, JSON.stringify(report, null, 2));
console.log(JSON.stringify({ ok: report.ok, explorer: report.explorer.map((r) => r.reproduced), tryit: report.tryit, csv: report.csv,
  mobile: report.mobile, errors: errors.slice(0, 5) }));
await browser.close();
