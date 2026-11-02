// Development tool (not deployed): run the gridmon WebAssembly module in Node on binary records.
// Usage: node web/tools/wasm_run.mjs <gridmon.wasm> <records.bin> [lsb] [--traces]
//   records.bin: N records of 3 x 5000 int16 little-endian, planar (ab | bc | ca)
// Prints one JSON line per record: {"out": [58 floats, null = NaN], "traces": [...]?}
import { readFileSync } from "node:fs";

const [wasmPath, binPath, lsbArg, ...rest] = process.argv.slice(2);
const lsb = lsbArg && !lsbArg.startsWith("--") ? Number(lsbArg) : 2 ** -10;
const withTraces = rest.includes("--traces") || lsbArg === "--traces";
const mod = new WebAssembly.Module(readFileSync(wasmPath));
const wasi = new Proxy({}, { get: () => () => 0 }); // stub for any WASI import (the build has none)
const inst = new WebAssembly.Instance(mod, { wasi_snapshot_preview1: wasi });
const ex = inst.exports;
if (ex._initialize) ex._initialize();
const n = 3 * 5000;
const buf = readFileSync(binPath);
const recs = new Int16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
const outLen = ex.gm_wasm_out_len();
const trLen = ex.gm_wasm_tr_len();
const lines = [];
for (let r = 0; r * n < recs.length; r++) {
  new Int16Array(ex.memory.buffer, ex.gm_wasm_input(), n).set(recs.subarray(r * n, (r + 1) * n));
  ex.gm_wasm_run(lsb, withTraces ? 1 : 0);
  const out = Array.from(new Float32Array(ex.memory.buffer, ex.gm_wasm_output(), outLen), (v) => (Number.isNaN(v) ? null : v));
  const line = { out };
  if (withTraces) {
    line.traces = Array.from(new Float32Array(ex.memory.buffer, ex.gm_wasm_traces(), trLen), (v) => (Number.isNaN(v) ? null : v));
  }
  lines.push(JSON.stringify(line));
}
process.stdout.write(lines.join("\n") + "\n");
