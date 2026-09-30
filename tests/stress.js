// Checks the lab's v0.7 stress operators (web/engine.js turnover and noiseHidden) against their definitions.
"use strict";
const path = require("path");
const P = require(path.join(__dirname, "..", "web", "engine.js"));

const x = new Float32Array(P.C * P.L);
for (let i = P.BODY_START; i < P.BODY_END; i++) for (let c = 0; c < P.C; c++) x[c * P.L + i] = c === P.ALPHA ? 1 : 0.5;

const out = {};
out.turnoverAll = P.turnover(x, () => 0, 0.1).every((v) => v === 0);          // every draw < 0.1: all die
out.turnoverNone = P.turnover(x, () => 0.99, 0.1).every((v, k) => v === x[k]); // no draw < 0.1: none die
let killed = 0;
const y = P.turnover(x, P.rng(3), 0.1);
for (let i = P.BODY_START; i < P.BODY_END; i++) if (y[P.ALPHA * P.L + i] === 0) killed++;
out.turnoverKilled = killed;

const z = P.noiseHidden(x, P.rng(4), 1.5);
let outside = 0, visible = 0, inside = 0, sq = 0;
for (let c = 0; c < P.C; c++) for (let i = 0; i < P.L; i++) {
  const d = z[c * P.L + i] - x[c * P.L + i];
  if (c < P.HIDDEN0) { if (d !== 0) visible++; continue; }
  if (i < P.BODY_START || i >= P.HEAD_END) { if (d !== 0) outside++; continue; }
  inside++; sq += d * d;
}
out.noiseOutside = outside;
out.noiseVisible = visible;
out.noiseSd = Math.sqrt(sq / inside);
console.log(JSON.stringify(out));
