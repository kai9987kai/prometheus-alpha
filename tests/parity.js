// Replays tests/fixtures/parity.json with web/engine.js and prints the largest differences.
"use strict";
const fs = require("fs");
const path = require("path");
const P = require(path.join(__dirname, "..", "web", "engine.js"));

const fx = JSON.parse(fs.readFileSync(process.argv[2] || path.join(__dirname, "fixtures", "parity.json"), "utf8"));
const report = [];
for (const cs of fx.cases) {
  const p = P.rule(cs.params);
  const B = cs.masks[0].length;
  let maxState = 0, maxResp = 0, regionMismatch = 0;
  for (let b = 0; b < B; b++) {
    let x = P.founder();
    let si = 0;
    for (let t = 0; t < cs.T; t++) {
      if (t === cs.cut_at) x = P.amputate(x, "head");
      x = P.step(p, x, cs.cue[t][b], cs.us[t][b], cs.masks[t][b], cs.gj[t][b]);
      maxResp = Math.max(maxResp, Math.abs(P.response(x) - cs.resp[t][b]));
      while (si < cs.states.length && cs.states[si].t < t) si++;
      if (si < cs.states.length && cs.states[si].t === t) {
        const want = cs.states[si].x[b];
        for (let c = 0; c < P.C; c++) for (let i = 0; i < P.L; i++)
          maxState = Math.max(maxState, Math.abs(x[c * P.L + i] - want[c][i]));
      }
    }
    const lab = P.region(x);
    for (let i = 0; i < P.L; i++) if (lab[i] !== cs.region_final[b][i]) regionMismatch++;
  }
  report.push({ name: cs.name, steps: cs.T, maxState, maxResp, regionMismatch });
}
console.log(JSON.stringify(report));
