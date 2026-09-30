/* Prometheus-alpha engine for the browser lab and for Node.
 * A line-for-line port of src/prometheus/tissue.py, body.py and the protocols in life.py.
 * tests/test_web_parity.py runs it under Node against states recorded from the Python engine.
 * State is one worm: a Float32Array of C*L values, channel-major (x[c*L + i]).
 */
(function (root) {
  "use strict";
  const L = 40, C = 16, ALPHA = 0, V = 1, HEAD = 2, TRUNK = 3, TAIL = 4, R = 5, HIDDEN0 = 6;
  const PERC = 3 * C + 3, FOUNDER = 20;
  const BODY_START = 4, HEAD_END = 12, TRUNK_END = 28, BODY_END = 36;
  const PHYS = { coupling: 0.25, substeps: 2, fireRate: 0.5, clip: 5 };
  const SLOT = 8, N_SLOTS = 6, CUE_ON = 6, US_ON = [3, 4, 5];
  const GROW = 32, REGEN = 32, DELAY = 12, TEST_T = 24;

  function rule(doc) {
    const H = doc.b1.length;
    return {
      H, W1: Float32Array.from(doc.W1.flat()), b1: Float32Array.from(doc.b1),
      W2: Float32Array.from(doc.W2.flat()), b2: Float32Array.from(doc.b2),
    };
  }

  function founder() {
    const x = new Float32Array(C * L);
    x[ALPHA * L + FOUNDER] = 1;
    for (let c = HIDDEN0; c < C; c++) x[c * L + FOUNDER] = 1;
    return x;
  }

  function alive(x) {
    const a = new Uint8Array(L);
    for (let i = 0; i < L; i++) {
      let m = x[ALPHA * L + i];
      if (i > 0) m = Math.max(m, x[ALPHA * L + i - 1]);
      if (i < L - 1) m = Math.max(m, x[ALPHA * L + i + 1]);
      a[i] = m > 0.1 ? 1 : 0;
    }
    return a;
  }

  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
  const gate = (head) => clamp01(2 * head - 1);   // a cell smells only when it is more than half head

  /* One update. cue = [a, b], us scalar, fire = per-site 0/1, gj = conductance multiplier. */
  function step(p, x, cue, us, fire, gj = 1, phys = PHYS) {
    const pre = alive(x);
    const y = new Float32Array(C * L);
    const inp = new Float32Array(PERC), hid = new Float32Array(p.H);
    for (let i = 0; i < L; i++) {
      for (let c = 0; c < C; c++) {
        const xc = x[c * L + i];
        const l = i > 0 ? x[c * L + i - 1] : 0, r = i < L - 1 ? x[c * L + i + 1] : 0;
        inp[c] = xc; inp[C + c] = 0.5 * (r - l); inp[2 * C + c] = l + r - 2 * xc;
      }
      const head = gate(x[HEAD * L + i]) * pre[i];
      inp[3 * C] = cue[0] * head; inp[3 * C + 1] = cue[1] * head; inp[3 * C + 2] = us * pre[i];
      for (let j = 0; j < p.H; j++) {
        let s = p.b1[j];
        const row = j * PERC;
        for (let k = 0; k < PERC; k++) s += p.W1[row + k] * inp[k];
        hid[j] = s > 0 ? s : 0;
      }
      for (let c = 0; c < C; c++) {
        let s = p.b2[c];
        const row = c * p.H;
        for (let j = 0; j < p.H; j++) s += p.W2[row + j] * hid[j];
        y[c * L + i] = x[c * L + i] + s * fire[i];
      }
    }
    const g = phys.coupling * gj;
    let v = y.slice(V * L, V * L + L);
    for (let s = 0; s < phys.substeps; s++) {
      const nv = new Float32Array(L);
      for (let i = 0; i < L; i++) {
        const al = i > 0 ? pre[i - 1] : 0, ar = i < L - 1 ? pre[i + 1] : 0;
        const vl = i > 0 ? v[i - 1] : 0, vr = i < L - 1 ? v[i + 1] : 0;
        nv[i] = v[i] + g * pre[i] * (al * (vl - v[i]) + ar * (vr - v[i]));
      }
      v = nv;
    }
    y.set(v, V * L);
    const post = alive(y);
    for (let i = 0; i < L; i++) {
      const live = pre[i] && post[i];
      for (let c = 0; c < C; c++) {
        const k = c * L + i;
        const val = live ? y[k] : 0;
        y[k] = val < -phys.clip ? -phys.clip : val > phys.clip ? phys.clip : val;
      }
    }
    return y;
  }

  function headWeight(x) {
    const a = alive(x), w = new Float32Array(L);
    for (let i = 0; i < L; i++) w[i] = gate(x[HEAD * L + i]) * a[i];
    return w;
  }

  function response(x) {
    const w = headWeight(x);
    let s = 0, num = 0;
    for (let i = 0; i < L; i++) { s += w[i]; num += x[R * L + i] * w[i]; }
    return s > 0.5 ? num / Math.max(s, 1e-6) : 0;
  }

  /* Region labels count a site as body when its own alpha exceeds 0.1 (not the wider alive
     mask, which also covers the empty growth frontier). */
  function region(x) {
    const lab = new Int8Array(L);
    for (let i = 0; i < L; i++) {
      if (!(x[ALPHA * L + i] > 0.1)) { lab[i] = -1; continue; }
      let best = 0, bv = x[HEAD * L + i];
      for (let k = 1; k < 3; k++) if (x[(HEAD + k) * L + i] > bv) { bv = x[(HEAD + k) * L + i]; best = k; }
      lab[i] = best;
    }
    return lab;
  }

  /* Complete amputation, as in experiments.amputate: every site up to the last head cell plus a
     margin (head, fragment), and every site from the first tail cell (tail, fragment). */
  function amputate(x, kind, margin = 2) {
    const lab = region(x), y = x.slice();
    const kill = (a, b) => { for (let i = a; i < b; i++) for (let c = 0; c < C; c++) y[c * L + i] = 0; };
    if (kind === "head" || kind === "fragment") {
      let last = HEAD_END - 1, any = false;
      for (let i = 0; i < L; i++) if (lab[i] === 0) { last = i; any = true; }
      if (!any) last = HEAD_END - 1;
      kill(0, Math.min(L, last + margin + 1));
    }
    if (kind === "tail" || kind === "fragment") {
      let first = TRUNK_END;
      for (let i = L - 1; i >= 0; i--) if (lab[i] === 2) first = i;
      kill(first, L);
    }
    return y;
  }

  function graft(headX, bodyX, at = HEAD_END) {
    const y = bodyX.slice();
    for (let c = 0; c < C; c++) for (let i = 0; i < at; i++) y[c * L + i] = headX[c * L + i];
    return y;
  }

  /* Add a compiled hidden-channel pattern (10 x L) to the living cells (as v2._written). */
  function writePattern(x, pattern) {
    const y = x.slice();
    for (let c = 0; c < pattern.length; c++)
      for (let i = 0; i < L; i++) if (x[ALPHA * L + i] > 0.1) y[(HIDDEN0 + c) * L + i] += pattern[c][i];
    return y;
  }

  /* Keep one half of a worm after fission at site ``at``. */
  function fission(x, keep, at = 20) {
    const y = x.slice();
    for (let c = 0; c < C; c++) for (let i = 0; i < L; i++)
      if ((keep === "anterior") !== (i < at)) y[c * L + i] = 0;
    return y;
  }

  /* Cell turnover (v0.5 H25, v0.7): each site dies with probability ``frac``; the tissue regrows it. */
  function turnover(x, rand, frac = 0.1) {
    const y = x.slice();
    for (let i = 0; i < L; i++) if (rand() < frac) for (let c = 0; c < C; c++) y[c * L + i] = 0;
    return y;
  }

  /* Gaussian noise on the hidden channels of living cells in sites [a, b) (v0.7 noise measure: head 4-11). */
  function noiseHidden(x, rand, sigma = 1.5, a = BODY_START, b = HEAD_END) {
    const y = x.slice();
    for (let i = a; i < b; i++) {
      if (!(x[ALPHA * L + i] > 0.1)) continue;
      for (let c = HIDDEN0; c < C; c++) {
        const u = Math.max(rand(), 1e-12), v = rand();
        y[c * L + i] += sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }
    }
    return y;
  }

  /* Linear engram decoder (v0.3 H16): which odour each cell's hidden state encodes. */
  function decode(x, dec) {
    const out = new Float32Array(L);
    for (let i = 0; i < L; i++) {
      let s = dec.b;
      for (let c = 0; c < dec.w.length; c++) s += dec.w[c] * x[(HIDDEN0 + c) * L + i];
      out[i] = s;
    }
    return out;
  }

  /* Deterministic generator for update masks and slot orders (mulberry32). */
  function rng(seed) {
    let s = seed >>> 0;
    return function () {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function fireMask(rand, rate = PHYS.fireRate) {
    const m = new Uint8Array(L);
    for (let i = 0; i < L; i++) m[i] = rand() < rate ? 1 : 0;
    return m;
  }

  /* Step programs: arrays of {cue:[a,b], us, tag} */
  function program(n, tag = "rest") {
    return Array.from({ length: n }, () => ({ cue: [0, 0], us: 0, tag }));
  }

  function conditioning(csplus, pairing, rand) {
    const slots = ["+", "+", "-", "-", "0", "0"];
    for (let i = slots.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [slots[i], slots[j]] = [slots[j], slots[i]]; }
    const out = [];
    for (const kind of slots) {
      for (let k = 0; k < SLOT; k++) {
        const cue = [0, 0];
        if (k < CUE_ON && kind !== "0") cue[kind === "+" ? csplus : 1 - csplus] = 1;
        const us = US_ON.includes(k) && ((pairing === "paired" && kind === "+") || (pairing === "unpaired" && kind === "0")) ? 1 : 0;
        out.push({ cue, us, tag: "condition", slot: kind });
      }
    }
    return out;
  }

  /* Probe one cue: rest 4, cue 6, rest 2. The response is read over the last 3 cue steps. */
  function probe(cueIndex) {
    const out = [];
    for (let k = 0; k < 12; k++) {
      const cue = [0, 0];
      if (k >= 4 && k < 10) cue[cueIndex] = 1;
      out.push({ cue, us: 0, tag: k >= 7 && k < 10 ? `read${cueIndex}` : "probe" });
    }
    return out;
  }

  const api = {
    L, C, ALPHA, V, HEAD, TRUNK, TAIL, R, HIDDEN0, PERC, FOUNDER, BODY_START, HEAD_END, TRUNK_END, BODY_END,
    PHYS, SLOT, N_SLOTS, GROW, REGEN, DELAY, TEST_T,
    rule, founder, alive, step, writePattern, fission, turnover, noiseHidden, decode, headWeight, response, region, amputate, graft, rng, fireMask, program, conditioning, probe,
  };
  root.Prometheus = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis);
