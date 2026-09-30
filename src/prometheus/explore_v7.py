"""Exploratory (not preregistered): how did stress training change the engram?

v0.6 offered, post hoc, a reading of the noise / cell-loss trade-off: a sharp code replicated across
many cells can be re-seeded by neighbours when cells die, while a graded code tolerates noise but is
lost with the cells that carry it. This script measures, for the S, B, F, C, T and N rules of each lineage,
the engram field of intact trained worms: identical twins (same founder, schedule and update masks)
trained with CS+ = A and CS+ = B, and the distance between their hidden states at every site.

* spread: participation ratio (sum d)^2 / sum d^2 of the per-site distance d, in cells;
* amplitude: mean distance over the living sites;
* head_cv: coefficient of variation of d over the head sites 4-11 (0 = every head cell carries the
  same-sized copy; large = a graded code concentrated in a few cells);
* snr: amplitude over the within-group spread of the same states (how far apart the two memories
  are relative to how much worms with the same memory differ).

Written after the v0.7 lock, while the rules trained and before any v0.7 result was seen, but not
preregistered: nothing here is a confirmatory claim.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from . import provenance, v2, v6
from . import experiments as X
from .train import load

N, SEED = 64, 8040


def engram(p, n=N) -> dict:
    co = X.cohort(SEED, n)
    base = X.Worms.found(p, co)
    xa = base.learn("paired", csplus=np.zeros(n, dtype=int)).x
    xb = base.learn("paired", csplus=np.ones(n, dtype=int)).x
    live = ((xa[:, 0] > 0.1) & (xb[:, 0] > 0.1)).float()               # (n, L)
    ha, hb = xa[:, v2.HIDDEN], xb[:, v2.HIDDEN]                          # (n, 10, L)
    d = ((ha - hb).norm(dim=1) * live).mean(0)                           # (L,)
    within = torch.cat([ha - ha.mean(0), hb - hb.mean(0)]).norm(dim=1)   # (2n, L)
    within = (within * torch.cat([live, live])).mean(0)
    alive = live.mean(0) > 0.5
    amp, head = float(d[alive].mean()), d[4:12]
    return {"profile": d.tolist(), "spread": float(d.sum() ** 2 / (d ** 2).sum()), "amplitude": amp,
            "head_cv": float(head.std() / head.mean()), "snr": amp / float(within[alive].mean())}


def run() -> dict:
    torch.set_num_threads(4)
    root = pathlib.Path(provenance.ROOT)
    res = {"exploratory": True, "note": __doc__.strip().splitlines()[0], "n": N, "seed": SEED,
           "provenance": provenance.stamp(), "rules": {}}
    with torch.no_grad():
        for s in (0, 1):
            for fam in ("S", "B", "F", "C", "T", "N"):
                r = f"{fam}{s}"
                res["rules"][r] = engram(load(str(root / "weights" / f"rule_{r}.json"))["tensors"])
                e = res["rules"][r]
                print(f"[{r}] spread {e['spread']:.1f} cells  amplitude {e['amplitude']:.3f}  head cv {e['head_cv']:.2f}  snr {e['snr']:.2f}", flush=True)
    return res


def relate(res: dict, v7: dict) -> dict:
    """Added AFTER the v0.7 results were seen (post hoc): rank correlations, over the eight S, C, T and N
    rules, between the engram measures and the v0.7 memories (exact one-sided Spearman tests)."""
    rules = [f"{f}{s}" for s in (0, 1) for f in "SCTN"]
    ex, mem = res["rules"], v7["rules"]
    out = {"note": "post hoc: computed after the v0.7 results were seen", "rules": rules}
    for name, a, b in (("amplitude_vs_noise_memory", "amplitude", "noise_memory"),
                       ("snr_vs_noise_memory", "snr", "noise_memory"),
                       ("flatness_vs_turnover_memory", "head_cv", "turnover_memory"),
                       ("amplitude_vs_turnover_memory", "amplitude", "turnover_memory")):
        x = np.array([-ex[r][a] if a == "head_cv" else ex[r][a] for r in rules])
        rho, pv = v6.spearman_exact(x, np.array([mem[r][b] for r in rules]))
        out[name] = {"rho": rho, "p_one_sided": pv}
    return out


if __name__ == "__main__":
    res = run()
    v7 = pathlib.Path("results/v7.json")
    if v7.exists():
        res["post_hoc"] = relate(res, json.loads(v7.read_text()))
        for k, v in res["post_hoc"].items():
            if isinstance(v, dict):
                print(f"{k}: rho {v['rho']:+.2f} (one-sided p {v['p_one_sided']:.3g})")
    pathlib.Path("results/exploratory_v7_engram.json").write_text(json.dumps(res, indent=1))
