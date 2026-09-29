"""Prometheus-alpha v0.5 experiments (prereg/PREREGISTRATION_v5.json). No new rules and no pilots:
every test is held out for all eight rules (E0, E1, S0, S1, B0, B1, F0, F1).

H25 turnover (Ship of Theseus): every 20 steps for 200 steps, 10% of the living cells, chosen at
    random, die (the tissue regrows them); the intact worm still remembers
H26 long delay: the memory survives 300 steps without any presentation (training used 4-16)
H27 minimal anterior fragment: only sites 12-19 (8 cells: the neck and anterior trunk) are kept;
    after 64 steps of regrowth the whole new worm remembers
H28 minimal posterior fragment: only sites 24-31 are kept; after 64 steps the new worm remembers
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch

from . import body, life, provenance, stats
from . import experiments as X
from . import tissue as T
from .train import load

N = X.N
RULES = ["E0", "E1", "S0", "S1", "B0", "B1", "F0", "F1"]
SEEDS = {"H25": 6025, "H26": 6026, "H27": 6027, "H28": 6028}
TURNOVER_EVERY, TURNOVER_STEPS, TURNOVER_FRACTION = 20, 200, 0.10
LONG_DELAY, FRAGMENT_REGEN = 300, 64


def turnover_timeline(n: int, seed: int) -> life.Timeline:
    g = torch.Generator().manual_seed(seed)
    tl = life.Timeline(n)
    for _ in range(TURNOVER_STEPS // TURNOVER_EVERY):
        u = torch.rand(n, 1, T.L, generator=g)
        tl.at(lambda x, u=u: x * (u >= TURNOVER_FRACTION).float())
        tl.free(TURNOVER_EVERY)
    return tl


def h25(p, n=N):
    co = X.cohort(SEEDS["H25"], n)
    P, U = X.twins(p, co)
    P2, U2 = P.play(turnover_timeline(n, 1))[0], U.play(turnover_timeline(n, 1))[0]
    m = X.di(P2) - X.di(U2)
    sc = body.anatomy_scores(P2.x)
    return {"test": stats.paired(m), "iou_after": float(sc["iou"].mean()), "region_acc_after": float(sc["region_acc"].mean())}


def h26(p, n=N):
    co = X.cohort(SEEDS["H26"], n)
    P, U = X.twins(p, co)
    m = X.di(P.free(LONG_DELAY)) - X.di(U.free(LONG_DELAY))
    return {"test": stats.paired(m)}


def fragment(p, seed, lo, hi, n=N):
    co = X.cohort(seed, n)
    P, U = X.twins(p, co)
    k = ((torch.arange(T.L) >= lo) & (torch.arange(T.L) < hi)).float()
    Pr, Ur = P.with_x(P.x * k).free(FRAGMENT_REGEN), U.with_x(U.x * k).free(FRAGMENT_REGEN)
    m = X.di(Pr) - X.di(Ur)
    sc = body.anatomy_scores(Pr.x)
    return {"test": stats.paired(m), "iou": float(sc["iou"].mean()), "region_acc": float(sc["region_acc"].mean()),
            "head_cells": float(sc["head_cells"].mean())}


def h27(p, n=N):
    return fragment(p, SEEDS["H27"], 12, 20, n)


def h28(p, n=N):
    return fragment(p, SEEDS["H28"], 24, 32, n)


CONFIRMATORY = {"H25": h25, "H26": h26, "H27": h27, "H28": h28}


def run_all(rule: str, n=N, log=print) -> dict:
    torch.set_num_threads(1)
    root = pathlib.Path(provenance.ROOT)
    w = root / "weights" / f"rule_{rule}.json"
    p = load(str(w))["tensors"]
    prereg = root / "prereg" / "PREREGISTRATION_v5.json"
    res = {"rule": rule, "weights_sha256": provenance.sha256_file(w),
           "prereg_sha256": provenance.sha256_file(prereg) if prereg.exists() else None,
           "provenance": provenance.stamp(), "n": n, "alpha": X.ALPHA, "sesoi": X.SESOI}
    with torch.no_grad():
        for k, fn in CONFIRMATORY.items():
            t0 = time.time()
            res[k] = fn(p, n)
            log(f"[{rule}] {k} {time.time() - t0:.0f}s mean {res[k]['test']['mean']:+.3f} p {res[k]['test']['p']:.3g}", flush=True)
    hm = stats.holm({h: res[h]["test"]["p"] for h in CONFIRMATORY}, X.ALPHA)
    res["verdicts"] = {h: {**v, "mean": res[h]["test"]["mean"], "sesoi_met": abs(res[h]["test"]["mean"]) >= X.SESOI,
                           "supported": bool(v["reject"] and abs(res[h]["test"]["mean"]) >= X.SESOI)} for h, v in hm.items()}
    return res


if __name__ == "__main__":
    import sys
    from . import prereg
    ok, bad = prereg.verify("5")
    print("v0.5 lock verifies: CONFIRMATORY" if ok else f"v0.5 lock does NOT verify {bad}", flush=True)
    r = run_all(sys.argv[1])
    r["confirmatory"] = ok
    pathlib.Path(f"results/v5_{sys.argv[1]}.json").write_text(json.dumps(r, indent=1, default=float))
    for h, v in r["verdicts"].items():
        print(f"{h}: mean {v['mean']:+.3f}  p_holm {v['p_adj']:.3g}  {'SUPPORTED' if v['supported'] else 'not supported'}")
