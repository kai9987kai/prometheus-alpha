"""Prometheus-alpha v0.4 experiments (prereg/PREREGISTRATION_v4.json). No pilots.

H19 fission, trained split: F rules' posterior half (cut at site 20) regrows a head that remembers
H20 fission, held-out split: the same when the cut is at site 24 (a smaller posterior piece, never trained)
H21 distributed copy: in INTACT trained worms, the hidden channels of sites 16-35 (far from any
    neck), copied into an untrained twin, make the twin's regrown head remember after decapitation,
    more in F rules than in their S sibling
H22 F rules still remember through complete decapitation (the S ability is kept)
H23 a universal code? A rule's v0.2 compiled patterns written into the OTHER S rule's untrained
    headless bodies write the intended memory
H24 single-cell memory: a pattern designed at one site (14) writes a memory
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch

from . import body, provenance, stats, v2, v3
from . import experiments as X
from .train import load

N = X.N
SEEDS = {"H19": 5019, "H20": 5020, "H21": 5021, "H22": 5022, "H23": 5023, "H24_design": 5240, "H24_test": 5241}
HIDDEN = v2.HIDDEN
REGEN_SPLIT = {20: 48, 24: 64}


def posterior_memory(p, seed, at, n=N):
    co = X.cohort(seed, n)
    P, U = X.twins(p, co)
    keep = (torch.arange(40) >= at).float()
    Pr, Ur = P.with_x(P.x * keep).free(REGEN_SPLIT[at]), U.with_x(U.x * keep).free(REGEN_SPLIT[at])
    m = X.di(Pr) - X.di(Ur)
    sc = body.anatomy_scores(Pr.x)
    return m, {"iou": float(sc["iou"].mean()), "region_acc": float(sc["region_acc"].mean()), "head_cells": float(sc["head_cells"].mean())}


def h19(p, n=N):
    m, a = posterior_memory(p, SEEDS["H19"], 20, n)
    return {"test": stats.paired(m), "anatomy": a}


def h20(p, n=N):
    m, a = posterior_memory(p, SEEDS["H20"], 24, n)
    return {"test": stats.paired(m), "anatomy": a}


def distributed(p, n=N, seed=SEEDS["H21"]):
    co = X.cohort(seed, n)
    P, U = X.twins(p, co)
    x = U.x.clone()
    x[:, HIDDEN, 16:36] = P.x[:, HIDDEN, 16:36]
    base = X.di(U.cut("head")[0].free(X.REGEN))
    return X.di(U.with_x(x).cut("head")[0].free(X.REGEN)) - base


def h21(p_f, p_s, n=N):
    f, s = distributed(p_f, n), distributed(p_s, n)
    return {"test": stats.paired(f - s), "F_memory": float(f.mean()), "S_memory": float(s.mean())}


def h22(p, n=N):
    co = X.cohort(SEEDS["H22"], n)
    P, U = X.twins(p, co)
    m = X.di(P.cut("head")[0].free(X.REGEN)) - X.di(U.cut("head")[0].free(X.REGEN))
    return {"test": stats.paired(m)}


def h23(p, other_patterns, n=N):
    w = v2._naive_decapitated(p, SEEDS["H23"], n)
    pa, pb = torch.tensor(other_patterns["A"]), torch.tensor(other_patterns["B"])
    wa, wb = v2._written(p, w, pa).numpy(), v2._written(p, w, pb).numpy()
    return {"test": stats.paired(0.5 * (wa - wb)), "written_A": float(wa.mean()), "written_B": float(wb.mean())}


def h24(p, n=N):
    with torch.enable_grad():
        pa = v3.compile_sparse(p, 0, sites=slice(14, 15), seed=SEEDS["H24_design"])
        pb = v3.compile_sparse(p, 1, sites=slice(14, 15), seed=SEEDS["H24_design"])
    w = v2._naive_decapitated(p, SEEDS["H24_test"], n)
    wa, wb = v2._written(p, w, pa).numpy(), v2._written(p, w, pb).numpy()
    return {"test": stats.paired(0.5 * (wa - wb)), "written_A": float(wa.mean()), "written_B": float(wb.mean()),
            "pattern_A_site14": pa[:, 14].tolist(), "pattern_B_site14": pb[:, 14].tolist()}


def run_all(rule: str, n=N, log=print) -> dict:
    torch.set_num_threads(1)
    root = pathlib.Path(provenance.ROOT)
    W = lambda r: load(str(root / "weights" / f"rule_{r}.json"))["tensors"]
    p = W(rule)
    prereg = root / "prereg" / "PREREGISTRATION_v4.json"
    res = {"rule": rule, "weights_sha256": provenance.sha256_file(root / "weights" / f"rule_{rule}.json"),
           "prereg_sha256": provenance.sha256_file(prereg) if prereg.exists() else None,
           "provenance": provenance.stamp(), "n": n, "alpha": X.ALPHA, "sesoi": X.SESOI}
    fam, s = rule[0], rule[1:]
    fns = {}
    if fam in "FS":
        fns.update(H19=lambda: h19(p, n), H20=lambda: h20(p, n))
    if fam == "F":
        fns.update(H21=lambda: h21(p, W("S" + s), n), H22=lambda: h22(p, n))
    if fam == "S":
        other = json.loads((root / "results" / f"v2_S{1 - int(s)}.json").read_text())["H12"]["patterns"]
        fns["H23"] = lambda: h23(p, other, n)
    fns["H24"] = lambda: h24(p, n)
    with torch.no_grad():
        for k, fn in fns.items():
            t0 = time.time()
            res[k] = fn()
            log(f"[{rule}] {k} {time.time() - t0:.0f}s mean {res[k]['test']['mean']:+.3f} p {res[k]['test']['p']:.3g}", flush=True)
    hm = stats.holm({h: res[h]["test"]["p"] for h in fns}, X.ALPHA)
    res["verdicts"] = {h: {**v, "mean": res[h]["test"]["mean"], "sesoi_met": abs(res[h]["test"]["mean"]) >= X.SESOI,
                           "supported": bool(v["reject"] and abs(res[h]["test"]["mean"]) >= X.SESOI)} for h, v in hm.items()}
    return res


if __name__ == "__main__":
    import sys
    from . import prereg
    ok, bad = prereg.verify("4")
    print("v0.4 lock verifies: CONFIRMATORY" if ok else f"v0.4 lock does NOT verify {bad}", flush=True)
    r = run_all(sys.argv[1])
    r["confirmatory"] = ok
    pathlib.Path(f"results/v4_{sys.argv[1]}.json").write_text(json.dumps(r, indent=1, default=float))
    for h, v in r["verdicts"].items():
        print(f"{h}: mean {v['mean']:+.3f}  p_holm {v['p_adj']:.3g}  {'SUPPORTED' if v['supported'] else 'not supported'}")
