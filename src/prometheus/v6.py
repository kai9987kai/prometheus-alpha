"""Prometheus-alpha v0.6 experiments (prereg/PREREGISTRATION_v6.json). No new rules, no pilots.

H29 transgenerational memory: a trained F worm is split at site 20 and the posterior half regrows
    (48 steps); the regrown worm is split again, and again. The third generation remembers.
    (F rules were trained on one split.) S rules, whose posterior halves forget, are the control.
H30 an attractor basin predicts robustness: for each S, B and F rule, the critical noise sigma*
    (smallest Gaussian perturbation of the neck cells' hidden channels, right after
    decapitation, that halves the regrown head's memory) is positively rank-correlated with the
    rule's v0.5 turnover memory (H25) across the six rules.
"""

from __future__ import annotations

import json
import pathlib
import time
from itertools import permutations

import numpy as np
import torch

from . import body, provenance, stats, v2
from . import experiments as X
from . import tissue as T
from .train import load

N = X.N
SEEDS = {"H29": 7029, "H30": 7030}
GENERATIONS, SPLIT, SPLIT_REGEN = 3, 20, 48
SIGMAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
NECK = slice(12, 20)
BASIN_RULES = ["S0", "S1", "B0", "B1", "F0", "F1"]


def generations(p, n=N):
    co = X.cohort(SEEDS["H29"], n)
    T_needed = X.GROW + 60 + X.DELAY + GENERATIONS * (SPLIT_REGEN + 24) + 24
    base = X.Worms(p, co, T.founder(n), 0, T.fire_masks(co.mask_seeds, T_needed))
    P, U = base.learn("paired"), base.learn("unpaired")
    keep = (torch.arange(T.L) >= SPLIT).float()
    per, anat = [], []
    for g in range(GENERATIONS):
        P, U = P.with_x(P.x * keep).free(SPLIT_REGEN), U.with_x(U.x * keep).free(SPLIT_REGEN)
        P, a = P.test()
        U, b = U.test()
        per.append(a["di"] - b["di"])
        anat.append(float(body.anatomy_scores(P.x)["iou"].mean()))
    return per, anat


def h29(p, n=N):
    per, anat = generations(p, n)
    return {"test": stats.paired(per[-1]), "by_generation": [float(d.mean()) for d in per], "iou_by_generation": anat}


def basin(p, n=64):
    """Memory of regrown heads vs noise sigma added to the neck's hidden channels (decapitated,
    trained bodies; noise per cell and channel, seed fixed)."""
    Pc, Uc = X._decapitated_twins(p, SEEDS["H30"], n)
    base = X.di(Uc.free(X.REGEN))
    g = torch.Generator().manual_seed(1)
    noise = torch.randn(n, len(v2.HIDDEN), T.L, generator=g)
    curve = []
    for s in SIGMAS:
        x = Pc.x.clone()
        live = (x[:, T.ALPHA] > 0.1).float()[:, None]
        x[:, v2.HIDDEN, NECK] = x[:, v2.HIDDEN, NECK] + s * (noise * live)[:, :, NECK]
        curve.append(float((X.di(Pc.with_x(x).free(X.REGEN)) - base).mean()))
    half = curve[0] / 2
    crit = next((s for s, m in zip(SIGMAS, curve) if m < half), float(SIGMAS[-1]) * 1.5)
    return {"sigmas": SIGMAS, "memory": curve, "sigma_star": crit}


def spearman_exact(x, y) -> tuple[float, float]:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    obs = np.corrcoef(rx, ry)[0, 1]
    null = [np.corrcoef(rx, np.array(perm))[0, 1] for perm in permutations(ry)]
    return float(obs), float(np.mean(np.array(null) >= obs - 1e-12))


def run(out_dir: str = "results", log=print) -> dict:
    torch.set_num_threads(1)
    root = pathlib.Path(provenance.ROOT)
    W = lambda r: load(str(root / "weights" / f"rule_{r}.json"))["tensors"]
    prereg = root / "prereg" / "PREREGISTRATION_v6.json"
    res = {"prereg_sha256": provenance.sha256_file(prereg) if prereg.exists() else None,
           "provenance": provenance.stamp(), "n": N, "alpha": X.ALPHA, "sesoi": X.SESOI, "H29": {}, "basins": {}}
    with torch.no_grad():
        for r in ["S0", "S1", "F0", "F1"]:
            t0 = time.time()
            res["H29"][r] = h29(W(r))
            log(f"[{r}] H29 {time.time() - t0:.0f}s by generation {[round(v, 3) for v in res['H29'][r]['by_generation']]}", flush=True)
        for r in BASIN_RULES:
            res["basins"][r] = basin(W(r))
            log(f"[{r}] basin sigma* {res['basins'][r]['sigma_star']} curve {[round(v, 2) for v in res['basins'][r]['memory']]}", flush=True)
    turnover = [json.loads((root / "results" / f"v5_{r}.json").read_text())["H25"]["test"]["mean"] for r in BASIN_RULES]
    sig = [res["basins"][r]["sigma_star"] for r in BASIN_RULES]
    rho, p = spearman_exact(sig, turnover)
    res["H30"] = {"test": {"mean": rho, "p": p, "n": len(BASIN_RULES), "alternative": "greater", "statistic": "Spearman rho, exact permutation"},
                  "sigma_star": dict(zip(BASIN_RULES, sig)), "turnover_memory": dict(zip(BASIN_RULES, turnover))}
    ps = {f"H29_{r}": res["H29"][r]["test"]["p"] for r in res["H29"]}
    hm = stats.holm(ps, X.ALPHA)
    res["verdicts"] = {k: {**v, "mean": res["H29"][k[4:]]["test"]["mean"],
                           "supported": bool(v["reject"] and res["H29"][k[4:]]["test"]["mean"] >= X.SESOI)} for k, v in hm.items()}
    res["verdicts"]["H30"] = {"p": p, "mean": rho, "supported": bool(p <= X.ALPHA and rho > 0)}
    pathlib.Path(root / out_dir / "v6.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    from . import prereg
    ok, bad = prereg.verify("6")
    print("v0.6 lock verifies: CONFIRMATORY" if ok else f"v0.6 lock does NOT verify {bad}", flush=True)
    r = run()
    r["confirmatory"] = ok
    pathlib.Path("results/v6.json").write_text(json.dumps(r, indent=1, default=float))
    for h, v in r["verdicts"].items():
        print(f"{h}: {v['mean']:+.3f} p {v['p']:.3g} {'SUPPORTED' if v['supported'] else 'not supported'}")
