"""Prometheus-alpha v0.7 experiments (prereg/PREREGISTRATION_v7.json): is there a causal trade-off
between tolerating noise and surviving cell loss?

For each lineage s in {0, 1}, the T, N and C siblings (train_v7.py) are measured on the SAME
cohort of worms (same founders, schedules and update masks), so every comparison is paired per
worm. Both measurements use held-out seeds and stress settings:

* turnover memory: after learning, 10% of the cells die at random every 20 steps for 200 steps
  (kill-mask seed 2, cohort seed 8025); then the memory test;
* noise tolerance: after learning, Gaussian noise with sigma 1.5 (noise seed 3) is added to the
  hidden channels of the head sites 4-11 of intact worms (cohort seed 8030); 12 steps; then the test.

H31 turnover training works:  turnover M(T) - M(C) > 0
H32 noise training works:     noise M(N) - M(C) > 0
H33 trade-off (one way):      noise M(C) - M(T) > 0, hardening against cell loss costs noise tolerance
H34 trade-off (other way):    turnover M(C) - M(N) > 0, hardening against noise costs survival of cell loss
"""

from __future__ import annotations

import json
import pathlib
import time

import torch

from . import provenance, stats, v2, v5
from . import experiments as X
from . import tissue as T
from .train import load

N = X.N
SEEDS = {"turnover": 8025, "noise": 8030}
NOISE_SIGMA, HEAD = 1.5, slice(4, 12)


def turnover_memory(p, n=N):
    co = X.cohort(SEEDS["turnover"], n)
    P, U = X.twins(p, co)
    return X.di(P.play(v5.turnover_timeline(n, 2))[0]) - X.di(U.play(v5.turnover_timeline(n, 2))[0])


def noise_memory(p, n=N):
    co = X.cohort(SEEDS["noise"], n)
    P, U = X.twins(p, co)
    eps = torch.randn(n, len(v2.HIDDEN), T.L, generator=torch.Generator().manual_seed(3))
    x = P.x.clone()
    x[:, v2.HIDDEN, HEAD] = x[:, v2.HIDDEN, HEAD] + NOISE_SIGMA * eps[:, :, HEAD]
    return X.di(P.with_x(x).free(12)) - X.di(U)


def run(log=print) -> dict:
    torch.set_num_threads(4)
    root = pathlib.Path(provenance.ROOT)
    prereg = root / "prereg" / "PREREGISTRATION_v7.json"
    res = {"prereg_sha256": provenance.sha256_file(prereg) if prereg.exists() else None,
           "provenance": provenance.stamp(), "n": N, "alpha": X.ALPHA, "sesoi": X.SESOI, "rules": {}, "tests": {}}
    with torch.no_grad():
        for s in (0, 1):
            m = {}
            for fam in ("S", "C", "T", "N"):
                w = root / "weights" / f"rule_{fam}{s}.json"
                p = load(str(w))["tensors"]
                t0 = time.time()
                m[fam] = {"turnover": turnover_memory(p), "noise": noise_memory(p)}
                res["rules"][f"{fam}{s}"] = {"weights_sha256": provenance.sha256_file(w),
                                             "turnover_memory": float(m[fam]["turnover"].mean()),
                                             "noise_memory": float(m[fam]["noise"].mean())}
                log(f"[{fam}{s}] turnover {m[fam]['turnover'].mean():+.3f} noise {m[fam]['noise'].mean():+.3f} ({time.time() - t0:.0f}s)", flush=True)
            tests = {"H31": m["T"]["turnover"] - m["C"]["turnover"], "H32": m["N"]["noise"] - m["C"]["noise"],
                     "H33": m["C"]["noise"] - m["T"]["noise"], "H34": m["C"]["turnover"] - m["N"]["turnover"]}
            summ = {h: stats.paired(d) for h, d in tests.items()}
            hm = stats.holm({h: v["p"] for h, v in summ.items()}, X.ALPHA)
            for h, v in summ.items():
                res["tests"][f"{h}_{s}"] = {"test": v, **hm[h], "mean": v["mean"],
                                            "supported": bool(hm[h]["reject"] and v["mean"] >= X.SESOI)}
    return res


if __name__ == "__main__":
    from . import prereg
    ok, bad = prereg.verify("7")
    print("v0.7 lock verifies: CONFIRMATORY" if ok else f"v0.7 lock does NOT verify {bad}", flush=True)
    r = run()
    r["confirmatory"] = ok
    pathlib.Path("results/v7.json").write_text(json.dumps(r, indent=1, default=float))
    for k, v in r["tests"].items():
        print(f"{k}: {v['mean']:+.3f} p_holm {v['p_adj']:.3g} {'SUPPORTED' if v['supported'] else 'not supported'}")
