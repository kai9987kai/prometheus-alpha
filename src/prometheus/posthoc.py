"""Post-hoc analyses (NOT preregistered), written after the confirmatory results were seen.

The confirmatory chimera and transfer results (H4, H4b, H5) were significant, but their
per-graft preferences showed that in each S rule one odour wins conflicts whatever tissue
carries it. This module quantifies that, and splits survival and Promethean cycles by which
odour was the CS+. It reuses the locked experiment code without changing it, runs on fresh
cohort seeds, and writes ``results/posthoc_<rule>.json``.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from . import experiments as X
from .provenance import stamp, sha256_file
from .train import load

SEEDS = {"survival_by_cue": 2002, "promethean_by_cue": 2006}


def _split(m: np.ndarray, csplus: np.ndarray) -> dict:
    return {"A": float(m[csplus == 0].mean()), "B": float(m[csplus == 1].mean())}


def from_results(res: dict) -> dict:
    """Numbers derived from a confirmatory results file (no new simulation)."""
    pre, post = res["H4"]["preference"], res["H4b"]["preference"]
    tr = res["H5"]["preference"]
    return {
        "chimera_cue_bias_intact": 0.5 * (pre["A|B"] + pre["B|A"]),
        "chimera_cue_bias_regrown": 0.5 * (post["A|B"] + post["B|A"]),
        "transfer_fraction_A": tr["N|A"] / tr["A|A"] if tr["A|A"] else float("nan"),
        "transfer_fraction_B": tr["N|B"] / tr["B|B"] if tr["B|B"] else float("nan"),
        "note": "cue bias = mean of R(A) - R(B) over the two conflicting grafts: > 0 favours odour A, < 0 odour B, whichever tissue carried it",
    }


def survival_by_cue(p, n=X.N) -> dict:
    co = X.cohort(SEEDS["survival_by_cue"], n)
    P, U = X.twins(p, co)
    intact = X.di(P) - X.di(U)
    regrown = X.di(P.cut("head")[0].free(X.REGEN)) - X.di(U.cut("head")[0].free(X.REGEN))
    return {"intact": _split(intact, co.csplus), "regrown": _split(regrown, co.csplus)}


def promethean_by_cue(p, n=X.N, cycles=X.CYCLES) -> dict:
    co = X.cohort(SEEDS["promethean_by_cue"], n)
    P, U = X.twins(p, co)
    out = []
    for k in range(cycles):
        P, U = P.cut("head")[0].free(X.REGEN), U.cut("head")[0].free(X.REGEN)
        P, a = P.test()
        U, b = U.test()
        out.append({"cycle": k + 1, **_split(a["di"] - b["di"], co.csplus)})
    return {"cycles": out}


def run(rule: str, root: pathlib.Path) -> dict:
    torch.set_num_threads(1)
    weights = root / "weights" / f"rule_{rule}.json"
    res = json.loads((root / "results" / f"rule_{rule}.json").read_text())
    p = load(str(weights))["tensors"]
    with torch.no_grad():
        out = {"rule": rule, "posthoc": True, "weights_sha256": sha256_file(weights), "provenance": stamp(),
               "derived": from_results(res), "survival_by_cue": survival_by_cue(p),
               "promethean_by_cue": promethean_by_cue(p)}
    (root / "results" / f"posthoc_{rule}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import sys
    from .provenance import ROOT
    for r in sys.argv[1:] or ["E0", "E1", "S0", "S1"]:
        o = run(r, ROOT)
        print(r, json.dumps({k: o[k] for k in ("derived", "survival_by_cue", "promethean_by_cue")}, default=lambda v: round(v, 3)))
