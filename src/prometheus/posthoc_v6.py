"""Post hoc (NOT preregistered), written after v0.6 H30 failed: is turnover robustness predicted
by the basin of the memory where turnover acts, in the HEAD of intact worms, rather than at the
neck of decapitated bodies (the preregistered measure)? Writes results/posthoc_v6.json."""

import json
import pathlib

import numpy as np
import torch

from . import experiments as X
from . import tissue as T
from . import v2, v6
from .provenance import ROOT
from .train import load

HEAD = slice(4, 12)


def head_basin(p, n=64):
    co = X.cohort(8030, n)
    P, U = X.twins(p, co)
    base = X.di(U)
    noise = torch.randn(n, len(v2.HIDDEN), T.L, generator=torch.Generator().manual_seed(2))
    curve = []
    for s in v6.SIGMAS:
        x = P.x.clone()
        x[:, v2.HIDDEN, HEAD] = x[:, v2.HIDDEN, HEAD] + s * noise[:, :, HEAD]
        curve.append(float((X.di(P.with_x(x).free(12)) - base).mean()))
    crit = next((s for s, m in zip(v6.SIGMAS, curve) if m < curve[0] / 2), v6.SIGMAS[-1] * 1.5)
    return {"memory": curve, "sigma_star": crit, "area": float(np.trapezoid(curve, v6.SIGMAS) / curve[0]) if curve[0] > 0 else 0.0}


if __name__ == "__main__":
    torch.set_num_threads(4)
    rules = v6.BASIN_RULES
    out = {"posthoc": True, "basins": {}}
    with torch.no_grad():
        for r in rules:
            out["basins"][r] = head_basin(load(str(ROOT / f"weights/rule_{r}.json"))["tensors"])
            print(r, out["basins"][r]["sigma_star"], round(out["basins"][r]["area"], 2), [round(v, 2) for v in out["basins"][r]["memory"]], flush=True)
    turn = [json.loads((ROOT / f"results/v5_{r}.json").read_text())["H25"]["test"]["mean"] for r in rules]
    for key in ("sigma_star", "area"):
        rho, p = v6.spearman_exact([out["basins"][r][key] for r in rules], turn)
        out[f"spearman_{key}_vs_turnover"] = {"rho": rho, "p_one_sided": p}
        print(key, round(rho, 3), round(p, 4))
    (ROOT / "results/posthoc_v6.json").write_text(json.dumps(out, indent=1))
