"""Prometheus-alpha v0.3 experiments (prereg/PREREGISTRATION_v3.json). No pilots were run for
v0.3: every prediction below is a guess made from the v0.1 and v0.2 results alone.

H14 memory surgery: writing the compiled pattern for the OTHER odour into a trained worm's
    decapitated body overwrites its own memory in the regrown head
H15 sparse compiler: a pattern designed at only the 4 wound-edge sites (12-15) writes a memory
H16 engram decoder: a linear readout of the hidden channels (fit on half the worms) tells which
    odour a decapitated body learned better at the wound edge than in the rest of the trunk
H17 balanced selection: B rules (S + two-amputation training) keep the S rule's disfavoured
    odour better at cycle 4 than their S sibling
H18 fission: cut a trained worm in half at site 20; the posterior half, which regrows a head,
    trunk and the missing anterior, remembers
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch

from . import life, provenance, stats, v2
from . import experiments as X
from . import tissue as T
from .train import load

N = X.N
SEEDS = {"H14": 4014, "H15_design": 4150, "H15_test": 4151, "H16": 4016, "H17": 4017, "H18": 4018}
EDGE, REST = slice(12, 16), slice(16, 28)
HIDDEN = v2.HIDDEN


def _pref(w: X.Worms) -> np.ndarray:
    return w.test(csplus=np.zeros(w.co.n, int))[1]["di"]


def _write(w: X.Worms, pattern: torch.Tensor) -> X.Worms:
    x = w.x.clone()
    live = (x[:, T.ALPHA:T.ALPHA + 1] > 0.1).float()
    x = torch.cat([x[:, :T.HIDDEN0], x[:, T.HIDDEN0:] + pattern[None] * live], 1)
    return w.with_x(x)


# ----------------------------------------------------------------------------- H14 surgery

def h14_surgery(p, patterns: dict, n=N) -> dict:
    """A-trained worms get pattern B, B-trained worms get pattern A. Overwrite per worm =
    (pref without surgery - pref with surgery) / 2 * sign, so 1 = a full switch of odour."""
    A = X.Worms.found(p, X.cohort(SEEDS["H14"] * 10 + 1, n, 0)).learn("paired").cut("head")[0]
    B = X.Worms.found(p, X.cohort(SEEDS["H14"] * 10 + 2, n, 1)).learn("paired").cut("head")[0]
    pa, pb = torch.tensor(patterns["A"]), torch.tensor(patterns["B"])
    a0, a1 = _pref(A.free(X.REGEN)), _pref(_write(A, pb).free(X.REGEN))
    b0, b1 = _pref(B.free(X.REGEN)), _pref(_write(B, pa).free(X.REGEN))
    over = 0.5 * ((a0 - a1) + (b1 - b0)) / 2
    return {"test": stats.paired(over), "A_trained": {"no_surgery": float(a0.mean()), "pattern_B": float(a1.mean())},
            "B_trained": {"no_surgery": float(b0.mean()), "pattern_A": float(b1.mean())},
            "switched_fraction": float(((a1 < 0).mean() + (b1 > 0).mean()) / 2)}


# ----------------------------------------------------------------------------- H15 sparse compiler

def compile_sparse(p, odour: int, sites=EDGE, iters=150, lr=0.05, n=32, l2=0.01, seed=SEEDS["H15_design"]):
    torch.manual_seed(seed)
    w = v2._naive_decapitated(p, seed, n)
    mask = torch.zeros(len(HIDDEN), T.L)
    mask[:, sites] = 1
    pat = torch.zeros(len(HIDDEN), T.L, requires_grad=True)
    opt = torch.optim.Adam([pat], lr=lr)
    sign = 1.0 if odour == 0 else -1.0
    for _ in range(iters):
        pref = v2._written(p, w, pat * mask)
        loss = ((sign * pref - 1) ** 2).mean() + l2 * (pat ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (pat * mask).detach()


def h15_sparse(p, n=N) -> dict:
    with torch.enable_grad():
        pa, pb = compile_sparse(p, 0), compile_sparse(p, 1)
    w = v2._naive_decapitated(p, SEEDS["H15_test"], n)
    wa, wb = v2._written(p, w, pa).numpy(), v2._written(p, w, pb).numpy()
    written = 0.5 * (wa - wb)
    return {"test": stats.paired(written), "written_A": float(wa.mean()), "written_B": float(wb.mean()),
            "sites": [EDGE.start, EDGE.stop], "pattern_rms_on_sites": float(pa[:, EDGE].pow(2).mean().sqrt()),
            "patterns": {"A": pa.tolist(), "B": pb.tolist()}}


# ----------------------------------------------------------------------------- H16 decoder

def _fit(Xtr, ytr, lam=1e-2):
    Xb = np.c_[Xtr, np.ones(len(Xtr))]
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ ytr)


def _predict(wt, X):
    return np.c_[X, np.ones(len(X))] @ wt


def h16_decoder(p, n=N) -> dict:
    """Ridge readout of +-1 (which odour was the CS+) from the 10 hidden channels of one cell,
    shared across sites, fit on half the decapitated paired worms at the edge sites, and scored
    on the other half at every site. Per held-out worm: correct at the edge (majority over edge
    cells) minus correct in the rest of the trunk (majority over rest cells)."""
    co = X.cohort(SEEDS["H16"], n)
    P = X.Worms.found(p, co).learn("paired")
    intact = P.x[:, HIDDEN].numpy()
    Pc = P.cut("head")[0]
    H = Pc.x[:, HIDDEN].numpy()                       # (n, 10, L)
    alive = (Pc.x[:, T.ALPHA] > 0.1).numpy()
    y = np.where(co.csplus == 0, 1.0, -1.0)
    tr, te = np.arange(n) < n // 2, np.arange(n) >= n // 2
    Xtr = np.concatenate([H[i][:, alive[i] & (np.arange(T.L) < EDGE.stop)].T for i in np.where(tr)[0]])
    ytr = np.concatenate([np.full((alive[i] & (np.arange(T.L) < EDGE.stop)).sum(), y[i]) for i in np.where(tr)[0]])
    wt = _fit(Xtr, ytr)
    per_site = []
    for s in range(T.L):
        ok = [np.sign(_predict(wt, H[i][:, s][None]))[0] == y[i] for i in np.where(te)[0] if alive[i, s]]
        per_site.append(float(np.mean(ok)) if ok else None)
        intact_ok = [np.sign(_predict(wt, intact[i][:, s][None]))[0] == y[i] for i in np.where(te)[0]]
        per_site[-1] = {"site": s, "decapitated": per_site[-1], "intact": float(np.mean(intact_ok))}

    def region_correct(i, sl):
        cells = [s for s in range(sl.start, sl.stop) if alive[i, s]]
        if not cells:
            return 0.5
        return float(np.sign(_predict(wt, H[i][:, cells].T).mean()) == y[i])
    d = np.array([region_correct(i, EDGE) - region_correct(i, REST) for i in np.where(te)[0]])
    return {"test": stats.paired(d), "edge_accuracy": float(np.mean([region_correct(i, EDGE) for i in np.where(te)[0]])),
            "rest_accuracy": float(np.mean([region_correct(i, REST) for i in np.where(te)[0]])),
            "per_site": per_site, "decoder": {"w": wt[:-1].tolist(), "b": float(wt[-1])}}


# ----------------------------------------------------------------------------- H17 balance

def cycle4_by_odour(p, seed, n=N, cycles=4):
    co = X.cohort(seed, n)
    P, U = X.twins(p, co)
    for _ in range(cycles):
        P, U = P.cut("head")[0].free(X.REGEN), U.cut("head")[0].free(X.REGEN)
        P, a = P.test()
        U, b = U.test()
    d = a["di"] - b["di"]
    return d, co.csplus


def h17_balance(p_b, p_s, favoured: int, n=N) -> dict:
    db, cb = cycle4_by_odour(p_b, SEEDS["H17"], n)
    ds, cs = cycle4_by_odour(p_s, SEEDS["H17"], n)
    ob, os_ = db[cb != favoured], ds[cs != favoured]     # same cohort, same worms: paired
    d = ob - os_
    return {"test": stats.paired(d), "B_other": float(ob.mean()), "S_other": float(os_.mean()),
            "B_favoured": float(db[cb == favoured].mean()), "S_favoured": float(ds[cs == favoured].mean())}


# ----------------------------------------------------------------------------- H18 fission

def h18_fission(p, n=N, at=20, regen=48) -> dict:
    co = X.cohort(SEEDS["H18"], n)
    P, U = X.twins(p, co)

    def halves(w):
        front = (torch.arange(T.L) < at).float()
        return w.with_x(w.x * front), w.with_x(w.x * (1 - front))
    Pa, Pp = halves(P)
    Ua, Up = halves(U)
    post = X.di(Pp.free(regen)) - X.di(Up.free(regen))
    ant = X.di(Pa.free(regen)) - X.di(Ua.free(regen))
    from . import body
    sc_p = body.anatomy_scores(Pp.free(regen).x)
    sc_a = body.anatomy_scores(Pa.free(regen).x)
    return {"test": stats.paired(post), "posterior_half_memory": float(post.mean()), "anterior_half_memory": float(ant.mean()),
            "anterior_test": stats.paired(ant, n_boot=2000),
            "posterior_iou": float(sc_p["iou"].mean()), "anterior_iou": float(sc_a["iou"].mean())}


CONFIRMATORY = ["H14", "H15", "H16", "H17", "H18"]


def run_all(rule: str, out: str, n=N, log=print) -> dict:
    torch.set_num_threads(1)
    root = pathlib.Path(provenance.ROOT)
    p = load(str(root / "weights" / f"rule_{rule}.json"))["tensors"]
    r2 = json.loads((root / "results" / f"v2_{rule}.json").read_text())
    prereg = root / "prereg" / "PREREGISTRATION_v3.json"
    res = {"rule": rule, "weights_sha256": provenance.sha256_file(root / "weights" / f"rule_{rule}.json"),
           "prereg_sha256": provenance.sha256_file(prereg) if prereg.exists() else None,
           "provenance": provenance.stamp(), "n": n, "alpha": X.ALPHA, "sesoi": X.SESOI}
    fns = {"H14": lambda: h14_surgery(p, r2["H12"]["patterns"], n), "H15": lambda: h15_sparse(p, n),
           "H16": lambda: h16_decoder(p, n), "H18": lambda: h18_fission(p, n)}
    if rule.startswith("S"):
        pb = load(str(root / "weights" / f"rule_B{rule[1:]}.json"))["tensors"]
        fns["H17"] = lambda: h17_balance(pb, p, v2.FAVOURED[rule], n)
    with torch.no_grad():
        for k, fn in fns.items():
            t0 = time.time()
            res[k] = fn()
            log(f"[{rule}] {k} {time.time() - t0:.0f}s mean {res[k]['test']['mean']:+.3f} p {res[k]['test']['p']:.3g}", flush=True)
    ps = {h: res[h]["test"]["p"] for h in CONFIRMATORY if h in res}
    hm = stats.holm(ps, X.ALPHA)
    res["verdicts"] = {h: {**v, "mean": res[h]["test"]["mean"], "sesoi_met": abs(res[h]["test"]["mean"]) >= X.SESOI,
                           "supported": bool(v["reject"] and abs(res[h]["test"]["mean"]) >= X.SESOI)} for h, v in hm.items()}
    pathlib.Path(out).write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    import sys
    from . import prereg
    ok, bad = prereg.verify("3")
    print("v0.3 lock verifies: CONFIRMATORY" if ok else f"v0.3 lock does NOT verify {bad}", flush=True)
    r = run_all(sys.argv[1], f"results/v3_{sys.argv[1]}.json")
    r["confirmatory"] = ok
    pathlib.Path(f"results/v3_{sys.argv[1]}.json").write_text(json.dumps(r, indent=1, default=float))
    for h, v in r["verdicts"].items():
        print(f"{h}: mean {v['mean']:+.3f}  p_holm {v['p_adj']:.3g}  {'SUPPORTED' if v['supported'] else 'not supported'}")
