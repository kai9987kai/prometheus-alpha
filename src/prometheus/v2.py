"""Prometheus-alpha v0.2 experiments (prereg/PREREGISTRATION_v2.json).

v0.1 found that selected (S) rules keep a copy of a learned association in the hidden channels
of the body, and, post hoc, that each S rule holds one odour's memory as an attractor and the
other as metastable. v0.2 asks five new questions of the same four rules, on fresh seeds, with
the v0.1 engine and experiment code unchanged (imported, never edited):

H8  attractor (confirmatory replication of the v0.1 post-hoc finding): after 4 Promethean cycles
    the rule's favoured odour, named in advance (S0: B, S1: A), is remembered better
H9  extinction: presenting both odours with no shock after learning weakens the memory
H10 extinction through regrowth: a head regrown after extinction differs from the intact
    extinguished head (two-sided; the pilot showed the body over-extinguishes, inverting it)
H11 reversal through regrowth: after learning one odour then the other, the regrown head's
    memory of the first lesson differs from the intact head's (two-sided; the pilot showed the
    body keeps the newer lesson more completely)
H12 memory compiler: a hidden-channel pattern designed by gradient descent on training worms,
    written into an untrained worm's headless body, makes held-out worms' regrown heads
    remember an odour they never learned, better than the same pattern with its sites shuffled
H13 engram locus: implanting the trained body's hidden channels at only the 4 sites next to the
    wound (12-15) transfers more memory than implanting them at the rest of the trunk (16-27)
Exploratory: the full engram map (4-site windows), long Promethean runs (8 cycles).
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
EXT_BLOCKS = 2                     # extinction: 2 blocks of 48 steps with cues and no shock
FAVOURED = {"S0": 1, "S1": 0}      # v0.1 post hoc: S0 favours odour B, S1 odour A (fixed before v0.2 data)
SEEDS = {"H8": 3008, "H9": 3009, "H11": 3011, "H13": 3013, "H12_design": 3120, "H12_test": 3121,
         "engram_map": 3201, "long_cycles": 3202}
HIDDEN = X.GROUPS["hidden"]


# ----------------------------------------------------------------------------- helpers

def extinguish(w: X.Worms, blocks: int = EXT_BLOCKS) -> X.Worms:
    """Unreinforced presentations: the conditioning schedule with no shock, then the delay."""
    tl = life.Timeline(w.co.n)
    for b in range(blocks):
        tl.condition(w.co.csplus, "naive", np.random.default_rng([w.co.seed, 50 + b]))
    tl.free(X.DELAY)
    return w.play(tl)[0]


def wait(w: X.Worms, steps: int) -> X.Worms:
    return w.free(steps)


def relearn(w: X.Worms, csplus, pairing: str) -> X.Worms:
    tl = life.Timeline(w.co.n).condition(csplus, pairing, np.random.default_rng([w.co.seed, 77])).free(X.DELAY)
    return w.play(tl)[0]


def m_intact(P, U, csplus=None):
    return P.test(csplus)[1]["di"] - U.test(csplus)[1]["di"]


def m_regrown(P, U, csplus=None):
    Pr = P.cut("head")[0].free(X.REGEN)
    Ur = U.cut("head")[0].free(X.REGEN)
    return Pr.test(csplus)[1]["di"] - Ur.test(csplus)[1]["di"]


# ----------------------------------------------------------------------------- confirmatory

def h8_attractor(p, rule: str, n=N, cycles=X.CYCLES) -> dict:
    """Memory after 4 Promethean cycles, favoured-odour worms minus other-odour worms."""
    co = X.cohort(SEEDS["H8"], n)
    P, U = X.twins(p, co)
    per = []
    for _ in range(cycles):
        P, U = P.cut("head")[0].free(X.REGEN), U.cut("head")[0].free(X.REGEN)
        P, a = P.test()
        U, b = U.test()
        per.append(a["di"] - b["di"])
    fav = FAVOURED.get(rule)
    if fav is None:
        return {"applicable": False, "note": "E rules have no favoured odour and no surviving memory"}
    last = per[-1]
    f, o = last[co.csplus == fav], last[co.csplus != fav]
    # unpaired two-sample comparison: permutation of group labels
    rng = np.random.default_rng(0)
    obs = f.mean() - o.mean()
    pool = np.concatenate([f, o])
    null = np.array([(lambda z: z[:len(f)].mean() - z[len(f):].mean())(rng.permutation(pool)) for _ in range(10000)])
    pval = float(((null >= obs - 1e-12).sum() + 1) / (len(null) + 1))
    return {"applicable": True, "favoured": "AB"[fav],
            "test": {"mean": float(obs), "p": pval, "n": int(n), "alternative": "greater",
                     "favoured_mean": float(f.mean()), "other_mean": float(o.mean())},
            "cycles": [{"cycle": k + 1, "favoured": float(d[co.csplus == fav].mean()), "other": float(d[co.csplus != fav].mean())}
                       for k, d in enumerate(per)]}


def _extinction_arms(p, n):
    co = X.cohort(SEEDS["H9"], n)
    P, U = X.twins(p, co)
    Pe, Ue = extinguish(P), extinguish(U)
    steps = EXT_BLOCKS * life.COND_T + X.DELAY
    Pk, Uk = wait(P, steps), wait(U, steps)       # same elapsed time, no presentations
    return {"kept": (Pk, Uk), "extinguished": (Pe, Ue)}


def h9_extinction(p, n=N) -> dict:
    a = _extinction_arms(p, n)
    kept, ext = m_intact(*a["kept"]), m_intact(*a["extinguished"])
    return {"test": stats.paired(kept - ext), "memory_kept": float(kept.mean()), "memory_extinguished": float(ext.mean()),
            "extinguished_fraction": float(1 - ext.mean() / kept.mean()) if kept.mean() > 0 else float("nan")}


def h10_renewal(p, n=N) -> dict:
    a = _extinction_arms(p, n)
    ext_intact = m_intact(*a["extinguished"])
    ext_regrown = m_regrown(*a["extinguished"])
    kept_regrown = m_regrown(*a["kept"])
    return {"test": stats.paired(ext_regrown - ext_intact, alternative="two-sided"), "extinguished_intact": float(ext_intact.mean()),
            "extinguished_regrown": float(ext_regrown.mean()), "kept_regrown": float(kept_regrown.mean())}


def h11_reversal(p, n=N) -> dict:
    """Learn the cohort's CS+ (first lesson), then the other odour as CS+ (reversal). Memory is
    scored against the FIRST lesson: > 0 first lesson, < 0 reversal."""
    co = X.cohort(SEEDS["H11"], n)
    P, U = X.twins(p, co)
    P2, U2 = relearn(P, 1 - co.csplus, "paired"), relearn(U, 1 - co.csplus, "unpaired")
    intact = m_intact(P2, U2)
    regrown = m_regrown(P2, U2)
    return {"test": stats.paired(regrown - intact, alternative="two-sided"), "first_lesson_intact": float(intact.mean()),
            "first_lesson_regrown": float(regrown.mean()), "first_lesson_before_reversal": float(m_intact(P, U).mean())}


# ----------------------------------------------------------------------------- memory compiler

def _naive_decapitated(p, seed, n):
    co = X.cohort(seed, n)
    return X.Worms.found(p, co).learn("unpaired").cut("head")[0]


def _written(p, w: X.Worms, pattern: torch.Tensor, grad=False):
    """Add ``pattern`` (hidden x L) to the hidden channels of the body; regrow; return R(A)-R(B)."""
    x = w.x.clone()
    live = (x[:, T.ALPHA:T.ALPHA + 1] > 0.1).float()
    x = torch.cat([x[:, :T.HIDDEN0], x[:, T.HIDDEN0:] + pattern[None] * live], 1)
    tl = life.Timeline(w.co.n).free(X.REGEN).test("t", np.zeros(w.co.n, int), w.co.order)
    out = life.run(p, tl, w.masks, w.phys, x0=x, t0=w.t)
    return life.readout(out["resp"], tl.readouts["t"])["di"]


def compile_memory(p, odour: int, iters: int = 150, lr: float = 0.05, n=32, l2=0.01, seed=SEEDS["H12_design"], log=None):
    """Gradient design of a hidden pattern that makes regrown heads prefer ``odour``."""
    torch.manual_seed(seed)
    w = _naive_decapitated(p, seed, n)
    pat = torch.zeros(len(HIDDEN), T.L, requires_grad=True)
    opt = torch.optim.Adam([pat], lr=lr)
    sign = 1.0 if odour == 0 else -1.0
    for i in range(iters):
        pref = _written(p, w, pat)
        loss = ((sign * pref - 1) ** 2).mean() + l2 * (pat ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log and i % 25 == 0:
            log(f"  compile {'AB'[odour]} {i:4d} loss {loss.item():.4f} pref {pref.mean().item():+.3f}")
    return pat.detach()


def _shuffled(pat: torch.Tensor, seed: int) -> torch.Tensor:
    """The same values at shuffled body sites (per channel), keeping each channel's dose."""
    g = np.random.default_rng(seed)
    out = pat.clone()
    sites = np.arange(body.HEAD_END, body.BODY_END)
    for c in range(pat.shape[0]):
        out[c, sites] = pat[c, g.permutation(sites)]
    return out


def h12_compiler(p, n=N, log=None) -> dict:
    with torch.enable_grad():
        pa = compile_memory(p, 0, log=log)
        pb = compile_memory(p, 1, log=log)
    w = _naive_decapitated(p, SEEDS["H12_test"], n)
    with torch.no_grad():
        written = 0.5 * (_written(p, w, pa) - _written(p, w, pb)).numpy()
        shuffled = 0.5 * (_written(p, w, _shuffled(pa, 1)) - _written(p, w, _shuffled(pb, 2))).numpy()
        none = _written(p, w, torch.zeros_like(pa)).numpy()
    return {"test": stats.paired(written - shuffled), "written_memory": float(written.mean()),
            "shuffled_memory": float(shuffled.mean()), "no_pattern_preference": float(none.mean()),
            "written_A": float(_written(p, w, pa).mean()), "written_B": float(_written(p, w, pb).mean()),
            "pattern_rms": float(pa.pow(2).mean().sqrt()), "patterns": {"A": pa.tolist(), "B": pb.tolist()}}


# ----------------------------------------------------------------------------- exploratory

def h13_engram_locus(p, n=N) -> dict:
    Pc, Uc = X._decapitated_twins(p, SEEDS["H13"], n)
    base = X.di(Uc.free(X.REGEN))
    donor = X.di(Pc.free(X.REGEN)) - base

    def implant(sl):
        x = Uc.x.clone()
        x[:, HIDDEN, sl] = Pc.x[:, HIDDEN, sl]
        return X.di(Uc.with_x(x).free(X.REGEN)) - base
    edge, rest = implant(slice(12, 16)), implant(slice(16, 28))
    d = float(donor.mean())
    return {"test": stats.paired(edge - rest), "donor_memory": d, "edge_memory": float(edge.mean()),
            "rest_memory": float(rest.mean()), "edge_fraction": float(edge.mean()) / d if d > 0 else float("nan"),
            "rest_fraction": float(rest.mean()) / d if d > 0 else float("nan")}


def engram_map(p, n=N, window=4) -> dict:
    """Implant the trained body's hidden channels at one window of sites only."""
    Pc, Uc = X._decapitated_twins(p, SEEDS["engram_map"], n)
    base = X.di(Uc.free(X.REGEN))
    donor = float((X.di(Pc.free(X.REGEN)) - base).mean())
    out = []
    for start in range(body.HEAD_END, body.BODY_END, window):
        x = Uc.x.clone()
        sl = slice(start, start + window)
        x[:, HIDDEN, sl] = Pc.x[:, HIDDEN, sl]
        m = float((X.di(Uc.with_x(x).free(X.REGEN)) - base).mean())
        out.append({"sites": [start, start + window], "memory": m, "fraction": m / donor if donor > 0 else float("nan")})
    return {"donor_memory": donor, "windows": out}


def long_cycles(p, rule: str, n=64, cycles=8) -> dict:
    r = h8_attractor(p, rule, n=n, cycles=cycles) if rule in FAVOURED else None
    if r is None:
        co = X.cohort(SEEDS["long_cycles"], n)
        P, U = X.twins(p, co)
        out = []
        for k in range(cycles):
            P, U = P.cut("head")[0].free(X.REGEN), U.cut("head")[0].free(X.REGEN)
            P, a = P.test()
            U, b = U.test()
            out.append({"cycle": k + 1, "memory": float((a["di"] - b["di"]).mean())})
        return {"cycles": out}
    return {"cycles": r["cycles"]}


CONFIRMATORY = ["H8", "H9", "H10", "H11", "H12", "H13"]


def run_all(rule: str, out: str, n=N, log=print, weights=None) -> dict:
    torch.set_num_threads(1)
    weights = pathlib.Path(weights or pathlib.Path(provenance.ROOT, "weights", f"rule_{rule}.json"))
    p = load(str(weights))["tensors"]
    prereg = pathlib.Path(provenance.ROOT, "prereg", "PREREGISTRATION_v2.json")
    res = {"rule": rule, "weights_sha256": provenance.sha256_file(weights),
           "prereg_sha256": provenance.sha256_file(prereg) if prereg.exists() else None,
           "provenance": provenance.stamp(), "n": n, "alpha": X.ALPHA, "sesoi": X.SESOI}
    fns = {"H8": lambda: h8_attractor(p, rule, n), "H9": lambda: h9_extinction(p, n), "H10": lambda: h10_renewal(p, n),
           "H11": lambda: h11_reversal(p, n), "H12": lambda: h12_compiler(p, n), "H13": lambda: h13_engram_locus(p, n),
           "engram_map": lambda: engram_map(p, n), "long_cycles": lambda: long_cycles(p, rule)}
    with torch.no_grad():
        for k, fn in fns.items():
            t0 = time.time()
            res[k] = fn()
            t = res[k].get("test")
            log(f"[{rule}] {k} {time.time() - t0:.0f}s" + (f" mean {t['mean']:+.3f} p {t['p']:.3g}" if t else ""))
    ps = {h: res[h]["test"]["p"] for h in CONFIRMATORY if "test" in res[h]}
    hm = stats.holm(ps, X.ALPHA)
    res["verdicts"] = {h: {**v, "mean": res[h]["test"]["mean"], "sesoi_met": abs(res[h]["test"]["mean"]) >= X.SESOI,
                           "supported": bool(v["reject"] and abs(res[h]["test"]["mean"]) >= X.SESOI)} for h, v in hm.items()}
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(res, indent=1, default=float))
    return res
