"""The preregistered experiments (prereg/PREREGISTRATION.json) and the exploratory ones.

Every experiment starts from cohorts of worms that grow from one founder cell (32 steps), are
conditioned (48 steps) and wait (12 steps). A worm and its *unpaired twin* share everything,
including every asynchronous-update mask, except the timing of the US, so the per-worm
difference ``DI(paired) - DI(unpaired twin)`` is a within-worm measure of the association.

Decapitation in experiments is *complete*: every site up to the worm's own last head cell, plus
a 2-site margin, is removed, so no cell that could sense a cue during conditioning survives.
Its completeness is reported (``residual_head``).

H1  learning            intact worms remember which cue was paired
H2  survival            the memory comes back after the head is removed and regrown
H3a voltage necessary   giving a decapitated trained body its untrained twin's voltage erases it
H3b voltage sufficient  writing a decapitated trained body's voltage into its untrained twin's
                        body makes the twin's regrown head remember
H3c hidden sufficient   the same with the 10 hidden channels instead of voltage
H4  chimera dominance   head of an A-trained worm on the body of a B-trained worm: whose memory
                        does it express? (two-sided)
H4b chimera regrown     cut the chimera's head off: the regrown head carries the body's memory
H5  transfer            a naive head grafted onto a trained body takes on the body's memory
H6  Promethean cycles   memory after the third successive decapitation
H7  trunk fragment      memory after both head and tail are removed
calibration             A/A tests: false-positive rate of the pipeline
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time

import numpy as np
import torch

from . import body, life, provenance, stats
from . import tissue as T
from .train import GROW, REGEN

N = 128
DELAY = 12
INTEGRATE = 32          # steps a grafted chimera heals before it is tested
CYCLES = 4
MAX_T = 420
ALPHA = 0.05
SESOI = 0.10            # smallest effect of interest, in response units (a trained CR is ~1)
SEEDS = {"H1": 1001, "H2": 1002, "H3a": 1031, "H3b": 1032, "H3c": 1033, "H4": 1004, "H5": 1005, "H6": 1006, "H7": 1007,
         "calibration": 1100, "locus": 1201, "gap_junctions": 1202, "anatomy": 1301}
GROUPS = {"identity+response": [T.HEAD, T.TRUNK, T.TAIL, T.R], "voltage": [T.V],
          "hidden": list(range(T.HIDDEN0, T.C)),
          "all but voltage": [c for c in range(T.C) if c != T.V]}


# ----------------------------------------------------------------------------- worms and cohorts

@dataclasses.dataclass(frozen=True)
class Cohort:
    seed: int
    csplus: np.ndarray
    order: np.ndarray
    mask_seeds: np.ndarray

    @property
    def n(self) -> int:
        return len(self.csplus)


def cohort(seed: int, n: int = N, csplus: int | None = None) -> Cohort:
    rng = np.random.default_rng([seed, 0])
    cs = rng.permutation(np.repeat([0, 1], n // 2)) if csplus is None else np.full(n, csplus)
    order = rng.permutation(np.repeat([0, 1], n // 2))
    return Cohort(seed, cs, order, rng.integers(0, 2 ** 31, n))


@dataclasses.dataclass
class Worms:
    p: dict
    co: Cohort
    x: torch.Tensor
    t: int
    masks: torch.Tensor
    phys: T.Physics = T.Physics()

    @classmethod
    def found(cls, p, co: Cohort, phys: T.Physics = T.Physics()) -> "Worms":
        return cls(p, co, T.founder(co.n), 0, T.fire_masks(co.mask_seeds, MAX_T, phys.fire_rate), phys)

    def play(self, tl: life.Timeline) -> tuple["Worms", dict]:
        out = life.run(self.p, tl, self.masks, self.phys, x0=self.x, t0=self.t)
        return dataclasses.replace(self, x=out["x"], t=self.t + tl.t), out

    def with_x(self, x) -> "Worms":
        return dataclasses.replace(self, x=x)

    def learn(self, pairing: str, block: bool = False, csplus=None) -> "Worms":
        cs = self.co.csplus if csplus is None else csplus
        tl = life.Timeline(self.co.n).free(GROW)
        if block:
            tl.gj_now = np.zeros(self.co.n)
        tl.condition(cs, pairing, np.random.default_rng([self.co.seed, 1])).free(DELAY)
        return self.play(tl)[0]

    def free(self, n: int = REGEN) -> "Worms":
        return self.play(life.Timeline(self.co.n).free(n))[0]

    def test(self, csplus=None) -> tuple["Worms", dict]:
        cs = self.co.csplus if csplus is None else csplus
        tl = life.Timeline(self.co.n).test("t", cs, self.co.order)
        w, out = self.play(tl)
        ro = life.readout(out["resp"], tl.readouts["t"])
        return w, {k: v.numpy().astype(float) for k, v in ro.items()}

    def cut(self, kind: str = "head") -> tuple["Worms", np.ndarray]:
        x, residual = amputate(self.x, kind)
        return self.with_x(x), residual


def amputate(x: torch.Tensor, kind: str = "head", margin: int = 2) -> tuple[torch.Tensor, np.ndarray]:
    """Complete amputation: remove every site up to the worm's own last head cell plus a margin
    (and, for a fragment, every site from its first tail cell on). Returns the residual head
    weight of what is left (should be ~0)."""
    lab = body.region(x)
    keep = torch.ones(x.shape[0], 1, T.L)
    for i in range(x.shape[0]):
        if kind in ("head", "fragment"):
            h = (lab[i] == 0).nonzero()
            last = int(h.max()) if len(h) else body.HEAD_END - 1
            keep[i, 0, : last + margin + 1] = 0
        if kind in ("tail", "fragment"):
            tl = (lab[i] == 2).nonzero()
            first = int(tl.min()) if len(tl) else body.TRUNK_END
            keep[i, 0, first:] = 0
    x = x * keep
    return x, T.head_weight(x).sum((1, 2)).numpy()


def graft(head_x: torch.Tensor, body_x: torch.Tensor, at: int = body.HEAD_END) -> torch.Tensor:
    """Sites anterior of ``at`` from the head donor, the rest from the body donor."""
    front = (torch.arange(T.L) < at).float()
    return head_x * front + body_x * (1 - front)


# ----------------------------------------------------------------------------- helpers

def _test_summary(d, alternative="greater", seed=0) -> dict:
    return stats.paired(d, seed=seed, alternative=alternative)


def twins(p, co, block=False, phys=T.Physics()):
    """(paired, unpaired twin) after learning."""
    base = Worms.found(p, co, phys)
    return base.learn("paired", block), base.learn("unpaired", block)


def di(w: Worms) -> np.ndarray:
    return w.test()[1]["di"]


# ----------------------------------------------------------------------------- confirmatory

def h1_learning(p, n=N) -> dict:
    co = cohort(SEEDS["H1"], n)
    P, U = twins(p, co)
    _, rp = P.test()
    _, ru = U.test()
    d = rp["di"] - ru["di"]
    return {"test": _test_summary(d), "paired": {k: float(v.mean()) for k, v in rp.items()},
            "unpaired": {k: float(v.mean()) for k, v in ru.items()}}


def h2_survival(p, n=N) -> dict:
    co = cohort(SEEDS["H2"], n)
    P, U = twins(p, co)
    intact = di(P) - di(U)
    Pc, res_p = P.cut("head")
    Uc, _ = U.cut("head")
    Pr, Ur = Pc.free(REGEN), Uc.free(REGEN)
    _, rp = Pr.test()
    _, ru = Ur.test()
    d = rp["di"] - ru["di"]
    anat = body.anatomy_scores(Pr.x)
    return {"test": _test_summary(d), "intact_memory": float(intact.mean()),
            "retention": float(d.mean() / intact.mean()) if intact.mean() > 0 else float("nan"),
            "regrown": {k: float(v.mean()) for k, v in rp.items()},
            "regrown_unpaired": {k: float(v.mean()) for k, v in ru.items()},
            "residual_head_max": float(res_p.max()), "residual_head_mean": float(res_p.mean()),
            "regen_iou": float(anat["iou"].mean()), "regen_region_acc": float(anat["region_acc"].mean()),
            "regen_head_cells": float(anat["head_cells"].mean())}


def _decapitated_twins(p, seed, n):
    co = cohort(seed, n)
    P, U = twins(p, co)
    return P.cut("head")[0], U.cut("head")[0]


def h3a_voltage_necessary(p, n=N) -> dict:
    """Right after decapitation, give the trained body its untrained twin's voltage (and nothing
    else). Does the regrown head lose the memory? d = M(untouched) - M(voltage swapped)."""
    Pc, Uc = _decapitated_twins(p, SEEDS["H3a"], n)
    base = di(Uc.free(REGEN))
    full = di(Pc.free(REGEN)) - base
    x = Pc.x.clone()
    x[:, T.V] = Uc.x[:, T.V]
    swapped = di(Pc.with_x(x).free(REGEN)) - base
    return {"test": _test_summary(full - swapped), "memory_untouched": float(full.mean()),
            "memory_voltage_swapped": float(swapped.mean()),
            "retained_fraction": float(swapped.mean() / full.mean()) if full.mean() > 0 else float("nan")}


def _implant(p, seed, chans, n) -> dict:
    """Right after decapitation, write the trained body's channels ``chans`` (and nothing else)
    into its untrained twin's body. Does the twin's regrown head show the memory? d = M(implanted)."""
    Pc, Uc = _decapitated_twins(p, seed, n)
    base = di(Uc.free(REGEN))
    donor = di(Pc.free(REGEN)) - base
    x = Uc.x.clone()
    x[:, chans] = Pc.x[:, chans]
    implanted = di(Uc.with_x(x).free(REGEN)) - base
    return {"test": _test_summary(implanted), "donor_memory": float(donor.mean()),
            "transferred_fraction": float(implanted.mean() / donor.mean()) if donor.mean() > 0 else float("nan")}


def h3b_voltage_sufficient(p, n=N) -> dict:
    return _implant(p, SEEDS["H3b"], [T.V], n)


def h3c_hidden_sufficient(p, n=N) -> dict:
    """The same implant with the 10 hidden channels instead of voltage."""
    return _implant(p, SEEDS["H3c"], GROUPS["hidden"], n)


def _conditioned(p, seed, csplus, n, pairing="paired"):
    co = cohort(seed, n, csplus)
    return Worms.found(p, co).learn(pairing)


def _chimera(p, head: Worms, bod: Worms, graft_co: Cohort) -> Worms:
    """Graft, heal for INTEGRATE steps on the graft cohort's noise. Tested with cue A as '+'."""
    w = Worms(p, graft_co, graft(head.x, bod.x), head.t, T.fire_masks(graft_co.mask_seeds, MAX_T))
    return w.free(INTEGRATE)


def _pref(w: Worms) -> np.ndarray:
    """R(cue A) - R(cue B)."""
    return w.test(csplus=np.zeros(w.co.n, int))[1]["di"]


def _chimera_suite(p, n):
    """Grafts of A-trained and B-trained worms (and same-memory controls), healed for INTEGRATE
    steps; preferences P = R(A) - R(B) before and after the chimera is decapitated and regrows."""
    s = SEEDS["H4"]
    A, A2 = _conditioned(p, s * 10 + 1, 0, n), _conditioned(p, s * 10 + 2, 0, n)
    B, B2 = _conditioned(p, s * 10 + 3, 1, n), _conditioned(p, s * 10 + 4, 1, n)
    gco = cohort(s * 10 + 5, n)
    ch = {"A|A'": _chimera(p, A, A2, gco), "B|B'": _chimera(p, B, B2, gco),
          "A|B": _chimera(p, A, B, gco), "B|A": _chimera(p, B, A, gco)}
    pref = {k: _pref(w) for k, w in ch.items()}
    after = {k: _pref(w.cut("head")[0].free(REGEN)) for k, w in ch.items()}
    return pref, after


def _dominance(pref: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per worm index: signed = (P(A|B) - P(B|A)) / 2, > 0 when the head's memory wins;
    scale = (P(A|A') - P(B|B')) / 2, the full, uncontested memory."""
    return 0.5 * (pref["A|B"] - pref["B|A"]), 0.5 * (pref["A|A'"] - pref["B|B'"])


def h4_chimera(p, n=N) -> dict:
    """Healed chimeras, intact: is the expressed memory an even split? (two-sided)"""
    pref, _ = _chimera_suite(p, n)
    signed, scale = _dominance(pref)
    return {"test": _test_summary(signed, alternative="two-sided"),
            "preference": {k: float(v.mean()) for k, v in pref.items()},
            "dominance": float(signed.mean() / scale.mean()) if scale.mean() > 0 else float("nan"),
            "uncontested_memory": float(scale.mean())}


def h4b_chimera_regrown(p, n=N) -> dict:
    """The same chimeras decapitated and regrown: does the new head carry the body's memory?
    (one-sided, signed < 0). Dominance is scaled by the intact uncontested memory."""
    pref, after = _chimera_suite(p, n)
    _, scale = _dominance(pref)
    signed, scale_after = _dominance(after)
    return {"test": _test_summary(signed, alternative="less"),
            "preference": {k: float(v.mean()) for k, v in after.items()},
            "dominance": float(signed.mean() / scale.mean()) if scale.mean() > 0 else float("nan"),
            "uncontested_memory_regrown": float(scale_after.mean())}


def h5_transfer(p, n=N) -> dict:
    s = SEEDS["H5"]
    A, B = _conditioned(p, s * 10 + 1, 0, n), _conditioned(p, s * 10 + 2, 1, n)
    Nv = _conditioned(p, s * 10 + 3, None, n, pairing="unpaired")
    gco = cohort(s * 10 + 5, n)
    pref = {"N|A": _pref(_chimera(p, Nv, A, gco)), "N|B": _pref(_chimera(p, Nv, B, gco)),
            "A|N": _pref(_chimera(p, A, Nv, gco)), "B|N": _pref(_chimera(p, B, Nv, gco)),
            "A|A": _pref(_chimera(p, A, A, gco)), "B|B": _pref(_chimera(p, B, B, gco))}
    transfer = 0.5 * (pref["N|A"] - pref["N|B"])
    head_kept = 0.5 * (pref["A|N"] - pref["B|N"])
    scale = 0.5 * (pref["A|A"] - pref["B|B"])
    return {"test": _test_summary(transfer), "preference": {k: float(v.mean()) for k, v in pref.items()},
            "transfer_fraction": float(transfer.mean() / scale.mean()) if scale.mean() > 0 else float("nan"),
            "head_memory_on_naive_body": _test_summary(head_kept),
            "head_kept_fraction": float(head_kept.mean() / scale.mean()) if scale.mean() > 0 else float("nan"),
            "uncontested_memory": float(scale.mean())}


def h6_promethean(p, n=N, cycles=CYCLES) -> dict:
    co = cohort(SEEDS["H6"], n)
    P, U = twins(p, co)
    per, anat, res = [], [], []
    for k in range(1, cycles + 1):
        P, rp = P.cut("head")
        U, _ = U.cut("head")
        P, U = P.free(REGEN), U.free(REGEN)
        P, a = P.test()
        U, b = U.test()
        per.append(a["di"] - b["di"])
        sc = body.anatomy_scores(P.x)
        anat.append({"iou": float(sc["iou"].mean()), "region_acc": float(sc["region_acc"].mean())})
        res.append(float(rp.max()))
    return {"test": _test_summary(per[2]),
            "cycles": [{"cycle": k + 1, **stats.mean_ci(d, seed=k), "anatomy": anat[k], "residual_head_max": res[k]}
                       for k, d in enumerate(per)]}


def h7_fragment(p, n=N) -> dict:
    co = cohort(SEEDS["H7"], n)
    P, U = twins(p, co)
    Pr = P.cut("fragment")[0].free(REGEN)
    Ur = U.cut("fragment")[0].free(REGEN)
    d = di(Pr) - di(Ur)
    sc = body.anatomy_scores(Pr.x)
    return {"test": _test_summary(d), "regen_iou": float(sc["iou"].mean()),
            "regen_region_acc": float(sc["region_acc"].mean())}


def calibration(p, n=N, tests=40) -> dict:
    """A/A: two independent cohorts of paired worms, identical protocol. Any 'effect' is noise."""
    ps = []
    for k in range(tests):
        a = Worms.found(p, cohort(SEEDS["calibration"] * 100 + 2 * k, n)).learn("paired")
        b = Worms.found(p, cohort(SEEDS["calibration"] * 100 + 2 * k + 1, n)).learn("paired")
        ps.append(stats.paired(di(a) - di(b), n_boot=10, seed=k)["p"])
    ps = np.array(ps)
    return {"tests": tests, "n": n, "false_positive_rate": float((ps <= ALPHA).mean()), "p_values": ps.tolist()}


# ----------------------------------------------------------------------------- exploratory

def locus(p, n=N) -> dict:
    """After decapitation, swap channel groups between a trained body and its untrained twin
    (same cut, same noise). 'swap out': the trained body gets the twin's group (how much memory
    is left?). 'implant': the twin gets the trained body's group (how much memory arrives?)."""
    Pc, Uc = _decapitated_twins(p, SEEDS["locus"], n)
    base = di(Uc.free(REGEN))
    full = float((di(Pc.free(REGEN)) - base).mean())
    out, imp = {"none": full}, {}
    for name, chans in GROUPS.items():
        x = Pc.x.clone()
        x[:, chans] = Uc.x[:, chans]
        out[name] = float((di(Pc.with_x(x).free(REGEN)) - base).mean())
        x = Uc.x.clone()
        x[:, chans] = Pc.x[:, chans]
        imp[name] = float((di(Uc.with_x(x).free(REGEN)) - base).mean())
    frac = lambda v: v / full if full > 0 else float("nan")
    return {"memory_trained_body": full, "memory_after_swap_out": out, "memory_after_implant": imp,
            "retained_fraction": {k: frac(v) for k, v in out.items()},
            "implanted_fraction": {k: frac(v) for k, v in imp.items()}}


def gap_junctions(p, n=N) -> dict:
    """Block gap junctions for the 12-step delay between conditioning and the cut (learning
    itself is untouched). Compared with the same worms with junctions open."""
    co = cohort(SEEDS["gap_junctions"], n)
    out = {}
    for block in (False, True):
        base = Worms.found(p, co)
        pl, ul = [], []
        for pairing, acc in (("paired", pl), ("unpaired", ul)):
            tl = life.Timeline(co.n).free(GROW)
            tl.condition(co.csplus, pairing, np.random.default_rng([co.seed, 1]))
            if block:
                tl.gj_now = np.zeros(co.n)
            tl.free(DELAY)
            acc.append(base.play(tl)[0])
        P, U = pl[0], ul[0]
        intact = di(P) - di(U)
        cut = di(P.cut("head")[0].free(REGEN)) - di(U.cut("head")[0].free(REGEN))
        out["blocked" if block else "open"] = {"intact": intact, "cut": cut}
    return {"memory": {a: {k: float(v.mean()) for k, v in d.items()} for a, d in out.items()},
            "block_effect_intact": stats.paired(out["open"]["intact"] - out["blocked"]["intact"], n_boot=2000),
            "block_effect_cut": stats.paired(out["open"]["cut"] - out["blocked"]["cut"], n_boot=2000)}


def anatomy(p, n=N) -> dict:
    co = cohort(SEEDS["anatomy"], n)
    w = Worms.found(p, co).free(GROW)
    grown = body.anatomy_scores(w.x)
    out = {"grown": {k: float(v.mean()) for k, v in grown.items()}}
    for kind in ("head", "tail", "fragment"):
        r = w.cut(kind)[0].free(REGEN)
        out[f"regen_{kind}"] = {k: float(v.mean()) for k, v in body.anatomy_scores(r.x).items()}
    long = w.free(300)
    out["held_300_steps"] = {k: float(v.mean()) for k, v in body.anatomy_scores(long.x).items()}
    return out


CONFIRMATORY = {"H1": h1_learning, "H2": h2_survival, "H3a": h3a_voltage_necessary, "H3b": h3b_voltage_sufficient,
                "H3c": h3c_hidden_sufficient, "H4": h4_chimera, "H4b": h4b_chimera_regrown, "H5": h5_transfer,
                "H6": h6_promethean, "H7": h7_fragment}
EXPLORATORY = {"calibration": calibration, "anatomy": anatomy, "locus": locus, "gap_junctions": gap_junctions}


def verdicts(res: dict) -> dict:
    """Holm across the confirmatory family, then the SESOI rule."""
    ps = {h: res[h]["test"]["p"] for h in CONFIRMATORY if h in res}
    hm = stats.holm(ps, ALPHA)
    out = {}
    for h, r in hm.items():
        m = res[h]["test"]["mean"]
        big = abs(m) >= SESOI
        out[h] = {**r, "mean": m, "sesoi_met": bool(big), "supported": bool(r["reject"] and big)}
    return out


def run_all(weights: str, out: str, which=None, n=N, log=print) -> dict:
    from .train import load
    torch.set_num_threads(1)
    doc = load(weights)
    p = doc["tensors"]
    prereg = pathlib.Path(provenance.ROOT, "prereg", "PREREGISTRATION.json")
    res = {"weights": weights, "family": doc["config"]["family"], "rule_seed": doc["config"]["seed"],
           "weights_sha256": provenance.sha256_file(weights),
           "prereg_sha256": provenance.sha256_file(prereg) if prereg.exists() else None,
           "provenance": provenance.stamp(), "n": n, "alpha": ALPHA, "sesoi": SESOI}
    names = which or list(CONFIRMATORY) + list(EXPLORATORY)
    with torch.no_grad():
        for name in names:
            t0 = time.time()
            fn = CONFIRMATORY.get(name) or EXPLORATORY[name]
            res[name] = fn(p, n=n)
            log(f"[{pathlib.Path(weights).stem}] {name} done in {time.time() - t0:.0f}s: "
                + (f"mean {res[name]['test']['mean']:+.3f} p {res[name]['test']['p']:.3g}" if "test" in res[name] else ""))
    res["verdicts"] = verdicts(res)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    return res
