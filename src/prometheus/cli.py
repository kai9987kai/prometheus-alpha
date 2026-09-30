"""Command line: ``prometheus <command>``."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch


def _show(args):
    from . import body, experiments as X, life
    from . import tissue as T
    from .train import load
    torch.set_num_threads(1)
    p = load(args.weights)["tensors"]
    co = X.cohort(args.seed, 2)
    co = X.Cohort(co.seed, np.array([args.csplus, args.csplus]), np.array([0, 0]), co.mask_seeds)
    glyph = {-1: " ", 0: "H", 1: "=", 2: "t"}

    def row(x, label):
        lab = body.region(x)[0].tolist()
        v = x[0, T.V].tolist()
        vs = "".join(" " if l < 0 else ".:-=+*#%@"[min(8, max(0, int((vi + 2) / 4 * 9)))] for l, vi in zip(lab, v))
        print(f"{label:<34s} |{''.join(glyph[l] for l in lab)}|  voltage |{vs}|")

    with torch.no_grad():
        w = X.Worms.found(p, co)
        row(w.x, "founder")
        w = w.free(X.GROW)
        row(w.x, "grown (32 steps)")
        pairing = "paired" if not args.unpaired else "unpaired"
        w = X.Worms.found(p, co).learn(pairing)
        row(w.x, f"conditioned ({pairing}, CS+ = {'AB'[args.csplus]})")
        _, r = w.test()
        print(f"{'':34s}  response to CS+ {r['plus'][0]:+.2f}   CS- {r['minus'][0]:+.2f}   DI {r['di'][0]:+.2f}")
        for k in range(args.cycles):
            w, res = w.cut(args.wound)
            row(w.x, f"cycle {k + 1}: {args.wound} removed")
            w = w.free(X.REGEN)
            row(w.x, f"cycle {k + 1}: regenerated")
            w, r = w.test()
            print(f"{'':34s}  response to CS+ {r['plus'][0]:+.2f}   CS- {r['minus'][0]:+.2f}   DI {r['di'][0]:+.2f}")
    print("\nH head  = trunk  t tail;  voltage from low (.) to high (@)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="prometheus", description="Memory that outlives its organ: a regenerating-worm laboratory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="meta-train a cell rule")
    t.add_argument("--family", choices=["E", "S"], required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--iterations", type=int, default=3000)
    t.add_argument("--threads", type=int, default=1)
    t.add_argument("--init", help="start from these weights (S rules start from their E sibling)")
    t.add_argument("--out", required=True)

    r = sub.add_parser("run", help="run the preregistered (and exploratory) experiments on a rule")
    r.add_argument("--weights", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--only", help="comma-separated experiment names")
    r.add_argument("--n", type=int, default=128)

    s = sub.add_parser("show", help="ASCII view of one worm learning, losing its head and regrowing it")
    s.add_argument("--weights", required=True)
    s.add_argument("--csplus", type=int, default=0)
    s.add_argument("--unpaired", action="store_true")
    s.add_argument("--wound", default="head", choices=["head", "tail", "fragment"])
    s.add_argument("--cycles", type=int, default=2)
    s.add_argument("--seed", type=int, default=7)

    pr = sub.add_parser("prereg", help="lock or verify the preregistration")
    pr.add_argument("action", choices=["lock", "verify"])
    pr.add_argument("--version", default="1", choices=["1", "2", "3", "4", "5", "6", "7"])

    r2 = sub.add_parser("run2", help="run the v0.2 experiments (extinction, reversal, compiler, engram locus, attractor)")
    r2.add_argument("--rule", required=True, help="E0, E1, S0 or S1")
    r2.add_argument("--out", required=True)
    r2.add_argument("--n", type=int, default=128)

    c = sub.add_parser("claims", help="check or render the claims ledger")
    c.add_argument("action", choices=["check", "render"])

    e = sub.add_parser("export-web", help="write web/data.js from weights and results")
    e.add_argument("--rules", nargs="+", default=["weights/rule_E0.json", "weights/rule_S0.json"])

    sub.add_parser("parity-fixture", help="write tests/fixtures/parity.json for the JS engine test")

    a = ap.parse_args(argv)
    if a.cmd == "train":
        from . import train
        res = train.train(train.Config(family=a.family, seed=a.seed, iterations=a.iterations, threads=a.threads, init=a.init))
        train.save(res, a.out)
    elif a.cmd == "run":
        from . import experiments, prereg
        ok, bad = prereg.verify()
        print("preregistration lock verifies: CONFIRMATORY run" if ok else f"lock does NOT verify ({bad}): EXPLORATORY run")
        res = experiments.run_all(a.weights, a.out, a.only.split(",") if a.only else None, n=a.n)
        res["confirmatory"] = ok
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=float)
        for h, v in res["verdicts"].items():
            print(f"{h}: mean {v['mean']:+.3f}  p_holm {v['p_adj']:.3g}  {'SUPPORTED' if v['supported'] else 'not supported'}")
    elif a.cmd == "run2":
        from . import prereg, v2
        ok, bad = prereg.verify("2")
        print("v0.2 lock verifies: CONFIRMATORY run" if ok else f"v0.2 lock does NOT verify ({bad}): EXPLORATORY run")
        res = v2.run_all(a.rule, a.out, n=a.n)
        res["confirmatory"] = ok
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=float)
        for h, v in res["verdicts"].items():
            print(f"{h}: mean {v['mean']:+.3f}  p_holm {v['p_adj']:.3g}  {'SUPPORTED' if v['supported'] else 'not supported'}")
    elif a.cmd == "show":
        _show(a)
    elif a.cmd == "prereg":
        from . import prereg
        if a.action == "lock":
            print(json.dumps(prereg.lock(a.version), indent=1))
        else:
            ok, bad = prereg.verify(a.version)
            print("ok: preregistration lock verifies" if ok else f"FAIL: changed since lock: {bad}")
            return 0 if ok else 1
    elif a.cmd == "claims":
        from . import claims
        return claims.main(a.action)
    elif a.cmd == "export-web":
        from . import export
        export.web(a.rules)
    elif a.cmd == "parity-fixture":
        from . import export
        export.parity_fixture()
    return 0


if __name__ == "__main__":
    sys.exit(main())
