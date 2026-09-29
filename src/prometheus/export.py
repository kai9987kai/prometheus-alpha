"""Exports for the browser lab (web/data.js) and the JS/Python parity fixture."""

from __future__ import annotations

import datetime
import json
import pathlib

import numpy as np
import torch

from . import body, claims, life
from . import tissue as T
from .provenance import ROOT
from .train import load

FIXTURE = ROOT / "tests" / "fixtures" / "parity.json"


def web(rules: list[str], out: str | pathlib.Path = ROOT / "web" / "data.js"):
    data = {"rules": {}, "claims": [], "generated": datetime.date.today().isoformat()}
    for path in rules:
        path = pathlib.Path(ROOT, path)
        if not path.exists():
            print(f"skip {path} (missing)")
            continue
        doc = load(str(path))
        name = f"{doc['config']['family']}{doc['config']['seed']}"
        data["rules"][name] = {"family": doc["config"]["family"], "seed": doc["config"]["seed"],
                               "n_params": doc["n_params"], **doc["params"]}
        v2 = ROOT / "results" / f"v2_{name}.json"
        if v2.exists():
            r2 = json.loads(v2.read_text())
            data["rules"][name]["compiled"] = {k: [[round(v, 4) for v in row] for row in pat]
                                               for k, pat in r2["H12"]["patterns"].items()}
        v3 = ROOT / "results" / f"v3_{name}.json"
        if v3.exists():
            data["rules"][name]["decoder"] = json.loads(v3.read_text())["H16"]["decoder"]
    data["default_rule"] = next((k for k in data["rules"] if k.startswith("S")), next(iter(data["rules"]), None))
    try:
        data["claims"] = claims.rendered_rows()
    except (OSError, KeyError, ValueError):
        pass
    js = "window.PROMETHEUS_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n"
    pathlib.Path(out).write_text(js, encoding="utf-8")
    print(f"wrote {out}: rules {list(data['rules'])}, {len(data['claims'])} claims")


def _parity_rule(seed: int = 123) -> dict:
    """A random rule with a live action head, so the parity test does not depend on training."""
    p = T.init_params(seed)
    g = torch.Generator().manual_seed(seed + 1)
    p["W2"] = torch.randn(T.C, T.HIDDEN, generator=g) * 0.05
    p["b2"] = torch.randn(T.C, generator=g) * 0.01
    return p


def _case(p: dict, name: str, rng: np.random.Generator) -> dict:
    """Grow, condition, cut the head (complete amputation), block gap junctions, regrow, probe."""
    B = 2
    tl = life.Timeline(B).free(24)
    tl.condition(np.array([0, 1]), ["paired", "unpaired"], rng)
    cut_at = tl.t
    tl.free(6)
    tl.gj_now = np.array([0.0, 1.0])
    tl.free(8)
    tl.gj_now = np.ones(B)
    tl.test("t", np.array([0, 1]), np.array([0, 1]))
    masks = T.fire_masks([11, 12], tl.t)
    from .experiments import amputate
    tl.events.setdefault(cut_at, []).append(lambda x: amputate(x, "head")[0])
    with torch.no_grad():
        states, x = [], T.founder(B)
        cue = torch.from_numpy(np.stack(tl.cue))
        us = torch.from_numpy(np.stack(tl.us))
        gj = torch.from_numpy(np.stack(tl.gj))
        resp = []
        for t in range(tl.t):
            for fn in tl.events.get(t, []):
                x = fn(x)
            g = gj[t]
            x = T.step(p, x, cue[t], us[t], masks[t], T.Physics(), None if bool((g == 1).all()) else g)
            resp.append(T.response(x))
            if t % 10 == 9 or t == tl.t - 1:
                states.append({"t": t, "x": x.tolist()})
    return {"name": name, "params": {k: v.tolist() for k, v in p.items()}, "T": tl.t, "cut_at": cut_at,
            "cue": np.stack(tl.cue).tolist(), "us": np.stack(tl.us).tolist(), "gj": np.stack(tl.gj).tolist(),
            "masks": masks[:, :, 0].tolist(), "resp": torch.stack(resp).tolist(), "states": states,
            "region_final": body.region(x).tolist()}


def parity_fixture(out: pathlib.Path = FIXTURE):
    cases = [_case(_parity_rule(), "random", np.random.default_rng(5))]
    for path in ("weights/rule_S0.json", "weights/rule_E0.json"):
        f = ROOT / path
        if f.exists():
            cases.append(_case(load(str(f))["tensors"], pathlib.Path(path).stem, np.random.default_rng(6)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    print(f"wrote {out} ({len(cases)} cases)")
