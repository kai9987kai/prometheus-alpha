"""v0.7 training: a causal test of the noise-tolerance / cell-loss trade-off.

v0.6 found, post hoc, that across six rules the memories that tolerate noise in the hidden state
are the ones that die when cells turn over. v0.7 intervenes. From each S rule, three siblings are
trained for the same number of iterations on the same lives, which gain a 60-step "stress"
segment between the delay and the second wound:

* T (turnover): three times, 10% of the cells, chosen at random, die, then 20 steps pass;
* N (noise): three times, Gaussian noise (sigma drawn from 0.25-1) is added to every living cell's
  hidden channels, then 20 steps pass;
* C (control): 20 quiet steps, three times.

Memory is scored after the stress (and after the wound, S objective). The measurements that decide
the trade-off (turnover memory, head-noise tolerance) use held-out seeds and stress parameters
(v7.py). The locked v0.1 trainer is imported, not edited.
"""

from __future__ import annotations

import dataclasses
import math
import time

import numpy as np
import torch

from . import body, life, train
from . import tissue as T
from .train import GROW, REGEN

STRESS = ("turnover", "noise", "control")
PREFIX = {"turnover": "T", "noise": "N", "control": "C"}


def build(spec: dict, rng: np.random.Generator, stress: str, g: torch.Generator) -> life.Timeline:
    B = len(spec["csplus"])
    tl = life.Timeline(B)
    tl.free(GROW).mark("grown")
    tl.wound(body.keep_masks(spec["pre"], spec["pre_jitter"])).free(REGEN).mark("pre_regen")
    tl.condition(spec["csplus"], spec["pairing"], rng)
    tl.free(spec["delay"])
    for _ in range(3):
        if stress == "turnover":
            keep = (torch.rand(B, 1, T.L, generator=g) >= 0.10).float()
            tl.at(lambda x, k=keep: x * k)
        elif stress == "noise":
            sig = 0.25 + 0.75 * torch.rand(B, 1, 1, generator=g)
            eps = torch.randn(B, T.C - T.HIDDEN0, T.L, generator=g) * sig

            def add(x, e=eps):
                live = (x[:, T.ALPHA:T.ALPHA + 1] > 0.1).float()
                return torch.cat([x[:, :T.HIDDEN0], x[:, T.HIDDEN0:] + e * live], 1)
            tl.at(add)
        tl.free(20)
    tl.mark("stressed")
    tl.test("test_stress", spec["csplus"], 1 - spec["order"])
    tl.wound(body.keep_masks(spec["post"], spec["post_jitter"])).free(REGEN).mark("post_regen")
    tl.test("test", spec["csplus"], spec["order"]).mark("end")
    return tl


def loss(p, spec, tl, cfg):
    out = life.run(p, tl, T.fire_masks(spec["mask_seeds"], tl.t))
    anat = sum(body.anatomy_loss(out["snaps"][k]).mean() for k in ("grown", "pre_regen", "stressed", "post_regen", "end"))
    want = torch.from_numpy((spec["pairing"] == "paired").astype(np.float32))
    mem, di = 0.0, []
    for name, w in (("test_stress", torch.ones(len(want))),
                    ("test", torch.tensor([train.memory_weight("S", k) for k in spec["post"]]))):
        ro = life.readout(out["resp"], tl.readouts[name])
        mem = mem + (w * ((ro["plus"] - want) ** 2 + ro["minus"] ** 2)).sum() / w.sum().clamp_min(1)
        di.append(ro["di"][want > 0].mean().item())
    reflex = torch.stack([((out["resp"][t] - 1) ** 2)[torch.from_numpy(m)].mean() for t, m in tl.reflex]).mean()
    base = torch.cat([out["resp"][torch.from_numpy(tl.readouts[k].rest)] for k in ("test_stress", "test")]) ** 2
    total = anat + cfg.w_memory * mem + cfg.w_reflex * reflex + cfg.w_baseline * base.mean()
    return total, {"loss": total.item(), "anat": anat.item(), "mem": float(mem), "di_stress": di[0], "di_wound": di[1]}


def train_stress(stress: str, seed: int, iterations: int = 1000, threads: int = 1, log=print) -> dict:
    init = f"weights/rule_S{seed}.json"
    cfg = train.Config(family="S", seed=seed, iterations=iterations, threads=threads, init=init)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, 7, 7])            # the same lives for T, N and C siblings
    g = torch.Generator().manual_seed(1000 + seed)
    p = {k: v.clone().requires_grad_() for k, v in train.load(init)["tensors"].items()}
    opt = torch.optim.Adam(p.values(), lr=cfg.lr)
    hist, t0, name = [], time.time(), f"{PREFIX[stress]}{seed}"
    for i in range(iterations):
        spec = train.sample_life(cfg, rng)
        total, info = loss(p, spec, build(spec, rng, stress, g), cfg)
        opt.zero_grad()
        total.backward()
        for v in p.values():
            v.grad = train.normalise(v.grad)
        lr = cfg.lr * min(1.0, (i + 1) / 50) * (0.3 if i > 0.6 * iterations else 1) * (0.3 if i > 0.85 * iterations else 1)
        for gr in opt.param_groups:
            gr["lr"] = lr
        opt.step()
        if not math.isfinite(info["loss"]):
            raise FloatingPointError(i)
        if i % 25 == 0 or i == iterations - 1:
            info.update(it=i, secs=round(time.time() - t0, 1))
            hist.append(info)
            log(f"[{name}] {i:5d} loss {info['loss']:.4f} anat {info['anat']:.4f} mem {info['mem']:.4f} "
                f"DI(stress) {info['di_stress']:+.3f} DI(wound) {info['di_wound']:+.3f} {info['secs']:.0f}s", flush=True)
    cfgd = dataclasses.asdict(cfg)
    cfgd.update(family=PREFIX[stress], stress=stress)
    return {"params": {k: v.detach() for k, v in p.items()}, "history": hist, "config": cfgd, "seconds": time.time() - t0}


if __name__ == "__main__":
    import sys
    stress, seed = sys.argv[1], int(sys.argv[2])
    res = train_stress(stress, seed, iterations=int(sys.argv[3]) if len(sys.argv) > 3 else 1000)
    train.save(res, f"weights/rule_{PREFIX[stress]}{seed}.json")
