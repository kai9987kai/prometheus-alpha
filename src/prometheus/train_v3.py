"""v0.3 training: B ("balanced") rules.

v0.2 found that S rules keep one odour's memory through repeated regrowth and lose the other's.
A B rule starts from its S sibling and is trained further on lives with TWO successive
amputations after conditioning, with the memory scored after each regrowth. Cycles 3-8, the
per-odour balance, chimeras and the compiler stay held out.

The v0.1 trainer (train.py) is locked, so this module builds on it without editing it.
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


def build(spec: dict, rng: np.random.Generator) -> life.Timeline:
    B = len(spec["csplus"])
    tl = life.Timeline(B)
    tl.free(GROW).mark("grown")
    tl.wound(body.keep_masks(spec["pre"], spec["pre_jitter"])).free(REGEN).mark("pre_regen")
    tl.condition(spec["csplus"], spec["pairing"], rng)
    tl.free(spec["delay"])
    tl.wound(body.keep_masks(spec["post"], spec["post_jitter"])).free(REGEN).mark("post_regen")
    tl.test("test", spec["csplus"], spec["order"])
    tl.wound(body.keep_masks(spec["post2"], spec["post2_jitter"])).free(REGEN).mark("post2_regen")
    tl.test("test2", spec["csplus"], 1 - spec["order"]).mark("end")
    return tl


def loss(p, spec, tl, cfg):
    masks = T.fire_masks(spec["mask_seeds"], tl.t)
    out = life.run(p, tl, masks)
    anat = sum(body.anatomy_loss(out["snaps"][k]).mean() for k in ("grown", "pre_regen", "post_regen", "post2_regen", "end"))
    want = torch.from_numpy((spec["pairing"] == "paired").astype(np.float32))
    mem, di = 0.0, []
    for name, post in (("test", spec["post"]), ("test2", spec["post2"])):
        ro = life.readout(out["resp"], tl.readouts[name])
        w = torch.tensor([train.memory_weight("S", k) for k in post])
        mem = mem + (w * ((ro["plus"] - want) ** 2 + ro["minus"] ** 2)).sum() / w.sum().clamp_min(1)
        di.append(ro["di"][want > 0].mean().item())
    rf = [((out["resp"][t] - 1) ** 2)[torch.from_numpy(m)].mean() for t, m in tl.reflex]
    reflex = torch.stack(rf).mean()
    base = torch.cat([out["resp"][torch.from_numpy(tl.readouts[k].rest)] for k in ("test", "test2")]) ** 2
    total = anat + cfg.w_memory * mem + cfg.w_reflex * reflex + cfg.w_baseline * base.mean()
    return total, {"loss": total.item(), "anat": anat.item(), "mem": float(mem), "di1": di[0], "di2": di[1]}


def train_b(init: str, seed: int, iterations: int = 1500, threads: int = 1, log=print) -> dict:
    cfg = train.Config(family="S", seed=seed, iterations=iterations, threads=threads, init=init)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, 7, 2])
    p = {k: v.clone().requires_grad_() for k, v in train.load(init)["tensors"].items()}
    opt = torch.optim.Adam(p.values(), lr=cfg.lr)
    hist, t0 = [], time.time()
    for i in range(iterations):
        spec = train.sample_life(cfg, rng)
        spec["post"] = rng.choice(["head", "tail", "none"], cfg.batch, p=[0.7, 0.15, 0.15])
        spec["post2"] = rng.choice(["head", "tail", "none"], cfg.batch, p=[0.7, 0.15, 0.15])
        spec["post2_jitter"] = body.random_jitter(rng, cfg.batch)
        total, info = loss(p, spec, build(spec, rng), cfg)
        opt.zero_grad()
        total.backward()
        for v in p.values():
            v.grad = train.normalise(v.grad)
        lr = cfg.lr * min(1.0, (i + 1) / 50) * (0.3 if i > 0.6 * iterations else 1) * (0.3 if i > 0.85 * iterations else 1)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        if not math.isfinite(info["loss"]):
            raise FloatingPointError(i)
        if i % 25 == 0 or i == iterations - 1:
            info.update(it=i, secs=round(time.time() - t0, 1))
            hist.append(info)
            log(f"[B{seed}] {i:5d} loss {info['loss']:.4f} anat {info['anat']:.4f} mem {info['mem']:.4f} "
                f"DI1 {info['di1']:+.3f} DI2 {info['di2']:+.3f} {info['secs']:.0f}s", flush=True)
    cfgd = dataclasses.asdict(cfg)
    cfgd["family"] = "B"
    return {"params": {k: v.detach() for k, v in p.items()}, "history": hist, "config": cfgd, "seconds": time.time() - t0}


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1])
    res = train_b(f"weights/rule_S{seed}.json", seed, iterations=int(sys.argv[2]) if len(sys.argv) > 2 else 1500)
    train.save(res, f"weights/rule_B{seed}.json")
