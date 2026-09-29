"""v0.4 training: F ("fission") rules.

v0.3 found that when a trained worm is cut in two, only the half that keeps the head remembers:
the selected rules' body copy exists only at the neck. An F rule starts from its S sibling and is
trained further on lives in which, after conditioning, the worm is cut at site 20 and the
POSTERIOR half (70%) or the anterior half (15%) must regrow into a whole worm that remembers, or
the head is cut off (15%, to keep the S ability). Other split positions, the location of the new
copy and every other v0.1-v0.3 test stay held out.
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

SPLIT, FISSION_REGEN = 20, 48
KINDS = ("posterior", "anterior", "head")


def keep(kind: str, jitter: int) -> torch.Tensor:
    k = torch.ones(T.L)
    if kind == "posterior":
        k[: SPLIT + jitter] = 0
    elif kind == "anterior":
        k[SPLIT + jitter:] = 0
    else:
        k[: body.HEAD_END + 2 + jitter] = 0
    return k


def build(spec: dict, rng: np.random.Generator) -> life.Timeline:
    B = len(spec["csplus"])
    tl = life.Timeline(B)
    tl.free(GROW).mark("grown")
    tl.wound(body.keep_masks(spec["pre"], spec["pre_jitter"])).free(REGEN).mark("pre_regen")
    tl.condition(spec["csplus"], spec["pairing"], rng)
    tl.free(spec["delay"])
    cut = torch.stack([keep(k, int(j)) for k, j in zip(spec["split"], spec["split_jitter"])])[:, None]
    tl.wound(cut).free(FISSION_REGEN).mark("post_regen")
    tl.test("test", spec["csplus"], spec["order"]).mark("end")
    return tl


def loss(p, spec, tl, cfg):
    out = life.run(p, tl, T.fire_masks(spec["mask_seeds"], tl.t))
    anat = sum(body.anatomy_loss(out["snaps"][k]).mean() for k in ("grown", "pre_regen", "post_regen", "end"))
    want = torch.from_numpy((spec["pairing"] == "paired").astype(np.float32))
    ro = life.readout(out["resp"], tl.readouts["test"])
    mem = ((ro["plus"] - want) ** 2 + ro["minus"] ** 2).mean()
    reflex = torch.stack([((out["resp"][t] - 1) ** 2)[torch.from_numpy(m)].mean() for t, m in tl.reflex]).mean()
    base = (out["resp"][torch.from_numpy(tl.readouts["test"].rest)] ** 2).mean()
    total = anat + cfg.w_memory * mem + cfg.w_reflex * reflex + cfg.w_baseline * base
    post = torch.from_numpy(spec["split"] == "posterior") & (want > 0)
    return total, {"loss": total.item(), "anat": anat.item(), "mem": mem.item(),
                   "di_posterior": ro["di"][post].mean().item() if post.any() else float("nan")}


def train_f(init: str, seed: int, iterations: int = 1500, threads: int = 1, log=print) -> dict:
    cfg = train.Config(family="S", seed=seed, iterations=iterations, threads=threads, init=init)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, 7, 4])
    p = {k: v.clone().requires_grad_() for k, v in train.load(init)["tensors"].items()}
    opt = torch.optim.Adam(p.values(), lr=cfg.lr)
    hist, t0 = [], time.time()
    for i in range(iterations):
        spec = train.sample_life(cfg, rng)
        spec["split"] = rng.choice(KINDS, cfg.batch, p=[0.7, 0.15, 0.15])
        spec["split_jitter"] = rng.integers(-1, 2, cfg.batch)
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
            log(f"[F{seed}] {i:5d} loss {info['loss']:.4f} anat {info['anat']:.4f} mem {info['mem']:.4f} "
                f"DI(posterior) {info['di_posterior']:+.3f} {info['secs']:.0f}s", flush=True)
    cfgd = dataclasses.asdict(cfg)
    cfgd["family"] = "F"
    return {"params": {k: v.detach() for k, v in p.items()}, "history": hist, "config": cfgd, "seconds": time.time() - t0}


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1])
    res = train_f(f"weights/rule_S{seed}.json", seed, iterations=int(sys.argv[2]) if len(sys.argv) > 2 else 1500)
    train.save(res, f"weights/rule_F{seed}.json")
