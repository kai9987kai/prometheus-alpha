"""Meta-training of a cell rule: exact backpropagation through whole lives.

Every training life is: grow from one founder cell (32 steps); possibly a first wound and
regeneration (32 steps); differential conditioning (48 steps, paired / unpaired / naive);
a delay; possibly a second wound and regeneration (32 steps); a memory test (24 steps).

Two families of rules differ in a single line of the objective:

* **E (emergent)**: the memory test is scored only when the body was *not* wounded after
  conditioning. E rules learn and they regenerate, often in the same life, but are never asked
  to remember through a regeneration. Whether they do anyway is the open question.
* **S (selected)**: each S rule starts from its E sibling's weights and is trained further with
  the memory test also scored after a head or tail amputation: selection for remembering through
  one regeneration. S rules are the positive control and the substrate for the held-out tests
  (trunk fragments, repeated amputation, chimeras, voltage transplants), none of which is trained.

The rule never changes within a life, so all learning is state. Anatomy is scored in every
life, and the unconditioned response (R follows the US) and a zero baseline at rest give the
response channel its meaning.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time

import numpy as np
import torch

from . import body, life, provenance
from . import tissue as T

GROW, REGEN = 32, 32


@dataclasses.dataclass(frozen=True)
class Config:
    family: str = "E"
    seed: int = 0
    iterations: int = 2000
    batch: int = 32
    lr: float = 2e-3
    w_memory: float = 2.0
    w_reflex: float = 0.1
    w_baseline: float = 0.3
    w_cr: float = 0.0                                    # anticipatory CR during acquisition (off; see docs/DEVIATIONS.md)
    init: str | None = None                              # start from these weights (S rules start from their E sibling)
    p_pairing: tuple = (0.5, 0.35, 0.15)                 # paired, unpaired, naive
    p_pre_wound: tuple = (0.5, 0.2, 0.15, 0.15)          # none, head, tail, fragment
    p_post_wound: tuple = (0.35, 0.3, 0.15, 0.2)
    delay: tuple = (4, 16)
    threads: int = 4


def memory_weight(family: str, post: str) -> float:
    if family == "E":
        return 1.0 if post == "none" else 0.0
    if family == "S":
        return 1.0 if post in ("none", "head", "tail") else 0.0     # trunk fragments are held out
    raise ValueError(family)


def sample_life(cfg: Config, rng: np.random.Generator) -> dict:
    B = cfg.batch
    return {
        "csplus": rng.integers(0, 2, B),
        "pairing": rng.choice(life.PAIRINGS, B, p=cfg.p_pairing),
        "pre": rng.choice(body.WOUNDS, B, p=cfg.p_pre_wound),
        "post": rng.choice(body.WOUNDS, B, p=cfg.p_post_wound),
        "pre_jitter": body.random_jitter(rng, B),
        "post_jitter": body.random_jitter(rng, B),
        "order": rng.integers(0, 2, B),
        "delay": int(rng.integers(cfg.delay[0], cfg.delay[1] + 1)),
        "mask_seeds": rng.integers(0, 2 ** 31, B),
    }


def build(spec: dict, rng: np.random.Generator) -> life.Timeline:
    B = len(spec["csplus"])
    tl = life.Timeline(B)
    tl.free(GROW).mark("grown")
    tl.wound(body.keep_masks(spec["pre"], spec["pre_jitter"])).free(REGEN).mark("pre_regen")
    tl.condition(spec["csplus"], spec["pairing"], rng)
    tl.free(spec["delay"])
    tl.wound(body.keep_masks(spec["post"], spec["post_jitter"])).free(REGEN).mark("post_regen")
    tl.test("test", spec["csplus"], spec["order"]).mark("end")
    return tl


def loss(p: dict, spec: dict, tl: life.Timeline, cfg: Config) -> tuple[torch.Tensor, dict]:
    masks = T.fire_masks(spec["mask_seeds"], tl.t)
    out = life.run(p, tl, masks)
    anat = sum(body.anatomy_loss(out["snaps"][k]).mean() for k in ("grown", "pre_regen", "post_regen", "end"))
    ro = life.readout(out["resp"], tl.readouts["test"])
    want_plus = torch.from_numpy((spec["pairing"] == "paired").astype(np.float32))
    w = torch.tensor([memory_weight(cfg.family, k) for k in spec["post"]])
    mem = (w * ((ro["plus"] - want_plus) ** 2 + ro["minus"] ** 2)).sum() / w.sum().clamp_min(1)
    rf = [((out["resp"][t] - 1) ** 2)[torch.from_numpy(m)].mean() for t, m in tl.reflex]
    reflex = torch.stack(rf).mean() if rf else torch.zeros(())
    base = (out["resp"][torch.from_numpy(tl.readouts["test"].rest)] ** 2).mean()
    crs = [(out["resp"][t][torch.from_numpy(~np.isnan(g))] - torch.from_numpy(g[~np.isnan(g)])) ** 2 for t, g in tl.cr]
    cr = torch.cat(crs).mean() if crs else torch.zeros(())
    total = anat + cfg.w_memory * mem + cfg.w_reflex * reflex + cfg.w_baseline * base + cfg.w_cr * cr
    paired = torch.from_numpy(spec["pairing"] == "paired") & (w > 0)
    di = ro["di"][paired].mean().item() if paired.any() else float("nan")
    return total, {"loss": total.item(), "anat": anat.item(), "mem": mem.item(), "reflex": reflex.item(),
                   "base": base.item(), "cr": cr.item(), "di_paired_scored": di}


def normalise(g: torch.Tensor) -> torch.Tensor:
    return g / (g.norm() + 1e-8)


def train(cfg: Config, log=print) -> dict:
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng([cfg.seed, 7, 0 if cfg.family == "E" else 1])
    start = load(cfg.init)["tensors"] if cfg.init else T.init_params(cfg.seed)
    p = {k: v.clone().requires_grad_() for k, v in start.items()}
    opt = torch.optim.Adam(p.values(), lr=cfg.lr)

    def lr_at(i):
        warm = min(1.0, (i + 1) / 50)
        return cfg.lr * warm * (0.3 if i > 0.6 * cfg.iterations else 1) * (0.3 if i > 0.85 * cfg.iterations else 1)

    history, t0 = [], time.time()
    for i in range(cfg.iterations):
        spec = sample_life(cfg, rng)
        tl = build(spec, rng)
        total, info = loss(p, spec, tl, cfg)
        opt.zero_grad()
        total.backward()
        for v in p.values():
            v.grad = normalise(v.grad)
        for g in opt.param_groups:
            g["lr"] = lr_at(i)
        opt.step()
        if not math.isfinite(info["loss"]):
            raise FloatingPointError(f"non-finite loss at iteration {i}")
        if i % 25 == 0 or i == cfg.iterations - 1:
            info.update(it=i, secs=round(time.time() - t0, 1))
            history.append(info)
            log(f"[{cfg.family}{cfg.seed}] {i:5d} loss {info['loss']:.4f} anat {info['anat']:.4f} "
                f"mem {info['mem']:.4f} cr {info['cr']:.3f} reflex {info['reflex']:.3f} base {info['base']:.4f} "
                f"DI {info['di_paired_scored']:+.3f}  {info['secs']:.0f}s")
    return {"params": {k: v.detach() for k, v in p.items()}, "history": history,
            "config": dataclasses.asdict(cfg), "seconds": time.time() - t0}


def save(result: dict, path: str):
    params = {k: v.tolist() for k, v in result["params"].items()}
    doc = {"format": "prometheus-rule-v1", "config": result["config"], "seconds": result["seconds"],
           "history": result["history"], "n_params": T.n_params(result["params"]),
           "provenance": provenance.stamp(), "params": params}
    with open(path, "w") as f:
        json.dump(doc, f)


def load(path: str) -> dict:
    with open(path) as f:
        doc = json.load(f)
    doc["tensors"] = {k: torch.tensor(v, dtype=torch.float32) for k, v in doc["params"].items()}
    return doc
