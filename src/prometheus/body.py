"""The target body plan, wounds and anatomical scores.

The worm occupies sites [4, 36) of the 40-site grid: head [4, 12), trunk [12, 28), tail [28, 36).
"""

from __future__ import annotations

import numpy as np
import torch

from .tissue import ALPHA, HEAD, L, TAIL

BODY_START, HEAD_END, TRUNK_END, BODY_END = 4, 12, 28, 36
WOUNDS = ("none", "head", "tail", "fragment")


def target() -> tuple[torch.Tensor, torch.Tensor]:
    """(alpha (L,), identity one-hot (3, L)) of the intact body."""
    a = torch.zeros(L)
    a[BODY_START:BODY_END] = 1
    ident = torch.zeros(3, L)
    ident[0, BODY_START:HEAD_END] = 1
    ident[1, HEAD_END:TRUNK_END] = 1
    ident[2, TRUNK_END:BODY_END] = 1
    return a, ident


_A, _I = target()


def anatomy_loss(x: torch.Tensor) -> torch.Tensor:
    """Per-worm squared error of alive-ness and of region identity inside the target body. (B,)"""
    a = x[:, ALPHA].clamp(0, 1)
    la = ((a - _A) ** 2).mean(-1)
    li = (((x[:, HEAD:TAIL + 1] - _I) ** 2) * _A).sum((1, 2)) / _A.sum()
    return la + li


def region(x: torch.Tensor) -> torch.Tensor:
    """Per-site region label: -1 not body, 0 head, 1 trunk, 2 tail. (B, L)

    A site is body when its own alpha exceeds 0.1. The alive mask used by the physics is wider:
    it also includes the growth frontier, empty sites next to a body cell, which update and pass
    current but are not yet tissue."""
    lab = x[:, HEAD:TAIL + 1].argmax(1)
    return torch.where(x[:, ALPHA] > 0.1, lab, torch.full_like(lab, -1))


def anatomy_scores(x: torch.Tensor) -> dict:
    """Body IoU with the target and region accuracy inside the target body, per worm."""
    lab = region(x)
    live = lab >= 0
    tgt = _A.bool()
    inter = (live & tgt).sum(-1).float()
    union = (live | tgt).sum(-1).float()
    want = _I.argmax(0)
    acc = ((lab == want) & tgt).sum(-1).float() / tgt.sum()
    head_cells = (lab == 0).sum(-1).float()
    return {"iou": inter / union, "region_acc": acc, "head_cells": head_cells}


def keep_mask(kind: str, jitter: int = 0) -> torch.Tensor:
    """Sites that survive a wound, (L,). ``jitter`` moves the cut by a few sites."""
    k = torch.ones(L)
    if kind in ("head", "fragment"):
        k[: HEAD_END + jitter] = 0
    if kind in ("tail", "fragment"):
        k[TRUNK_END + jitter:] = 0
    if kind not in WOUNDS:
        raise ValueError(kind)
    return k


def keep_masks(kinds, jitters) -> torch.Tensor:
    """(B, 1, L) keep masks for a batch of wounds."""
    return torch.stack([keep_mask(k, int(j)) for k, j in zip(kinds, jitters)])[:, None]


def random_jitter(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(-2, 3, n)
