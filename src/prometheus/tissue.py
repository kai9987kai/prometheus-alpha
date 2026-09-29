"""The cell rule, perception and gap-junction physics of a one-dimensional regenerating worm.

A worm is a row of cells on a grid of ``L`` sites. Every cell carries ``C`` channels and runs
the same small rule (a two-layer MLP). A cell perceives its own state and its two neighbours
(identity, gradient and Laplacian of every channel) plus three signals from the world:

* cue A and cue B (two "odours"), which a cell can sense only when it is more than half head
  tissue (gate ``clamp(2*head - 1, 0, 1)``): sensing is a property of head tissue, so a
  headless body cannot smell at all;
* the unconditioned stimulus (US, a "shock"), broadcast to every living cell.

The rule proposes a change to all ``C`` channels, applied asynchronously (each cell fires with
probability ``fire_rate``). Voltage then diffuses through gap junctions between living
neighbours by a fixed physical law that the rule cannot change, only use. Cells die when no
neighbour is alive (the growing-NCA alive mask, Mordvintsev et al. 2020).

The learned rule is fixed for a worm's whole life. Anything the worm learns, it learns in its
*state*, which is what makes "does the memory survive losing the head?" a question about where
in the body that state lives.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

L = 40                          # grid sites
C = 16                          # channels per cell
ALPHA, V, HEAD, TRUNK, TAIL, R = range(6)
HIDDEN0 = 6                     # channels 6..15 are hidden, with no target
N_IN = 3                        # cue A, cue B, US
PERC = 3 * C + N_IN             # perception vector length
HIDDEN = 64
FOUNDER = 20                    # where the founder cell sits

CHANNEL_NAMES = ["alpha", "voltage", "head", "trunk", "tail", "response"] + [f"h{i}" for i in range(C - HIDDEN0)]


@dataclasses.dataclass(frozen=True)
class Physics:
    coupling: float = 0.25      # gap-junction conductance per substep (fixed law)
    substeps: int = 2
    fire_rate: float = 0.5
    clip: float = 5.0


def init_params(seed: int, hidden: int = HIDDEN) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "W1": torch.randn(hidden, PERC, generator=g) / PERC ** 0.5,
        "b1": torch.zeros(hidden),
        "W2": torch.zeros(C, hidden),      # zero-initialised action head: a new rule does nothing
        "b2": torch.zeros(C),
    }


def n_params(p: dict) -> int:
    return int(sum(v.numel() for v in p.values()))


def founder(batch: int) -> torch.Tensor:
    x = torch.zeros(batch, C, L)
    x[:, ALPHA, FOUNDER] = 1.0
    x[:, HIDDEN0:, FOUNDER] = 1.0
    return x


def shift(x: torch.Tensor):
    """Left (i-1) and right (i+1) neighbours, zero beyond the grid."""
    left = F.pad(x[..., :-1], (1, 0))
    right = F.pad(x[..., 1:], (0, 1))
    return left, right


def alive(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool1d(x[:, ALPHA:ALPHA + 1], 3, 1, 1) > 0.1


def gate(head: torch.Tensor) -> torch.Tensor:
    """Sensory gate: 0 up to half head identity, rising to 1 at full head identity."""
    return (2 * head - 1).clamp(0, 1)


def head_weight(x: torch.Tensor) -> torch.Tensor:
    """How much each cell counts as a sensing, behaving head cell (0..1). Shape (B, 1, L)."""
    return gate(x[:, HEAD:HEAD + 1]) * alive(x).float()


def response(x: torch.Tensor) -> torch.Tensor:
    """The worm's behaviour: its head-weighted mean response channel. Shape (B,).

    Behaviour is read from the head, as a planarian's is driven by its brain. A worm with no
    head tissue has no behaviour (0)."""
    w = head_weight(x)[:, 0]
    s = w.sum(-1)
    return torch.where(s > 0.5, (x[:, R] * w).sum(-1) / s.clamp_min(1e-6), torch.zeros_like(s))


def step(p: dict, x: torch.Tensor, cue: torch.Tensor, us: torch.Tensor, fire: torch.Tensor,
         phys: Physics = Physics(), gj: torch.Tensor | None = None) -> torch.Tensor:
    """One update of every worm in the batch.

    x    (B, C, L) state
    cue  (B, 2)    cue A / cue B intensity this step
    us   (B,)      unconditioned stimulus this step
    fire (B, 1, L) asynchronous update mask (0/1)
    gj   (B,)      optional per-worm multiplier on gap-junction conductance (0 = blocked)
    """
    pre = alive(x)
    pre_f = pre.float()
    left, right = shift(x)
    perc = torch.cat([x, 0.5 * (right - left), left + right - 2 * x], 1)
    sense = cue[:, :, None] * (gate(x[:, HEAD:HEAD + 1]) * pre_f)
    inp = torch.cat([perc, sense, us[:, None, None] * pre_f], 1)
    hid = F.relu(F.conv1d(inp, p["W1"][:, :, None], p["b1"]))
    x = x + F.conv1d(hid, p["W2"][:, :, None], p["b2"]) * fire

    g = phys.coupling if gj is None else phys.coupling * gj[:, None, None]
    al, ar = shift(pre_f)
    v = x[:, V:V + 1]
    for _ in range(phys.substeps):
        vl, vr = shift(v)
        v = v + g * pre_f * (al * (vl - v) + ar * (vr - v))
    x = torch.cat([x[:, :V], v, x[:, V + 1:]], 1)

    x = x * (pre & alive(x)).float()
    return x.clamp(-phys.clip, phys.clip)


def fire_masks(seeds, T: int, rate: float = 0.5) -> torch.Tensor:
    """Per-worm asynchronous update masks, (T, B, 1, L). Worm i's masks depend only on seeds[i],
    so the same worm under two conditions sees identical noise (common random numbers)."""
    out = torch.empty(T, len(seeds), 1, L)
    for i, s in enumerate(seeds):
        g = torch.Generator().manual_seed(int(s))
        out[:, i] = (torch.rand(T, 1, L, generator=g) < rate).float()
    return out
