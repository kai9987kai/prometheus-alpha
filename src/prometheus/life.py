"""Life histories: growth, Pavlovian conditioning, wounds, regeneration and memory tests.

A :class:`Timeline` lays out, step by step, what the world does to a batch of worms (which cue
is on, when the unconditioned stimulus arrives, whether gap junctions are open) and which
events happen between steps (a wound, a graft, a voltage transplant). :func:`run` plays a
timeline on a batch.

**Conditioning** is differential Pavlovian conditioning in 6 slots of 8 steps, in a random
order per worm: two CS+ slots, two CS- slots and two empty slots. A cue is on for the first 6
steps of its slot. The *paired* worm gets the US on steps 3-5 of each CS+ slot, overlapping the
cue. Its *unpaired* twin gets exactly the same US, at the same steps, but inside the empty slots,
so it sees the same cues and the same shocks without the contingency (the standard explicitly
unpaired control). A *naive* worm gets no US at all. Which of the two odours is the CS+ is
drawn per worm, so the association has to be learned within the life. Steps 1-2 of a cue slot
come before any US, so a response there is anticipatory: the conditioned response (CR) of
acquisition, expected only for a CS+ that has already been paired once.

**Testing** presents CS+ alone and CS- alone for 6 steps each, in a random order, with rests.
Behaviour is read over the last 3 steps of each cue. The discrimination index
``DI = response(CS+) - response(CS-)`` is measured within one worm, so any change of baseline
cancels (the "evoked" readout Supermix Expanse learned it needed).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from . import tissue as T

SLOT, N_SLOTS, CUE_ON, US_ON = 8, 6, 6, (3, 4, 5)
CR_STEPS = (1, 2)       # anticipatory window inside a cue slot, before any US can arrive
COND_T = SLOT * N_SLOTS
TEST_T = 24
PAIRINGS = ("paired", "unpaired", "naive")


@dataclasses.dataclass
class Readout:
    plus: np.ndarray        # (B, 3) step indices of the CS+ window
    minus: np.ndarray       # (B, 3) step indices of the CS- window
    rest: np.ndarray        # (k,) rest steps (no cue, no US)


class Timeline:
    def __init__(self, batch: int):
        self.B = batch
        self.cue: list[np.ndarray] = []
        self.us: list[np.ndarray] = []
        self.gj: list[np.ndarray] = []
        self.gj_now = np.ones(batch)
        self.events: dict[int, list] = {}
        self.marks: dict[str, int] = {}
        self.readouts: dict[str, Readout] = {}
        self.reflex: list[tuple[int, np.ndarray]] = []   # (step, which worms have the US on)
        self.cr: list[tuple[int, np.ndarray]] = []       # (step, anticipatory target per worm)

    @property
    def t(self) -> int:
        return len(self.us)

    def _push(self, cue, us):
        self.cue.append(np.asarray(cue, np.float32))
        self.us.append(np.asarray(us, np.float32))
        self.gj.append(self.gj_now.astype(np.float32).copy())

    def free(self, n: int):
        for _ in range(n):
            self._push(np.zeros((self.B, 2)), np.zeros(self.B))
        return self

    def at(self, fn):
        """Apply ``fn(x) -> x`` to the state just before the next step."""
        self.events.setdefault(self.t, []).append(fn)
        return self

    def wound(self, keep: torch.Tensor):
        return self.at(lambda x: x * keep)

    def mark(self, name: str):
        self.marks[name] = self.t
        return self

    def condition(self, csplus: np.ndarray, pairing, rng: np.random.Generator):
        """48 steps of differential conditioning. ``pairing`` is one name or a list per worm."""
        pairing = [pairing] * self.B if isinstance(pairing, str) else list(pairing)
        orders = np.stack([rng.permutation(np.array(list("++--00"))) for _ in range(self.B)])
        self.condition_orders = orders
        paired_before = np.zeros(self.B, bool)
        for s in range(N_SLOTS):
            is_cs = np.isin(orders[:, s], ["+", "-"])
            want = (orders[:, s] == "+") & paired_before
            for k in range(SLOT):
                if k in CR_STEPS and is_cs.any():
                    self.cr.append((self.t, np.where(is_cs, want.astype(np.float32), np.nan)))
                cue = np.zeros((self.B, 2))
                us = np.zeros(self.B)
                for i in range(self.B):
                    kind = orders[i, s]
                    if k < CUE_ON and kind in "+-":
                        cue[i, csplus[i] if kind == "+" else 1 - csplus[i]] = 1
                    if k in US_ON and ((pairing[i] == "paired" and kind == "+") or
                                       (pairing[i] == "unpaired" and kind == "0")):
                        us[i] = 1
                if us.any() and k >= US_ON[1]:           # give the reflex one step of latency
                    self.reflex.append((self.t, us > 0))
                self._push(cue, us)
            paired_before |= (orders[:, s] == "+") & (np.asarray(pairing) == "paired")
        return self

    def test(self, name: str, csplus: np.ndarray, order: np.ndarray):
        """24 steps: rest 4, first cue 6, rest 6, second cue 6, rest 2. order 0 = CS+ first."""
        t0 = self.t
        first = np.where(order == 0, csplus, 1 - csplus)
        second = 1 - first
        for k in range(TEST_T):
            cue = np.zeros((self.B, 2))
            if 4 <= k < 10:
                cue[np.arange(self.B), first] = 1
            elif 16 <= k < 22:
                cue[np.arange(self.B), second] = 1
            self._push(cue, np.zeros(self.B))
        w1 = t0 + np.array([7, 8, 9])
        w2 = t0 + np.array([19, 20, 21])
        plus = np.where((order == 0)[:, None], w1, w2)
        minus = np.where((order == 0)[:, None], w2, w1)
        self.readouts[name] = Readout(plus, minus, t0 + np.array([2, 3, 13, 14, 15]))
        return self


def run(p: dict, tl: Timeline, masks: torch.Tensor, phys: T.Physics = T.Physics(),
        x0: torch.Tensor | None = None, t0: int = 0) -> dict:
    """Play ``tl`` on a batch. ``masks[t0 + t]`` is the update mask of step t."""
    x = T.founder(tl.B) if x0 is None else x0
    cue = torch.from_numpy(np.stack(tl.cue)) if tl.cue else torch.zeros(0, tl.B, 2)
    us = torch.from_numpy(np.stack(tl.us)) if tl.us else torch.zeros(0, tl.B)
    gj = torch.from_numpy(np.stack(tl.gj)) if tl.gj else torch.zeros(0, tl.B)
    at_mark = {}
    for name, t in tl.marks.items():
        at_mark.setdefault(t, []).append(name)
    snaps, resp = {}, []
    for t in range(tl.t):
        for name in at_mark.get(t, []):
            snaps[name] = x
        for fn in tl.events.get(t, []):
            x = fn(x)
        g = gj[t]
        x = T.step(p, x, cue[t], us[t], masks[t0 + t], phys, None if bool((g == 1).all()) else g)
        resp.append(T.response(x))
    for name in at_mark.get(tl.t, []):
        snaps[name] = x
    for fn in tl.events.get(tl.t, []):
        x = fn(x)
    return {"resp": torch.stack(resp) if resp else torch.zeros(0, tl.B), "snaps": snaps, "x": x}


def readout(resp: torch.Tensor, r: Readout) -> dict:
    """Per-worm response to CS+, to CS-, and the discrimination index."""
    b = torch.arange(resp.shape[1])[:, None]
    plus = resp[torch.from_numpy(r.plus), b].mean(-1)
    minus = resp[torch.from_numpy(r.minus), b].mean(-1)
    return {"plus": plus, "minus": minus, "di": plus - minus}
