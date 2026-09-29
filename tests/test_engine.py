"""Engine invariants: the physics, the wounds, the grafts and common random numbers."""

import numpy as np
import pytest
import torch

from prometheus import body, experiments as X, life
from prometheus import tissue as T


def grown_worm(batch=1):
    """A hand-built intact worm with the target anatomy."""
    a, ident = body.target()
    x = torch.zeros(batch, T.C, T.L)
    x[:, T.ALPHA] = a
    x[:, T.HEAD:T.TAIL + 1] = ident
    return x


def test_zero_rule_does_nothing():
    p = T.init_params(0)
    x = T.founder(3)
    fire = torch.ones(3, 1, T.L)
    y = T.step(p, x, torch.zeros(3, 2), torch.zeros(3), fire)
    assert torch.equal(x, y)


def test_voltage_diffuses_only_within_the_alive_mask_and_is_conserved():
    p = T.init_params(0)
    x = grown_worm()
    x[0, T.V, 20] = 4.0
    y = T.step(p, x, torch.zeros(1, 2), torch.zeros(1), torch.ones(1, 1, T.L))
    assert y[0, T.V, 19] > 0 and y[0, T.V, 21] > 0
    assert torch.isclose(y[0, T.V].sum(), torch.tensor(4.0), atol=1e-5)
    blocked = T.step(p, x, torch.zeros(1, 2), torch.zeros(1), torch.ones(1, 1, T.L), gj=torch.zeros(1))
    assert blocked[0, T.V, 19] == 0 and blocked[0, T.V, 20] == 4.0
    x2 = grown_worm()
    x2[0, T.V, body.BODY_START] = 3.0
    y2 = T.step(p, x2, torch.zeros(1, 2), torch.zeros(1), torch.ones(1, 1, T.L))
    assert y2[0, T.V, body.BODY_START - 1] > 0          # the growth frontier is in the alive mask
    assert y2[0, T.V, : body.BODY_START - 1].abs().sum() == 0   # nothing beyond it
    assert torch.isclose(y2[0, T.V].sum(), torch.tensor(3.0), atol=1e-5)


def test_only_head_tissue_senses_cues():
    x = grown_worm()
    w = T.head_weight(x)[0, 0]
    assert w[body.BODY_START:body.HEAD_END].min() == 1
    assert w[body.HEAD_END:].max() == 0


def test_headless_worm_has_no_behaviour():
    x = grown_worm()
    x[:, T.R] = 1.0
    assert T.response(x).item() == pytest.approx(1.0)
    cut, residual = X.amputate(x, "head")
    assert residual.max() == 0
    assert T.response(cut).item() == 0.0


def test_complete_amputation_removes_every_head_cell_plus_margin():
    x = grown_worm()
    x[:, T.HEAD, 12] = 2.0          # a stray head cell in the trunk moves the cut back
    cut, _ = X.amputate(x, "head")
    assert cut[0, T.ALPHA, :15].sum() == 0 and cut[0, T.ALPHA, 15] == 1
    frag, _ = X.amputate(grown_worm(), "fragment")
    live = frag[0, T.ALPHA].nonzero().flatten()
    assert live.min() == body.HEAD_END + 2 and live.max() == body.TRUNK_END - 1


def test_graft_takes_the_front_from_the_head_donor():
    a, b = torch.zeros(1, T.C, T.L), torch.ones(1, T.C, T.L)
    g = X.graft(a, b)
    assert g[0, :, : body.HEAD_END].sum() == 0 and torch.all(g[0, :, body.HEAD_END:] == 1)


def test_fire_masks_are_common_random_numbers():
    m1 = T.fire_masks([5, 9], 50)
    m2 = T.fire_masks([9], 50)
    assert torch.equal(m1[:, 1], m2[:, 0])
    assert 0.4 < m1.mean().item() < 0.6


def test_conditioning_twins_differ_only_in_us_timing():
    B = 64
    cs = np.random.default_rng(0).integers(0, 2, B)
    a = life.Timeline(B).condition(cs, "paired", np.random.default_rng(1))
    b = life.Timeline(B).condition(cs, "unpaired", np.random.default_rng(1))
    n = life.Timeline(B).condition(cs, "naive", np.random.default_rng(1))
    ca, cb = np.stack(a.cue), np.stack(b.cue)
    ua, ub = np.stack(a.us), np.stack(b.us)
    assert np.array_equal(ca, cb)
    assert np.array_equal(ua.sum(0), ub.sum(0)) and ua.sum(0).min() == 6
    assert np.stack(n.us).sum() == 0
    plus = ca[np.arange(len(ca))[:, None], np.arange(B)[None, :], cs[None, :]]      # CS+ on
    assert np.all(ua[ua > 0] == plus[ua > 0])                                        # paired US always with CS+
    assert (ub * ca.sum(-1)).sum() == 0                                              # unpaired US never with a cue


def test_cr_targets_only_after_a_first_pairing():
    B = 16
    cs = np.zeros(B, int)
    tl = life.Timeline(B).condition(cs, "paired", np.random.default_rng(3))
    first_plus = [list(tl.condition_orders[i]).index("+") for i in range(B)]
    for t, target in tl.cr:
        slot = t // life.SLOT
        for i in range(B):
            if np.isnan(target[i]):
                continue
            assert target[i] == (1.0 if tl.condition_orders[i, slot] == "+" and slot > first_plus[i] else 0.0)
    un = life.Timeline(B).condition(cs, "unpaired", np.random.default_rng(3))
    assert all(np.nansum(t) == 0 for _, t in un.cr)


def test_test_readouts_follow_the_order():
    B = 4
    tl = life.Timeline(B).test("t", np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    r = tl.readouts["t"]
    cue = np.stack(tl.cue)
    for i, cs in enumerate([0, 0, 1, 1]):
        assert all(cue[t, i, cs] == 1 for t in r.plus[i])
        assert all(cue[t, i, 1 - cs] == 1 for t in r.minus[i])
    assert all(cue[t].sum() == 0 for t in r.rest)


def test_twins_share_noise_and_founder():
    p = T.init_params(1)
    p["W2"] = torch.randn(T.C, T.HIDDEN) * 0.05
    co = X.cohort(42, 4)
    P, U = X.twins(p, co)
    grow_p = X.Worms.found(p, co).free(X.GROW)
    grow_u = X.Worms.found(p, co).free(X.GROW)
    assert torch.equal(grow_p.x, grow_u.x)
    assert P.t == U.t == X.GROW + life.COND_T + X.DELAY
