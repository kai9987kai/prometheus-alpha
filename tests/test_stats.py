import numpy as np

from prometheus import stats


def test_sign_flip_exact_small_n():
    assert stats.sign_flip_p([1.0] * 5) == 1 / 32
    assert stats.sign_flip_p([-1.0] * 5) == 1.0


def test_sign_flip_is_calibrated_under_the_null():
    rng = np.random.default_rng(0)
    ps = np.array([stats.sign_flip_p(rng.standard_normal(40), n_perm=2000, rng=np.random.default_rng(k)) for k in range(300)])
    assert 0.02 < (ps <= 0.05).mean() < 0.09


def test_two_sided():
    assert stats.sign_flip_p(-np.ones(20), alternative="two-sided") < 0.001
    assert stats.sign_flip_p(-np.ones(20), alternative="greater") > 0.99


def test_holm_steps_down():
    r = stats.holm({"a": 0.001, "b": 0.02, "c": 0.04}, alpha=0.05)
    assert r["a"]["reject"] and r["b"]["reject"] and r["c"]["reject"]
    r = stats.holm({"a": 0.001, "b": 0.04, "c": 0.04}, alpha=0.05)
    assert r["a"]["reject"] and not r["b"]["reject"] and not r["c"]["reject"]


def test_paired_summary():
    s = stats.paired(np.full(30, 0.5) + np.linspace(-0.1, 0.1, 30))
    assert abs(s["mean"] - 0.5) < 1e-9 and s["frac_positive"] == 1.0 and s["p"] < 0.001
