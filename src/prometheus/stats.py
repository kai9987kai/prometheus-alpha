"""Paired effect sizes, sign-flip permutation tests, Holm, Stouffer.

Adapted from Morpheus (kai9987kai/morpheus, MIT): same estimators, so numbers are comparable
across the two labs. One addition carried over from Ghost in the Machine: a confirmatory claim
needs a smallest effect size of interest (SESOI) as well as a p-value, so a tiny, precisely
estimated effect cannot pass on sample size alone.
"""

from __future__ import annotations

import math

import numpy as np


def sign_flip_p(d, n_perm: int = 10000, rng=None, alternative: str = "greater") -> float:
    d = np.asarray(d, float)
    n = d.size
    obs = d.mean()
    if n <= 16:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        rng = rng or np.random.default_rng(0)
        signs = rng.choice([-1, 1], (n_perm, n))
    null = (signs * d).mean(axis=1)
    if alternative == "greater":
        k = (null >= obs - 1e-12).sum()
    elif alternative == "less":
        k = (null <= obs + 1e-12).sum()
    else:
        k = (np.abs(null) >= abs(obs) - 1e-12).sum()
    if n <= 16:
        return float(k / len(null))
    return float((k + 1) / (len(null) + 1))


def paired(diff, n_boot: int = 10000, n_perm: int = 10000, seed: int = 0, alternative: str = "greater") -> dict:
    """Summary of a per-worm difference vector: mean, bootstrap 95% CI, d_z, fraction positive, p."""
    d = np.asarray(diff, float)
    n = d.size
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, (n_boot, n), replace=True).mean(axis=1)
    sd = d.std(ddof=1) if n > 1 else float("nan")
    return {
        "n": int(n),
        "mean": float(d.mean()),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "d_z": float(d.mean() / sd) if sd and sd > 0 else 0.0,
        "frac_positive": float((d > 0).mean()),
        "p": sign_flip_p(d, n_perm, rng, alternative),
        "alternative": alternative,
    }


def mean_ci(x, seed: int = 0, n_boot: int = 10000) -> dict:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n_boot, x.size)).mean(axis=1)
    return {"mean": float(x.mean()), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))], "n": int(x.size)}


def holm(pvals: dict, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down. {name: p} -> {name: {"p", "p_adj", "reject"}}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running, stop = len(items), {}, 0.0, False
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        reject = not stop and p <= alpha / (m - i)
        stop = stop or not reject
        out[k] = {"p": p, "p_adj": running, "reject": reject}
    return out


def _norm_ppf(p: float) -> float:
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def stouffer(pvals) -> tuple[float, float]:
    zs = [_norm_ppf(1 - min(max(p, 1e-15), 1 - 1e-15)) for p in pvals]
    z = sum(zs) / math.sqrt(len(zs))
    return float(z), float(0.5 * math.erfc(z / math.sqrt(2)))


def fmt_p(p: float) -> str:
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"
