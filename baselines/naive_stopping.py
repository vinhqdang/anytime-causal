"""Naive optional-stopping baseline (algorithm.md sec 5.4).

The anytime-validity stress test: run a standard fixed-sample independence test
(HSIC permutation) on the growing sample and "declare dependence" the first time
p < alpha, with NO multiplicity / no-repeated-looks correction. Under H0 this
inflates the type-I error far above alpha and the inflation grows with the
number of looks -- exactly the failure ORACLE's martingale avoids.

This is the most persuasive single comparison for the anytime claim, so it is
built from scratch here (no external dependency) to keep it transparent.
"""

from __future__ import annotations

import numpy as np


def _rbf_gram(x: np.ndarray, sigma: float) -> np.ndarray:
    sq = (x[:, None] - x[None, :]) ** 2
    return np.exp(-sq / (2 * sigma ** 2))


def _median_sigma(x: np.ndarray) -> float:
    n = len(x)
    if n > 200:
        x = x[np.linspace(0, n - 1, 200).astype(int)]
    d = np.abs(x[:, None] - x[None, :])
    iu = np.triu_indices(len(x), 1)
    med = np.median(d[iu])
    return float(med) if med > 0 else 1.0


def hsic_pvalue(u: np.ndarray, v: np.ndarray, n_perm: int = 200,
                rng: np.random.Generator | None = None) -> float:
    """Permutation HSIC test p-value for H0: u _||_ v."""
    rng = rng or np.random.default_rng(0)
    n = len(u)
    if n < 6:
        return 1.0
    su, sv = _median_sigma(u), _median_sigma(v)
    K = _rbf_gram(u, su)
    L = _rbf_gram(v, sv)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    stat = np.sum(Kc * L) / (n * n)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        Lp = L[np.ix_(perm, perm)]
        s = np.sum(Kc * Lp) / (n * n)
        if s >= stat:
            count += 1
    return (count + 1) / (n_perm + 1)


class NaiveSequentialTester:
    """Repeated-looks p-value tester for one pair, no correction.

    Recomputes the HSIC test every ``stride`` observations on the *growing*
    sample (expanding window) and records the first time p < alpha.
    """

    def __init__(self, alpha: float = 0.05, stride: int = 25,
                 max_window: int = 600, n_perm: int = 100, seed: int = 0):
        self.alpha = alpha
        self.stride = stride
        self.max_window = max_window
        self.n_perm = n_perm
        self.rng = np.random.default_rng(seed)
        self.u: list = []
        self.v: list = []
        self.t = 0
        self.rejected = False
        self.reject_time = None

    def update(self, u: float, v: float) -> bool:
        self.t += 1
        self.u.append(u)
        self.v.append(v)
        if self.rejected:
            return True
        if self.t % self.stride == 0 and self.t >= 6:
            uu = np.array(self.u[-self.max_window:])
            vv = np.array(self.v[-self.max_window:])
            p = hsic_pvalue(uu, vv, self.n_perm, self.rng)
            if p < self.alpha:
                self.rejected = True
                self.reject_time = self.t
        return self.rejected
