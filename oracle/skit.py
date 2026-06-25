"""SKIT: Sequential Kernelized Independence Testing by betting.

Reference: Podkopaev, Bloebaum, Kasiviswanathan, Ramdas (ICML 2023),
"Sequential Kernelized Independence Testing".

This module implements the *valid* betting wealth process that replaces the
broken ``exp(HSIC*lambda)`` e-value of the original ORACLE draft. The wealth is
a nonnegative supermartingale (in fact a martingale) under H0: U _||_ V, so
``K_t >= 1/alpha`` is an anytime-valid rejection by Ville's inequality.

Construction (rigorous, exchangeability-based)
----------------------------------------------
We process the paired stream two observations at a time. Given two fresh pairs
``(u_a, v_a)`` and ``(u_b, v_b)`` we form

    joint points    P = {(u_a, v_a), (u_b, v_b)}        # real pairs
    product points  Q = {(u_a, v_b), (u_b, v_a)}        # swapped pairs

Under H0 (U _||_ V) the joint distribution equals the product distribution, so
the multiset {P, Q} is *exchangeable*: relabelling joint<->product does not
change the law. The bet payoff

    g = 0.5 * ( mean_{x in P} f_t(x) - mean_{x in Q} f_t(x) )

uses a *previsible* witness function ``f_t`` (mean-embedding difference between
joint and product, estimated from data strictly before this bet, clipped to
[-1, 1]). Exchangeability gives ``E[g | F_{t-1}] = 0`` under H0 for ANY
previsible ``f_t``; clipping keeps ``|g| <= 1`` so ``1 + lambda*g > 0`` and the
wealth stays nonnegative. Validity therefore does not depend on witness quality
-- only power does.

The witness uses Random Fourier Features (RFF) for the RBF kernel so per-step
cost is O(D^2) with O(D^2) memory (no growing sample), giving a genuinely
streaming test.
"""

from __future__ import annotations

import numpy as np

_LOG3 = np.log(3.0)


def _median_heuristic(samples: np.ndarray) -> float:
    """RBF bandwidth via the median pairwise-distance heuristic."""
    n = len(samples)
    if n < 2:
        return 1.0
    # subsample for cost control
    if n > 200:
        idx = np.linspace(0, n - 1, 200).astype(int)
        samples = samples[idx]
    diffs = samples[:, None, :] - samples[None, :, :]
    sq = np.sum(diffs ** 2, axis=-1)
    iu = np.triu_indices(len(samples), k=1)
    med = np.median(sq[iu])
    if med <= 0 or not np.isfinite(med):
        med = 1.0
    return float(np.sqrt(med / 2.0))  # sigma s.t. exp(-||.||^2/(2 sigma^2))


class _RFF:
    """Random Fourier Features for the RBF kernel k(x,x')=exp(-||x-x'||^2/2sig^2)."""

    def __init__(self, dim: int, n_features: int, sigma: float, rng: np.random.Generator):
        self.D = n_features
        self.sigma = max(sigma, 1e-6)
        self.W = rng.standard_normal((dim, n_features)) / self.sigma
        self.b = rng.uniform(0, 2 * np.pi, n_features)
        self.scale = np.sqrt(2.0 / n_features)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (dim,) -> (D,)
        return self.scale * np.cos(x @ self.W + self.b)


class SKIT:
    """Independence-by-betting test for H0: U _||_ V.

    Parameters
    ----------
    dim_u, dim_v : int
        Dimensions of the two arguments.
    alpha : float
        Test level; reject when wealth >= 1/alpha.
    n_features : int
        RFF count for the witness embedding.
    warmup : int
        Number of raw observations used (previsibly) to set bandwidth & seed
        the running witness statistics before betting starts.
    bet : {"mixture", "agrapa", "fixed"}
        Bet-sizing strategy. ``mixture`` averages wealth over a grid of lambda
        (robust, parameter-free, default). ``agrapa`` is previsible
        approximate-GRAPA. ``fixed`` uses ``lam``.
    lam : float
        Fixed bet size when ``bet="fixed"``.
    seed : int
        RNG seed for the RFF draw (kept separate so the test is reproducible).
    """

    LAMBDA_GRID = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9])

    def __init__(
        self,
        dim_u: int = 1,
        dim_v: int = 1,
        alpha: float = 0.05,
        n_features: int = 64,
        warmup: int = 40,
        bet: str = "mixture",
        lam: float = 0.5,
        standardize: bool = True,
        seed: int = 0,
    ):
        self.dim_u = dim_u
        self.dim_v = dim_v
        self.alpha = alpha
        self.n_features = n_features
        self.warmup = warmup
        self.bet = bet
        self.lam_fixed = lam
        self.standardize = standardize
        self.rng = np.random.default_rng(seed)

        self._reset_state()

    # ------------------------------------------------------------------ #
    def _reset_state(self) -> None:
        self._warm_u: list = []
        self._warm_v: list = []
        self._initialised = False
        self._phi_u = None
        self._phi_v = None
        self._n = 0
        self._mean_u = None       # (D,)
        self._mean_v = None       # (D,)
        self._M = None            # (D,D) running mean of outer(phi_u, phi_v)
        self._pending = None      # pending (u, v) awaiting a partner

        # wealth bookkeeping
        self.log_wealth = 0.0
        self._log_w_grid = np.zeros(len(self.LAMBDA_GRID))  # per-lambda log-wealth
        self.max_log_wealth = 0.0
        self.rejected = False
        self.reject_time = None
        self.last_log_increment = 0.0
        self.n_steps = 0          # observations consumed
        self.n_bets = 0           # bets placed

        # agrapa accumulators
        self._g_sum = 0.0
        self._g2_sum = 0.0

        # running witness scale (previsible standardisation for power)
        self._f_sq_sum = 0.0
        self._f_count = 0

    def reset(self) -> None:
        """Restart wealth from K=1 (used after a CUSUM alarm)."""
        self._reset_state()

    # ------------------------------------------------------------------ #
    @property
    def wealth(self) -> float:
        return float(np.exp(min(self.log_wealth, 700.0)))

    def _as_vec(self, x, dim) -> np.ndarray:
        a = np.atleast_1d(np.asarray(x, dtype=float))
        if a.shape[0] != dim:
            a = a.reshape(dim)
        return a

    def _initialise(self) -> None:
        U = np.array(self._warm_u)
        V = np.array(self._warm_v)
        sig_u = _median_heuristic(U)
        sig_v = _median_heuristic(V)
        self._phi_u = _RFF(self.dim_u, self.n_features, sig_u, self.rng)
        self._phi_v = _RFF(self.dim_v, self.n_features, sig_v, self.rng)
        # seed running stats from warmup buffer (all previsible)
        Pu = np.array([self._phi_u(u) for u in U])      # (n, D)
        Pv = np.array([self._phi_v(v) for v in V])
        self._n = len(U)
        self._mean_u = Pu.mean(axis=0)
        self._mean_v = Pv.mean(axis=0)
        self._M = (Pu.T @ Pv) / self._n
        self._initialised = True

    def _witness_raw(self, pu: np.ndarray, pv: np.ndarray) -> float:
        """Raw mean-embedding-difference witness (HSIC-style), unbounded."""
        joint = pu @ self._M @ pv
        prod = (self._mean_u @ pu) * (self._mean_v @ pv)
        return float(joint - prod)

    def _witness_scale(self) -> float:
        if not self.standardize:
            return 1.0
        if self._f_count < 5:
            return 1.0
        return float(np.sqrt(self._f_sq_sum / self._f_count) + 1e-8)

    def _witness(self, pu: np.ndarray, pv: np.ndarray) -> float:
        """Previsible standardised witness, clipped to [-1, 1].

        Standardising by the running witness scale (estimated from past
        evaluations only) turns a small-magnitude HSIC signal into an order-1
        payoff, giving the bet real power. The scale is previsible, so under H0
        the swap symmetry still gives E[g | F_{t-1}] = 0.
        """
        raw = self._witness_raw(pu, pv)
        return float(np.clip(raw / self._witness_scale(), -1.0, 1.0))

    def _ingest_joint(self, pu: np.ndarray, pv: np.ndarray) -> None:
        """Update running mean-embedding statistics with a real joint point."""
        self._n += 1
        eta = 1.0 / self._n
        self._mean_u += eta * (pu - self._mean_u)
        self._mean_v += eta * (pv - self._mean_v)
        self._M += eta * (np.outer(pu, pv) - self._M)

    def _bet_lambda(self) -> float:
        if self.bet == "fixed":
            return self.lam_fixed
        if self.bet == "agrapa":
            if self._g2_sum <= 0:
                return 0.0
            lam = self._g_sum / (self._g2_sum + 1e-12)
            return float(np.clip(lam, 0.0, 0.9))
        return 0.0  # mixture handled separately

    def update(self, u, v) -> float:
        """Feed one paired observation; return current wealth."""
        self.n_steps += 1
        u = self._as_vec(u, self.dim_u)
        v = self._as_vec(v, self.dim_v)

        if not self._initialised:
            self._warm_u.append(u)
            self._warm_v.append(v)
            if len(self._warm_u) >= self.warmup:
                self._initialise()
            return self.wealth

        # need pairs for the swap construction
        if self._pending is None:
            self._pending = (u, v)
            return self.wealth

        (ua, va) = self._pending
        (ub, vb) = (u, v)
        self._pending = None

        pua, pva = self._phi_u(ua), self._phi_v(va)
        pub, pvb = self._phi_u(ub), self._phi_v(vb)

        # joint points P = {(ua,va),(ub,vb)}, product points Q = {(ua,vb),(ub,va)}
        scale = self._witness_scale()
        r_pa = self._witness_raw(pua, pva)
        r_pb = self._witness_raw(pub, pvb)
        r_qa = self._witness_raw(pua, pvb)
        r_qb = self._witness_raw(pub, pva)
        f_pa = np.clip(r_pa / scale, -1.0, 1.0)
        f_pb = np.clip(r_pb / scale, -1.0, 1.0)
        f_qa = np.clip(r_qa / scale, -1.0, 1.0)
        f_qb = np.clip(r_qb / scale, -1.0, 1.0)
        g = 0.5 * (0.5 * (f_pa + f_pb) - 0.5 * (f_qa + f_qb))
        g = float(np.clip(g, -1.0, 1.0))

        # update witness-scale accumulator with raw values (after use -> previsible)
        self._f_sq_sum += r_pa ** 2 + r_pb ** 2 + r_qa ** 2 + r_qb ** 2
        self._f_count += 4

        prev_log = self.log_wealth
        if self.bet == "mixture":
            self._log_w_grid += np.log1p(self.LAMBDA_GRID * g)
            # mixture wealth = mean over grid; log via logsumexp - log(K)
            m = self._log_w_grid.max()
            self.log_wealth = m + np.log(np.mean(np.exp(self._log_w_grid - m)))
        else:
            lam = self._bet_lambda()
            self.log_wealth += np.log1p(lam * g)

        self.last_log_increment = self.log_wealth - prev_log
        self.max_log_wealth = max(self.max_log_wealth, self.log_wealth)
        self.n_bets += 1

        # agrapa accumulators (use AFTER betting -> previsible for next step)
        self._g_sum += g
        self._g2_sum += g * g

        # update witness stats with the real joint points (after using them)
        self._ingest_joint(pua, pva)
        self._ingest_joint(pub, pvb)

        if not self.rejected and self.wealth >= 1.0 / self.alpha:
            self.rejected = True
            self.reject_time = self.n_steps

        return self.wealth


class PageCUSUM:
    """Page-CUSUM on log e-increments: G_t = max(0, G_{t-1} + log e_t - c)."""

    def __init__(self, c: float = 0.0, h: float = 10.0):
        self.c = c
        self.h = h
        self.G = 0.0
        self.alarm = False
        self.alarm_time = None

    def update(self, log_e: float, t: int | None = None) -> bool:
        self.G = max(0.0, self.G + log_e - self.c)
        if self.G > self.h:
            self.alarm = True
            self.alarm_time = t
            return True
        return False

    def reset(self) -> None:
        self.G = 0.0
        self.alarm = False
        self.alarm_time = None
