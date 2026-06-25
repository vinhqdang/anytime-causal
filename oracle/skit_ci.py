"""SKIT_CI: sequential conditional-independence test by betting.

Construction (a) from the spec -- RESIT residual reduction (default, scalable):
maintain online regressors f_hat_{U|S} and f_hat_{V|S}, residualise

    u_tilde = u - f_hat_{U|S}(s),   v_tilde = v - f_hat_{V|S}(s)

and run an ordinary SKIT independence test on the residuals. Valid under the
additive-noise structure (A3). Regressors are previsible (predict-then-update),
so the residual stream fed to SKIT is itself previsible.

If S is empty this reduces exactly to an unconditional SKIT(U, V).
"""

from __future__ import annotations

import numpy as np

from oracle.online_reg import OnlineRFFRidge
from oracle.skit import SKIT


class SKIT_CI:
    """Test H0: U _||_ V | S from a stream of (u, v, s) triples."""

    def __init__(self, dim_s: int, alpha: float = 0.05, n_features: int = 64,
                 reg_features: int = 128, warmup: int = 40, seed: int = 0):
        self.dim_s = dim_s
        self.alpha = alpha
        self._unconditional = dim_s == 0
        if not self._unconditional:
            self.reg_u = OnlineRFFRidge(dim_s, reg_features, warmup=warmup, seed=seed + 1)
            self.reg_v = OnlineRFFRidge(dim_s, reg_features, warmup=warmup, seed=seed + 2)
        self.skit = SKIT(dim_u=1, dim_v=1, alpha=alpha, n_features=n_features,
                         warmup=warmup, seed=seed + 3)

    @property
    def wealth(self) -> float:
        return self.skit.wealth

    @property
    def log_wealth(self) -> float:
        return self.skit.log_wealth

    @property
    def rejected(self) -> bool:
        return self.skit.rejected

    @property
    def reject_time(self):
        return self.skit.reject_time

    @property
    def last_log_increment(self) -> float:
        return self.skit.last_log_increment

    def update(self, u: float, v: float, s=None) -> float:
        if self._unconditional:
            return self.skit.update(u, v)
        s = np.atleast_1d(np.asarray(s, dtype=float))
        u_res = self.reg_u.update(s, u)
        v_res = self.reg_v.update(s, v)
        return self.skit.update(u_res, v_res)

    def reset(self) -> None:
        if not self._unconditional:
            self.reg_u.reset()
            self.reg_v.reset()
        self.skit.reset()
