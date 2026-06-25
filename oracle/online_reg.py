"""Online RFF ridge regression for previsible residualisation.

Used by SKIT_CI (RESIT-style reduction) and by ORACLE's orientation step to
maintain online regressors f_hat_{j|S}: X_S -> X_j. Predictions at time t use
weights fit on data strictly before t (predict-then-update), so residuals stay
previsible -- a requirement the original ORACLE draft silently violated.

Recursive least squares (Sherman-Morrison) gives O(D^2) updates with O(D^2)
memory, no growing design matrix.
"""

from __future__ import annotations

import numpy as np


def _median_heuristic(samples: np.ndarray) -> float:
    n = len(samples)
    if n < 2:
        return 1.0
    if n > 200:
        idx = np.linspace(0, n - 1, 200).astype(int)
        samples = samples[idx]
    diffs = samples[:, None, :] - samples[None, :, :]
    sq = np.sum(diffs ** 2, axis=-1)
    iu = np.triu_indices(len(samples), k=1)
    med = np.median(sq[iu])
    if med <= 0 or not np.isfinite(med):
        med = 1.0
    return float(np.sqrt(med / 2.0))


class OnlineRFFRidge:
    """Streaming kernel-ridge regressor via Random Fourier Features + RLS.

    Parameters
    ----------
    dim : int
        Input dimension (size of conditioning/parent set).
    n_features : int
        Number of random features.
    ridge : float
        Ridge penalty (initial diagonal of the inverse-covariance).
    warmup : int
        Observations buffered to set the RBF bandwidth before predicting.
    seed : int
        RNG seed for the feature draw.
    """

    def __init__(self, dim: int, n_features: int = 128, ridge: float = 1.0,
                 warmup: int = 30, seed: int = 0):
        self.dim = dim
        self.D = n_features
        self.ridge = ridge
        self.warmup = warmup
        self.rng = np.random.default_rng(seed)
        self._reset()

    def _reset(self) -> None:
        self._buf_x: list = []
        self._buf_y: list = []
        self._ready = False
        self.W = None          # feature matrix (dim, D)
        self.b = None
        self.scale = np.sqrt(2.0 / self.D)
        self.theta = np.zeros(self.D)   # weights
        self.P = None          # inverse covariance (D, D)
        self.n = 0
        self.y_mean = 0.0      # running target mean (for intercept)

    def reset(self) -> None:
        self._reset()

    def _features(self, x: np.ndarray) -> np.ndarray:
        return self.scale * np.cos(x @ self.W + self.b)

    def _initialise(self) -> None:
        X = np.array(self._buf_x)
        y = np.array(self._buf_y)
        sigma = max(_median_heuristic(X), 1e-6)
        self.W = self.rng.standard_normal((self.dim, self.D)) / sigma
        self.b = self.rng.uniform(0, 2 * np.pi, self.D)
        self.P = np.eye(self.D) / self.ridge
        self.theta = np.zeros(self.D)
        self.y_mean = float(np.mean(y))
        # batch-seed RLS with warmup data
        for xi, yi in zip(X, y):
            self._rls_step(self._features(xi), yi - self.y_mean)
        self.n = len(X)
        self._ready = True

    def _rls_step(self, phi: np.ndarray, y_centered: float) -> None:
        Pphi = self.P @ phi
        denom = 1.0 + phi @ Pphi
        K = Pphi / denom
        err = y_centered - phi @ self.theta
        self.theta = self.theta + K * err
        self.P = self.P - np.outer(K, Pphi)

    def predict(self, x: np.ndarray) -> float:
        if not self._ready:
            return self.y_mean if self.n else 0.0
        phi = self._features(np.atleast_1d(x))
        return float(phi @ self.theta + self.y_mean)

    def update(self, x, y) -> float:
        """Predict-then-update. Returns the previsible residual y - f_hat(x)."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = float(y)
        if not self._ready:
            self._buf_x.append(x)
            self._buf_y.append(y)
            if len(self._buf_x) >= self.warmup:
                self._initialise()
            return y - (self.y_mean if self.n else 0.0)

        pred = self.predict(x)
        residual = y - pred
        # update running target mean and RLS (after predicting -> previsible)
        self.n += 1
        self.y_mean += (y - self.y_mean) / self.n
        self._rls_step(self._features(x), y - self.y_mean)
        return residual
