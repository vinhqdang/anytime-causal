"""Critical unit tests. Per algorithm.md sec 8, do not proceed until these pass:
  1. SKIT type-I control on a null (independent) stream.
  2. SKIT power on a dependent stream.
  3. The ANM orientation SIGN CONVENTION on a 2-node graph.
  4. Conditioning removes a spurious edge on a 3-node chain.
  5. e-BH / metric sanity.
"""

import numpy as np
import pytest

from oracle.skit import SKIT
from oracle.skit_ci import SKIT_CI
from oracle.online_reg import OnlineRFFRidge
from oracle.ebh import ebh, e_bonferroni
from oracle import metrics


# --------------------------------------------------------------------------- #
def test_skit_type_one_control():
    """Under H0 (independent), false-rejection rate over many seeds <= ~alpha."""
    alpha = 0.05
    n_trials = 200
    n = 600
    rejections = 0
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        u = rng.standard_normal(n)
        v = rng.standard_normal(n)  # independent of u
        skit = SKIT(1, 1, alpha=alpha, warmup=40, seed=seed)
        for t in range(n):
            skit.update(u[t], v[t])
        if skit.rejected:
            rejections += 1
    rate = rejections / n_trials
    # Ville's inequality guarantees <= alpha in expectation; allow MC slack.
    assert rate <= alpha + 0.03, f"type-I rate {rate} exceeds alpha={alpha}"


def test_skit_power():
    """Under dependence, the wealth should reject within the stream."""
    rng = np.random.default_rng(0)
    n = 1500
    u = rng.standard_normal(n)
    v = np.sin(2.0 * u) + 0.3 * rng.standard_normal(n)
    skit = SKIT(1, 1, alpha=0.05, warmup=40, seed=1)
    for t in range(n):
        skit.update(u[t], v[t])
    assert skit.rejected, "SKIT failed to detect a clear dependence"


def test_online_regressor_residual_independent():
    """Residual of a clean ANM regression should lose dependence on the input."""
    rng = np.random.default_rng(3)
    n = 2000
    x = rng.standard_normal(n)
    y = np.sin(2 * x) + 0.3 * rng.laplace(0, 1, n)
    reg = OnlineRFFRidge(1, n_features=128, warmup=50, seed=2)
    res = np.array([reg.update(x[t], y[t]) for t in range(n)])
    # late residuals should correlate with neither x nor sin(2x) strongly
    late = slice(n // 2, n)
    corr = abs(np.corrcoef(res[late], x[late])[0, 1])
    assert corr < 0.2, f"residual still correlated with input (|r|={corr:.3f})"


def test_orientation_sign_convention():
    """THE sign test: 2-node ANM X->Y must be oriented X->Y, not Y->X.

    Forward (X->Y): residual Y - f(X) is independent of X  => A stays low.
    Reverse (Y->X): residual X - g(Y) remains dependent on Y => B grows.
    Decision rule: B rejects and A does not  => orient fwd (i->j) = X->Y.
    """
    rng = np.random.default_rng(7)
    n = 3000
    x = rng.standard_normal(n)               # i = X (cause)
    y = np.sin(1.5 * x) + 0.3 * rng.laplace(0, 1, n)  # j = Y (effect)

    reg_fwd = OnlineRFFRidge(1, n_features=128, warmup=50, seed=10)  # i->j
    reg_rev = OnlineRFFRidge(1, n_features=128, warmup=50, seed=11)  # j->i
    A = SKIT(1, 1, alpha=0.05, n_features=128, warmup=40, seed=12)  # dep fwd resid on X_i
    B = SKIT(1, 1, alpha=0.05, n_features=128, warmup=40, seed=13)  # dep rev resid on X_j
    for t in range(n):
        res_fwd = reg_fwd.update(x[t], y[t])   # Y - f(X)
        res_rev = reg_rev.update(y[t], x[t])   # X - g(Y)
        A.update(res_fwd, x[t])
        B.update(res_rev, y[t])

    assert B.rejected, "reverse residual should stay dependent (B must reject)"
    assert not A.rejected, "forward residual should be independent (A must not reject)"


def test_conditioning_removes_spurious_edge():
    """3-node chain X->Y->Z: X and Z are marginally dependent but X _||_ Z | Y."""
    rng = np.random.default_rng(5)
    n = 3000
    x = rng.standard_normal(n)
    y = np.sin(1.5 * x) + 0.3 * rng.laplace(0, 1, n)
    z = np.tanh(1.5 * y) + 0.3 * rng.laplace(0, 1, n)

    marginal = SKIT(1, 1, alpha=0.05, warmup=40, seed=20)
    conditional = SKIT_CI(1, alpha=0.05, warmup=40, seed=21)
    for t in range(n):
        marginal.update(x[t], z[t])
        conditional.update(x[t], z[t], y[t])

    assert marginal.rejected, "X and Z should be marginally dependent"
    assert not conditional.rejected, "X _||_ Z | Y should NOT reject (sep set works)"


def test_ebh_and_bonferroni():
    e = [100.0, 50.0, 0.5, 0.1, 0.2]
    rej = ebh(e, alpha=0.1, n_hypotheses=5)
    assert set(rej) == {0, 1}, f"unexpected e-BH rejections: {rej}"
    rej_b = e_bonferroni(e, alpha=0.1, n_hypotheses=5)
    assert set(rej_b) == {0, 1}  # threshold 5/0.1 = 50


def test_metrics_identity():
    A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert metrics.shd(A, A) == 0
    assert metrics.sid(A, A) == 0
    m = metrics.all_metrics(A, A)
    assert m["orient_f1"] == 1.0
    assert m["skeleton_f1"] == 1.0


def test_metrics_reversed_edge():
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0, 0], [1, 0]])  # reversed
    assert metrics.shd(A, B) == 1            # one reversal
    assert metrics.skeleton_metrics(A, B)["f1"] == 1.0  # same skeleton
    assert metrics.orientation_metrics(A, B)["f1"] == 0.0
    assert metrics.sid(A, B) > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
