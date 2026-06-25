"""Multiplicity control over per-edge e-values.

e-BH (Wang & Ramdas 2022, JRSS-B) controls FDR for arbitrarily dependent
e-values -- exactly the regime here, where edge wealths share the data stream.
e-Bonferroni is the conservative FWER option.
"""

from __future__ import annotations

import numpy as np


def ebh(evalues, alpha: float, n_hypotheses: int | None = None):
    """e-BH procedure.

    Parameters
    ----------
    evalues : array-like
        Nonnegative e-values, one per candidate hypothesis (edge).
    alpha : float
        Target FDR level.
    n_hypotheses : int, optional
        Total number of hypotheses K. Defaults to ``len(evalues)``. Pass the
        full candidate count when ``evalues`` only lists a subset.

    Returns
    -------
    rejected : list[int]
        Indices (into ``evalues``) of rejected hypotheses.
    """
    e = np.asarray(evalues, dtype=float)
    K = n_hypotheses if n_hypotheses is not None else len(e)
    if len(e) == 0 or K == 0:
        return []
    order = np.argsort(-e)  # descending
    e_sorted = e[order]
    thresh = K / (alpha * np.arange(1, len(e_sorted) + 1))
    passing = np.where(e_sorted >= thresh)[0]
    if len(passing) == 0:
        return []
    k_star = passing.max() + 1  # largest rank with e_(k) >= K/(alpha k)
    return sorted(order[:k_star].tolist())


def e_bonferroni(evalues, alpha: float, n_hypotheses: int | None = None):
    """e-Bonferroni: reject hypothesis i iff e_i >= K/alpha (FWER control)."""
    e = np.asarray(evalues, dtype=float)
    K = n_hypotheses if n_hypotheses is not None else len(e)
    if K == 0:
        return []
    return [i for i, ev in enumerate(e) if ev >= K / alpha]
