"""Batch baselines run via the checkpoint harness (algorithm.md sec 5.0).

PC-stable and GES from causal-learn. Batch methods cannot emit a graph at every
t, so they are wrapped: at a checkpoint, run on the expanding or sliding window
and convert the CPDAG to a directed-adjacency estimate for metric comparison.
"""

from __future__ import annotations

import numpy as np

try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges
    _HAS_CL = True
except Exception:  # pragma: no cover
    _HAS_CL = False


def _cpdag_to_adj(G) -> np.ndarray:
    """Convert a causal-learn graph matrix to a directed adjacency.

    causal-learn encoding: graph[i,j] = -1 (tail) & graph[j,i] = 1 (arrow)
    means i --> j. graph[i,j] = graph[j,i] = -1 means i --- j (undirected);
    undirected edges are dropped from the directed estimate.
    """
    M = G.graph
    p = M.shape[0]
    A = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if M[i, j] == -1 and M[j, i] == 1:
                A[i, j] = 1
    return A


def run_pc(X: np.ndarray, alpha: float = 0.05, indep_test: str = "fisherz") -> np.ndarray:
    if not _HAS_CL:
        raise RuntimeError("causal-learn not installed")
    cg = pc(X, alpha=alpha, indep_test=indep_test, show_progress=False)
    return _cpdag_to_adj(cg.G)


def run_ges(X: np.ndarray, score_func: str = "local_score_BIC") -> np.ndarray:
    if not _HAS_CL:
        raise RuntimeError("causal-learn not installed")
    rec = ges(X, score_func=score_func)
    return _cpdag_to_adj(rec["G"])


def available() -> bool:
    return _HAS_CL
