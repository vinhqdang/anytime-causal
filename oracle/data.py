"""Synthetic data generators for ORACLE evaluation.

Provides:
- Random DAGs (Erdos-Renyi and scale-free).
- Additive non-Gaussian noise models (ANM) with nonlinear mechanisms.
- Piecewise-stationary streams with planted change-points.
- Null streams (mutually independent series) for the validity audit.
- Varsortability measurement and normalisation (Reisach et al. 2021).
"""

from __future__ import annotations

import numpy as np
import networkx as nx


# --------------------------------------------------------------------------- #
# Random DAGs
# --------------------------------------------------------------------------- #
def random_dag(p: int, density: float = 2.0, kind: str = "er",
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Random DAG adjacency (A[i,j]=1 => i->j) in a fixed topological order.

    density : expected number of edges per node (controls edge probability).
    kind : "er" (Erdos-Renyi) or "sf" (scale-free / preferential attachment).
    """
    rng = rng or np.random.default_rng()
    A = np.zeros((p, p), dtype=int)
    if kind == "sf":
        m = max(1, int(round(density)))
        g = nx.barabasi_albert_graph(p, min(m, p - 1), seed=int(rng.integers(1 << 30)))
        for u, v in g.edges():
            a, b = (u, v) if u < v else (v, u)
            A[a, b] = 1
    else:
        prob = min(1.0, density / max(1, (p - 1)))
        for i in range(p):
            for j in range(i + 1, p):
                if rng.random() < prob:
                    A[i, j] = 1
    # random relabel to avoid trivial variance-ordered topology
    perm = rng.permutation(p)
    return A[np.ix_(perm, perm)]


# --------------------------------------------------------------------------- #
# Mechanisms & noise
# --------------------------------------------------------------------------- #
def _noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "laplace":
        return rng.laplace(0, 1, n)
    if kind == "t3":
        return rng.standard_t(3, n)
    if kind == "gumbel":
        return rng.gumbel(0, 1, n)
    if kind == "uniform":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), n)
    return rng.standard_normal(n)


def _make_mechanism(rng: np.random.Generator):
    """Return a random smooth nonlinear scalar mechanism g(z)."""
    choice = rng.integers(0, 4)
    a = rng.uniform(0.5, 2.0)
    phase = rng.uniform(0, 2 * np.pi)
    if choice == 0:
        return lambda z: np.sin(a * z + phase)
    if choice == 1:
        return lambda z: a * np.tanh(z)
    if choice == 2:
        return lambda z: a * (z ** 2) / (1 + np.abs(z))  # bounded-ish quadratic
    return lambda z: a * z + 0.5 * np.sin(2 * z)


def sample_anm(A: np.ndarray, n: int, noise: str = "laplace",
               nonlinear: bool = True, rng: np.random.Generator | None = None,
               weights: dict | None = None, mechanisms: dict | None = None,
               noise_scale: float = 0.5):
    """Sample n observations from an additive-noise SCM with structure A.

    Returns (X, params) where params = {"weights", "mechanisms"} so a caller can
    reuse identical mechanisms across regimes (piecewise stationarity).
    """
    rng = rng or np.random.default_rng()
    p = A.shape[0]
    order = list(nx.topological_sort(_to_digraph(A)))
    X = np.zeros((n, p))
    weights = {} if weights is None else weights
    mechanisms = {} if mechanisms is None else mechanisms

    for j in order:
        parents = np.where(A[:, j] == 1)[0]
        contrib = np.zeros(n)
        for i in parents:
            key = (i, j)
            if key not in weights:
                weights[key] = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
            if nonlinear:
                if key not in mechanisms:
                    mechanisms[key] = _make_mechanism(rng)
                contrib += weights[key] * mechanisms[key](X[:, i])
            else:
                contrib += weights[key] * X[:, i]
        eps = noise_scale * _noise(noise, n, rng)
        X[:, j] = contrib + eps
    return X, {"weights": weights, "mechanisms": mechanisms}


def _to_digraph(A: np.ndarray) -> nx.DiGraph:
    p = A.shape[0]
    g = nx.DiGraph()
    g.add_nodes_from(range(p))
    for i in range(p):
        for j in range(p):
            if A[i, j]:
                g.add_edge(i, j)
    return g


# --------------------------------------------------------------------------- #
# Piecewise-stationary streams
# --------------------------------------------------------------------------- #
def piecewise_stream(p: int, segment_len: int, n_changes: int = 2,
                     density: float = 1.5, noise: str = "laplace",
                     nonlinear: bool = True, rng: np.random.Generator | None = None):
    """Stream whose causal mechanism changes at planted change-points.

    Returns (X, graphs, change_points) where ``graphs`` is the list of DAGs per
    segment and ``change_points`` are the indices where the regime switches.
    """
    rng = rng or np.random.default_rng()
    n_segments = n_changes + 1
    segments = []
    graphs = []
    base_A = random_dag(p, density, "er", rng)
    for seg in range(n_segments):
        if seg == 0:
            A = base_A
        else:
            # perturb: flip a few edges to change structure/mechanism
            A = base_A.copy()
            # add or remove an edge among valid (acyclic) positions
            _perturb_dag(A, rng, k=2)
        graphs.append(A)
        X, _ = sample_anm(A, segment_len, noise, nonlinear, rng)
        segments.append(X)
    X = np.vstack(segments)
    change_points = [segment_len * (s + 1) for s in range(n_changes)]
    return X, graphs, change_points


def _perturb_dag(A: np.ndarray, rng: np.random.Generator, k: int = 2) -> None:
    p = A.shape[0]
    for _ in range(k):
        i, j = rng.integers(0, p, 2)
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)  # keep acyclic w.r.t. index order
        A[a, b] = 1 - A[a, b]


# --------------------------------------------------------------------------- #
# Null streams (validity audit)
# --------------------------------------------------------------------------- #
def null_stream(p: int, n: int, noise: str = "laplace",
                rng: np.random.Generator | None = None) -> np.ndarray:
    """p mutually independent series (no edges) -- the H0 audit data."""
    rng = rng or np.random.default_rng()
    return np.column_stack([_noise(noise, n, rng) for _ in range(p)])


# --------------------------------------------------------------------------- #
# Varsortability (Reisach et al. 2021)
# --------------------------------------------------------------------------- #
def varsortability(X: np.ndarray, A: np.ndarray) -> float:
    """Fraction of directed paths whose variance is correctly increasing.

    ~0.5 means marginal variance leaks no topological information; values near
    1.0 mean a trivial variance-sorting heuristic could recover the order.
    """
    var = np.var(X, axis=0)
    p = A.shape[0]
    reach = np.linalg.matrix_power(A + np.eye(p, dtype=int), p) > 0
    correct = total = 0
    for i in range(p):
        for j in range(p):
            if i != j and reach[i, j] and A[i, j] == 1:
                total += 1
                if var[j] > var[i]:
                    correct += 1
    if total == 0:
        return 0.5
    return correct / total


def normalize(X: np.ndarray) -> np.ndarray:
    """Standardise each column to zero mean, unit variance (kills varsortability)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd
