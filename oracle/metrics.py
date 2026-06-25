"""Graph-recovery metrics: SHD, SID, and skeleton/orientation precision-recall.

Graphs are represented as binary adjacency matrices ``A`` with ``A[i, j] == 1``
meaning a directed edge i -> j.
"""

from __future__ import annotations

import numpy as np
import networkx as nx


def _as_array(A) -> np.ndarray:
    return (np.asarray(A) != 0).astype(int)


def shd(true_A, est_A) -> int:
    """Structural Hamming Distance for directed graphs.

    Counts: missing edges + extra edges + reversed edges. A reversal counts as
    a single error (not two).
    """
    G = _as_array(true_A)
    H = _as_array(est_A)
    p = G.shape[0]
    dist = 0
    for i in range(p):
        for j in range(i + 1, p):
            t = (G[i, j], G[j, i])
            e = (H[i, j], H[j, i])
            if t == e:
                continue
            # any disagreement on this pair: reversal is one error,
            # presence/absence mismatch is one error
            dist += 1
    return dist


def skeleton_metrics(true_A, est_A) -> dict:
    """Precision/recall/F1 on the undirected skeleton (adjacency)."""
    G = _as_array(true_A)
    H = _as_array(est_A)
    p = G.shape[0]
    tp = fp = fn = 0
    for i in range(p):
        for j in range(i + 1, p):
            t = G[i, j] or G[j, i]
            e = H[i, j] or H[j, i]
            if t and e:
                tp += 1
            elif e and not t:
                fp += 1
            elif t and not e:
                fn += 1
    return _prf(tp, fp, fn)


def orientation_metrics(true_A, est_A) -> dict:
    """Precision/recall/F1 on directed edges (exact i->j match)."""
    G = _as_array(true_A)
    H = _as_array(est_A)
    p = G.shape[0]
    tp = fp = fn = 0
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if G[i, j] and H[i, j]:
                tp += 1
            elif H[i, j] and not G[i, j]:
                fp += 1
            elif G[i, j] and not H[i, j]:
                fn += 1
    return _prf(tp, fp, fn)


def _prf(tp: int, fp: int, fn: int) -> dict:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


# --------------------------------------------------------------------------- #
# Structural Intervention Distance (Peters & Buehlmann 2015)
# --------------------------------------------------------------------------- #
def sid(true_A, est_A) -> int:
    """Structural Intervention Distance.

    Counts ordered pairs (i, j), i != j, for which the parent-adjustment set of
    the *estimated* DAG is not a valid back-door adjustment set for the causal
    effect of i on j in the *true* DAG. SID(G, G) == 0.

    This implementation follows the parent-adjustment criterion: to estimate
    the interventional distribution p(x_j | do(x_i)) one adjusts for
    Pa_est(i). The pair is correct iff that set satisfies the back-door
    criterion relative to (i, j) in the true graph (and the trivial cases where
    j == i or j is not a descendant of i are handled by the criterion).
    """
    G = _as_array(true_A)
    H = _as_array(est_A)
    p = G.shape[0]
    true_dag = nx.DiGraph()
    true_dag.add_nodes_from(range(p))
    for a in range(p):
        for b in range(p):
            if G[a, b]:
                true_dag.add_edge(a, b)
    if not nx.is_directed_acyclic_graph(true_dag):
        # SID is defined for DAGs; fall back to treating cycles conservatively
        true_dag = _break_cycles(true_dag)

    mistakes = 0
    for i in range(p):
        est_parents = set(np.where(H[:, i] == 1)[0].tolist())
        desc_i = nx.descendants(true_dag, i)
        for j in range(p):
            if i == j:
                continue
            if not _valid_parent_adjustment(true_dag, i, j, est_parents, desc_i):
                mistakes += 1
    return mistakes


def _valid_parent_adjustment(dag, i, j, Z, desc_i) -> bool:
    """Is adjusting for Z = Pa_est(i) a valid identification of p(x_j | do x_i)
    in the true graph ``dag``? (Peters & Buehlmann 2015 parent-adjustment.)"""
    if j in Z:
        # estimated graph adjusts on the outcome -> claims effect = marginal p(x_j);
        # correct iff i has no causal effect on j (j not a descendant of i)
        return j not in desc_i
    if i in Z:
        return False
    # adjustment set may not contain descendants of i
    if Z & desc_i:
        return False
    # Z must block all back-door paths from i to j
    return _backdoor_paths_blocked(dag, i, j, Z)


def _break_cycles(dag: nx.DiGraph) -> nx.DiGraph:
    g = dag.copy()
    while True:
        try:
            cyc = nx.find_cycle(g, orientation="original")
        except nx.NetworkXNoCycle:
            break
        g.remove_edge(cyc[0][0], cyc[0][1])
    return g


def _backdoor_paths_blocked(dag: nx.DiGraph, i: int, j: int, Z: set) -> bool:
    """Check d-separation of i and j given Z in the graph with i's outgoing
    edges removed (back-door graph)."""
    bd = dag.copy()
    for child in list(dag.successors(i)):
        bd.remove_edge(i, child)
    return _d_separated(bd, i, j, Z)


def _d_separated(dag: nx.DiGraph, x: int, y: int, Z: set) -> bool:
    """d-separation of x and y given Z, via networkx."""
    if x == y:
        return False
    Z = {z for z in Z if z != x and z != y}
    return nx.is_d_separator(dag, {x}, {y}, Z)


def all_metrics(true_A, est_A) -> dict:
    out = {"shd": shd(true_A, est_A), "sid": sid(true_A, est_A)}
    sk = skeleton_metrics(true_A, est_A)
    ori = orientation_metrics(true_A, est_A)
    out.update({f"skeleton_{k}": v for k, v in sk.items()})
    out.update({f"orient_{k}": v for k, v in ori.items()})
    return out
