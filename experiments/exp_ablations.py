"""Ablations (algorithm.md sec 7.6).

Sweeps alpha, the degree cap k (the pairwise-vs-DAG gap), bet sizing, and RFF
feature count on a fixed family of synthetic ANM DAGs; reports SHD and F1.
"""

from __future__ import annotations

import numpy as np

from oracle.discovery import ORACLE
from oracle.data import random_dag, sample_anm, normalize
from oracle import metrics
from experiments.common import save_json


def _eval(p, n, n_graphs, seed0, **orc_kwargs):
    ms = []
    for g in range(n_graphs):
        rng = np.random.default_rng(seed0 + g)
        A = random_dag(p, 1.5, "er", rng)
        X, _ = sample_anm(A, n, noise="laplace", nonlinear=True, rng=rng)
        Xn = normalize(X)
        orc = ORACLE(p, warmup=40, recond_every=250, cusum_h=1e9,
                     seed=seed0 + g, **orc_kwargs)
        for t in range(n):
            orc.step(Xn[t])
        ms.append(metrics.all_metrics(A, orc.graph()))
    return {k: float(np.mean([m[k] for m in ms]))
            for k in ("shd", "sid", "skeleton_f1", "orient_f1")}


def main(quick=False):
    p = 6
    n = 1500 if quick else 2500
    ng = 2 if quick else 4
    out = {"p": p, "n": n, "n_graphs": ng}

    # alpha sweep
    out["alpha"] = {str(a): _eval(p, n, ng, 1000, alpha=a, k=2, n_features=64)
                    for a in ([0.05] if quick else [0.01, 0.05, 0.10])}
    # degree cap k (k=0 => pairwise dependency graph, not a DAG)
    out["k"] = {str(k): _eval(p, n, ng, 1100, alpha=0.05, k=k, n_features=64)
                for k in ([0, 2] if quick else [0, 1, 2, 3])}
    # bet sizing: fixed lambda vs aGRAPA vs mixture
    out["bet"] = {b: _eval(p, n, ng, 1300, alpha=0.05, k=2, n_features=64, bet=b)
                  for b in (["mixture"] if quick else ["fixed", "agrapa", "mixture"])}
    # RFF features
    out["n_features"] = {str(D): _eval(p, n, ng, 1200, alpha=0.05, k=2, n_features=D)
                         for D in ([64] if quick else [32, 64, 128])}

    save_json("ablations.json", out)
    print("ablations k-sweep (pairwise vs DAG):",
          {k: round(v["shd"], 2) for k, v in out["k"].items()})
    return out


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
