"""Anytime behaviour (algorithm.md sec 7.3 + 5.4): detection efficiency and the
false-edge contrast between ORACLE and naive optional stopping on a real DAG.

On a planted ANM DAG we measure, over the stream:
  - ORACLE: when each TRUE edge first enters the estimated skeleton
    (sample-to-detection) and how many FALSE edges are ever declared.
  - Naive optional stopping: the union of edges it would declare with repeated
    uncorrected looks -- expected to accrue false edges as the stream grows.
"""

from __future__ import annotations

import numpy as np
from itertools import combinations

from oracle.discovery import ORACLE
from oracle.data import random_dag, sample_anm, normalize
from baselines.naive_stopping import NaiveSequentialTester
from experiments.common import save_json, savefig, newfig


def main(quick=False):
    p = 6
    n = 1500 if quick else 4000
    n_graphs = 2 if quick else 5
    alpha = 0.05

    detect_times = []        # normalised sample-to-detection over true edges
    oracle_false = []        # false skeleton edges at end
    naive_false_curve = None
    grid = np.arange(0, n + 1, 100)
    naive_false_acc = np.zeros(len(grid))
    oracle_false_acc = np.zeros(len(grid))

    for g in range(n_graphs):
        rng = np.random.default_rng(500 + g)
        A = random_dag(p, 1.5, "er", rng)
        skel_true = {(min(i, j), max(i, j)) for i in range(p) for j in range(p) if A[i, j]}
        X, _ = sample_anm(A, n, noise="laplace", nonlinear=True, rng=rng)
        Xn = normalize(X)

        orc = ORACLE(p, alpha=alpha, k=2, n_features=64, warmup=40,
                     recond_every=250, cusum_h=1e9, seed=500 + g)
        naive = {(i, j): NaiveSequentialTester(alpha, stride=60, max_window=300,
                                               n_perm=30, seed=g * 31 + i * 7 + j)
                 for i, j in combinations(range(p), 2)}

        first_seen = {e: None for e in skel_true}
        for t in range(n):
            orc.step(Xn[t])
            sk = orc.skeleton()
            for e in skel_true:
                if first_seen[e] is None and sk[e[0], e[1]]:
                    first_seen[e] = t + 1
            for k, tt in naive.items():
                tt.update(Xn[t, k[0]], Xn[t, k[1]])
            gi = np.searchsorted(grid, t + 1, side="right") - 1
            if 0 <= gi < len(grid):
                # current false edges
                of = sum(1 for i in range(p) for j in range(i + 1, p)
                         if sk[i, j] and (i, j) not in skel_true)
                nf = sum(1 for k, tt in naive.items()
                         if tt.rejected and k not in skel_true)
                oracle_false_acc[gi] += of
                naive_false_acc[gi] += nf

        for e, ts in first_seen.items():
            if ts is not None:
                detect_times.append(ts / n)
        sk = orc.skeleton()
        oracle_false.append(sum(1 for i in range(p) for j in range(i + 1, p)
                                if sk[i, j] and (i, j) not in skel_true))

    oracle_false_acc /= n_graphs
    naive_false_acc /= n_graphs

    result = {
        "p": p, "n": n, "n_graphs": n_graphs, "alpha": alpha,
        "median_detection_fraction": float(np.median(detect_times)) if detect_times else None,
        "detected_fraction_of_true_edges": len(detect_times),
        "oracle_false_edges_end_mean": float(np.mean(oracle_false)),
        "grid": grid.tolist(),
        "oracle_false_curve": oracle_false_acc.tolist(),
        "naive_false_curve": naive_false_acc.tolist(),
    }
    save_json("anytime.json", result)

    fig, ax = newfig(figsize=(7, 4.5))
    ax.plot(grid, oracle_false_acc, "-o", ms=3, color="#1b7837",
            label="ORACLE false skeleton edges")
    ax.plot(grid, naive_false_acc, "-s", ms=3, color="#b2182b",
            label="Naive optional-stopping false edges")
    ax.set_xlabel("stream length t")
    ax.set_ylabel("mean # false edges declared")
    ax.set_title(f"False-edge accumulation (p={p}, true edges fixed)")
    ax.legend(); savefig(fig, "anytime_false_edges.png")
    print("anytime:", {k: result[k] for k in
                       ("median_detection_fraction", "oracle_false_edges_end_mean")})
    return result


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
