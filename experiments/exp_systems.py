"""Systems (algorithm.md sec 7.5): throughput and memory vs p.

Verifies the real-time claim empirically by measuring observations/second and
peak memory of the ORACLE update as the number of variables grows.
"""

from __future__ import annotations

import time
import tracemalloc
import numpy as np

from oracle.discovery import ORACLE
from oracle.data import random_dag, sample_anm, normalize
from experiments.common import save_json, savefig, newfig


def main(quick=False):
    ps = [5, 10, 15] if quick else [5, 10, 20, 30, 50]
    n = 400 if quick else 800
    rows = []
    for p in ps:
        rng = np.random.default_rng(900 + p)
        A = random_dag(p, 1.5, "er", rng)
        X, _ = sample_anm(A, n, noise="laplace", nonlinear=True, rng=rng)
        Xn = normalize(X)
        D = 32 if p >= 30 else 64
        orc = ORACLE(p, alpha=0.05, k=2, n_features=D, warmup=40,
                     recond_every=250, cusum_h=1e9, seed=p)
        tracemalloc.start()
        t0 = time.time()
        for t in range(n):
            orc.step(Xn[t])
        dt = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append({"p": p, "n_pairs": p * (p - 1) // 2, "n_features": D,
                     "obs_per_sec": n / dt, "peak_mem_mb": peak / 1e6})
        print(f"[systems p={p}] {n/dt:.0f} obs/s, {peak/1e6:.1f} MB peak, "
              f"{p*(p-1)//2} pairs")

    save_json("systems.json", {"n": n, "rows": rows})

    fig, (ax1, ax2) = newfig(1, 2, figsize=(11, 4.2))
    xs = [r["p"] for r in rows]
    ax1.plot(xs, [r["obs_per_sec"] for r in rows], "-o", color="#2166ac")
    ax1.set_xlabel("variables p"); ax1.set_ylabel("throughput (obs/sec)")
    ax1.set_yscale("log"); ax1.set_title("Throughput vs p")
    ax2.plot(xs, [r["peak_mem_mb"] for r in rows], "-s", color="#b2182b")
    ax2.set_xlabel("variables p"); ax2.set_ylabel("peak memory (MB)")
    ax2.set_title("Memory vs p")
    savefig(fig, "systems_scaling.png")
    return rows


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
