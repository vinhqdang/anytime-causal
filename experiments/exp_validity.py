"""Validity audit (algorithm.md sec 7.1 + 5.4) -- the falsification experiment.

Generate p mutually independent series (H0: no edges) and, over many seeds,
measure two things as a function of stream length t:

  (A) PER-EDGE type-I error -- the pure anytime claim. A single SKIT wealth
      obeys P(ever cross 1/alpha) <= alpha by Ville's inequality, for ALL t,
      with no correction for repeated looks. The naive HSIC p-value test, looked
      at repeatedly with no correction, has type-I error that CLIMBS toward 1.

  (B) FAMILY-WISE error across all K = p(p-1)/2 candidate edges. ORACLE's
      e-Bonferroni layer (threshold wealth at K/alpha) keeps FWER <= alpha at
      all t; the naive union of uncorrected per-edge declarations does not.

A method using the original draft's exp(HSIC*lambda) e-value would diverge and
fail (A) immediately. The corrected SKIT wealth passes.
"""

from __future__ import annotations

import numpy as np
from itertools import combinations

from oracle.skit import SKIT
from oracle.data import null_stream
from baselines.naive_stopping import NaiveSequentialTester
from experiments.common import save_json, savefig, newfig


def run_skit(p, n, n_seeds, alpha, noise="laplace"):
    """Return grid, per-edge type-I(t), e-Bonferroni FWER(t), uncorrected FWER(t)."""
    grid = np.arange(0, n + 1, 50)
    K = p * (p - 1) // 2
    thr_edge = 1.0 / alpha
    thr_bonf = K / alpha
    edge_cross = []        # first crossing of 1/alpha per individual pair
    seed_first_edge = []   # min over pairs of 1/alpha crossing (uncorrected FWER)
    seed_first_bonf = []   # min over pairs of K/alpha crossing (e-Bonferroni FWER)
    for seed in range(n_seeds):
        rng = np.random.default_rng(10_000 + seed)
        X = null_stream(p, n, noise, rng)
        skits = {(i, j): SKIT(1, 1, alpha=alpha, n_features=64, warmup=40,
                              seed=seed * 97 + 7 * i + j)
                 for i, j in combinations(range(p), 2)}
        pair_edge = {k: n + 1 for k in skits}
        pair_bonf = {k: n + 1 for k in skits}
        for t in range(n):
            for k, sk in skits.items():
                sk.update(X[t, k[0]], X[t, k[1]])
                w = sk.wealth
                if pair_edge[k] > n and w >= thr_edge:
                    pair_edge[k] = t + 1
                if pair_bonf[k] > n and w >= thr_bonf:
                    pair_bonf[k] = t + 1
        edge_cross.extend(pair_edge.values())
        seed_first_edge.append(min(pair_edge.values()))
        seed_first_bonf.append(min(pair_bonf.values()))
    edge_cross = np.array(edge_cross)
    sfe = np.array(seed_first_edge)
    sfb = np.array(seed_first_bonf)
    return (grid.tolist(),
            [float(np.mean(edge_cross <= t)) for t in grid],
            [float(np.mean(sfb <= t)) for t in grid],
            [float(np.mean(sfe <= t)) for t in grid])


def run_naive(p, n, n_seeds, alpha, noise="laplace",
              stride=60, max_window=250, n_perm=30):
    grid = np.arange(0, n + 1, 50)
    edge_cross = []
    seed_first = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(20_000 + seed)
        X = null_stream(p, n, noise, rng)
        testers = {(i, j): NaiveSequentialTester(alpha, stride, max_window,
                                                 n_perm, seed * 13 + i * 5 + j)
                   for i, j in combinations(range(p), 2)}
        pair_cross = {k: n + 1 for k in testers}
        for t in range(n):
            for k, tt in testers.items():
                tt.update(X[t, k[0]], X[t, k[1]])
                if pair_cross[k] > n and tt.rejected:
                    pair_cross[k] = tt.reject_time
        edge_cross.extend(pair_cross.values())
        seed_first.append(min(pair_cross.values()))
    edge_cross = np.array(edge_cross)
    sf = np.array(seed_first)
    return (grid.tolist(),
            [float(np.mean(edge_cross <= t)) for t in grid],
            [float(np.mean(sf <= t)) for t in grid])


def main(quick=False):
    alpha = 0.05
    p = 4
    if quick:
        sg, s_edge, s_bonf, s_unc = run_skit(p, 800, 20, alpha)
        ng, n_edge, n_fwer = run_naive(p, 800, 15, alpha)
    else:
        sg, s_edge, s_bonf, s_unc = run_skit(p, 2000, 100, alpha)
        ng, n_edge, n_fwer = run_naive(p, 1500, 60, alpha)

    result = {
        "alpha": alpha, "p": p, "K": p * (p - 1) // 2,
        "skit": {"t": sg, "per_edge_typeI": s_edge,
                 "fwer_ebonf": s_bonf, "fwer_uncorrected": s_unc},
        "naive": {"t": ng, "per_edge_typeI": n_edge, "fwer_uncorrected": n_fwer},
        "skit_max_per_edge": max(s_edge),
        "skit_max_fwer_ebonf": max(s_bonf),
        "naive_max_per_edge": max(n_edge),
        "naive_max_fwer": max(n_fwer),
    }
    save_json("validity.json", result)

    fig, (ax1, ax2) = newfig(1, 2, figsize=(12, 4.5))
    ax1.plot(sg, s_edge, "-o", ms=3, color="#1b7837", label="ORACLE SKIT wealth")
    ax1.plot(ng, n_edge, "-s", ms=3, color="#b2182b", label="Naive HSIC p-value")
    ax1.axhline(alpha, ls="--", color="k", lw=1, label=f"alpha = {alpha}")
    ax1.set_xlabel("stream length t"); ax1.set_ylabel("per-edge type-I error")
    ax1.set_title("(A) Per-edge type-I vs repeated looks")
    ax1.legend(fontsize=8); ax1.set_ylim(-0.02, max(0.5, max(n_edge) * 1.1))

    ax2.plot(sg, s_bonf, "-o", ms=3, color="#1b7837",
             label="ORACLE (e-Bonferroni)")
    ax2.plot(sg, s_unc, "-^", ms=3, color="#762a83", alpha=0.7,
             label="ORACLE (uncorrected union)")
    ax2.plot(ng, n_fwer, "-s", ms=3, color="#b2182b", label="Naive (uncorrected)")
    ax2.axhline(alpha, ls="--", color="k", lw=1, label=f"alpha = {alpha}")
    ax2.set_xlabel("stream length t"); ax2.set_ylabel("FWER (any false edge)")
    ax2.set_title(f"(B) Family-wise error, K={p*(p-1)//2} edges")
    ax2.legend(fontsize=8); ax2.set_ylim(-0.02, max(0.5, max(n_fwer) * 1.1))
    savefig(fig, "validity_fwer.png")

    print("validity:", {k: round(result[k], 3) for k in
                         ("skit_max_per_edge", "skit_max_fwer_ebonf",
                          "naive_max_per_edge", "naive_max_fwer")})
    return result


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
