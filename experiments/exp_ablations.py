"""Ablations (algorithm.md sec 7.6), expanded.

ORACLE-level sweeps on a fixed family of synthetic ANM DAGs (report SHD, SID,
skeleton/orientation F1):
  - alpha level
  - degree cap k  (k=0 is a pairwise dependency graph, not a DAG)
  - bet sizing    (fixed / aGRAPA / mixture)
  - RFF feature count D
  - multiplicity  (e-BH / e-Bonferroni / none)
  - noise family  (Laplace / Student-t3 / Gumbel / Gaussian; Gaussian is the
                   non-identifiable control)
  - warm-up window

Plus a SKIT-level micro-study isolating the witness-standardisation power fix
(rejection rate and median sample-to-detection, standardisation on vs off).
"""

from __future__ import annotations

import numpy as np

from oracle.discovery import ORACLE
from oracle.skit import SKIT
from oracle.data import random_dag, sample_anm, normalize
from oracle import metrics
from experiments.common import save_json


def _eval(p, n, n_graphs, seed0, noise="laplace", density=1.5, **orc_kwargs):
    ms = []
    for g in range(n_graphs):
        rng = np.random.default_rng(seed0 + g)
        A = random_dag(p, density, "er", rng)
        X, _ = sample_anm(A, n, noise=noise, nonlinear=True, rng=rng)
        Xn = normalize(X)
        orc = ORACLE(p, warmup=orc_kwargs.pop("warmup", 40),
                     recond_every=orc_kwargs.pop("recond_every", 250),
                     cusum_h=1e9, seed=seed0 + g, **orc_kwargs)
        for t in range(n):
            orc.step(Xn[t])
        ms.append(metrics.all_metrics(A, orc.graph()))
    return {k: float(np.mean([m[k] for m in ms]))
            for k in ("shd", "sid", "skeleton_f1", "orient_f1")}


def _standardization_study(quick):
    """Isolate the power effect of witness standardisation at the SKIT level."""
    n = 1200 if quick else 2000
    n_trials = 15 if quick else 40
    out = {}
    for std in (True, False):
        rej, times = 0, []
        for s in range(n_trials):
            rng = np.random.default_rng(4000 + s)
            x = rng.standard_normal(n)
            y = np.sin(1.5 * x) + 0.3 * rng.laplace(0, 1, n)
            sk = SKIT(1, 1, alpha=0.05, n_features=64, warmup=40,
                      standardize=std, seed=s)
            for t in range(n):
                sk.update(x[t], y[t])
            if sk.rejected:
                rej += 1
                times.append(sk.reject_time)
        out["on" if std else "off"] = {
            "rejection_rate": rej / n_trials,
            "median_detection": float(np.median(times)) if times else None,
        }
    return out


def main(quick=False):
    p = 6
    n = 1500 if quick else 2200
    ng = 2 if quick else 4
    out = {"p": p, "n": n, "n_graphs": ng}

    out["alpha"] = {str(a): _eval(p, n, ng, 1000, alpha=a, k=2, n_features=64)
                    for a in ([0.05] if quick else [0.01, 0.05, 0.10])}
    out["k"] = {str(k): _eval(p, n, ng, 1100, alpha=0.05, k=k, n_features=64)
                for k in ([0, 2] if quick else [0, 1, 2, 3])}
    out["bet"] = {b: _eval(p, n, ng, 1300, alpha=0.05, k=2, n_features=64, bet=b)
                  for b in (["mixture"] if quick else ["fixed", "agrapa", "mixture"])}
    out["n_features"] = {str(D): _eval(p, n, ng, 1200, alpha=0.05, k=2, n_features=D)
                         for D in ([64] if quick else [32, 64, 128])}
    out["multiplicity"] = {
        m: _eval(p, n, ng, 1400, alpha=0.05, k=2, n_features=64, multiplicity=m)
        for m in (["ebh"] if quick else ["ebh", "bonferroni", "none"])}
    out["noise"] = {
        nz: _eval(p, n, ng, 1500, alpha=0.05, k=2, n_features=64, noise=nz)
        for nz in (["laplace"] if quick else ["laplace", "t3", "gumbel", "gaussian"])}
    out["warmup"] = {
        str(w): _eval(p, n, ng, 1600, alpha=0.05, k=2, n_features=64, warmup=w)
        for w in ([40] if quick else [25, 40, 60])}
    out["recond_every"] = {
        str(r): _eval(p, n, ng, 1700, alpha=0.05, k=2, n_features=64, recond_every=r)
        for r in ([250] if quick else [100, 250, 500])}
    out["density"] = {
        str(d): _eval(p, n, ng, 1800, alpha=0.05, k=2, n_features=64, density=d)
        for d in ([1.5] if quick else [1.0, 1.5, 2.5])}
    out["stream_length"] = {
        str(nn): _eval(p, nn, ng, 1900, alpha=0.05, k=2, n_features=64)
        for nn in ([n] if quick else [1000, 2000, 3000])}
    out["standardization"] = _standardization_study(quick)

    save_json("ablations.json", out)
    print("ablations k-sweep (pairwise vs DAG):",
          {k: round(v["shd"], 2) for k, v in out["k"].items()})
    print("ablations standardization:", out["standardization"])
    print("ablations noise SHD:",
          {k: round(v["shd"], 2) for k, v in out["noise"].items()})
    return out


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
