"""Graph recovery on ground-truthed synthetic ANM DAGs (algorithm.md sec 7.2).

Streams non-Gaussian additive-noise data from random DAGs (normalised to kill
varsortability) and compares ORACLE's online estimate against batch baselines
run in the checkpoint harness. Reports SHD, SID, and skeleton/orientation F1.
"""

from __future__ import annotations

import time
import numpy as np

from oracle.discovery import ORACLE
from oracle.data import random_dag, sample_anm, normalize, varsortability
from oracle import metrics
from baselines import batch
from experiments.common import save_json, savefig, newfig


def _avg(dicts, key):
    return float(np.mean([d[key] for d in dicts]))


def run_setting(p, density, n, n_graphs, noise, kind, seed0, kci_window=400):
    oracle_m, pc_m, pck_m, ges_m = [], [], [], []
    vs_raw, vs_norm = [], []
    checkpoints = list(range(1000, n + 1, 1000))
    oracle_curves = {c: [] for c in checkpoints}
    throughput = []

    for g in range(n_graphs):
        rng = np.random.default_rng(seed0 + g)
        A = random_dag(p, density, kind, rng)
        X, _ = sample_anm(A, n, noise=noise, nonlinear=True, rng=rng)
        vs_raw.append(varsortability(X, A))
        Xn = normalize(X)
        vs_norm.append(varsortability(Xn, A))

        # ---- ORACLE (online) ----
        orc = ORACLE(p, alpha=0.05, k=2, n_features=64, warmup=40,
                     recond_every=250, cusum_h=1e9, seed=seed0 + g)
        t0 = time.time()
        for t in range(n):
            orc.step(Xn[t])
            if (t + 1) in oracle_curves:
                oracle_curves[t + 1].append(metrics.all_metrics(A, orc.graph()))
        throughput.append(n / (time.time() - t0))
        oracle_m.append(metrics.all_metrics(A, orc.graph()))

        # ---- batch baselines at final t ----
        if batch.available():
            try:
                pc_m.append(metrics.all_metrics(A, batch.run_pc(Xn, 0.05, "fisherz")))
            except Exception:
                pass
            try:
                win = Xn[-kci_window:]
                pck_m.append(metrics.all_metrics(A, batch.run_pc(win, 0.05, "kci")))
            except Exception:
                pass
            try:
                ges_m.append(metrics.all_metrics(A, batch.run_ges(Xn)))
            except Exception:
                pass

    def summarise(ms):
        if not ms:
            return None
        return {k: _avg(ms, k) for k in
                ("shd", "sid", "skeleton_f1", "orient_f1",
                 "skeleton_precision", "skeleton_recall")}

    curve = {str(c): summarise(oracle_curves[c]) for c in checkpoints
             if oracle_curves[c]}
    return {
        "p": p, "density": density, "n": n, "n_graphs": n_graphs,
        "noise": noise, "kind": kind,
        "varsortability_raw": float(np.mean(vs_raw)),
        "varsortability_norm": float(np.mean(vs_norm)),
        "throughput_obs_per_sec": float(np.mean(throughput)),
        "ORACLE": summarise(oracle_m),
        "PC_fisherz": summarise(pc_m),
        "PC_kci_window": summarise(pck_m),
        "GES_BIC": summarise(ges_m),
        "oracle_curve": curve,
    }


def main(quick=False):
    results = []
    if quick:
        settings = [dict(p=8, density=1.5, n=1500, n_graphs=2,
                         noise="laplace", kind="er", seed0=100)]
    else:
        settings = [
            dict(p=10, density=1.5, n=3000, n_graphs=5, noise="laplace",
                 kind="er", seed0=100),
            dict(p=10, density=2.0, n=2500, n_graphs=4, noise="t3",
                 kind="sf", seed0=200),
            dict(p=15, density=1.5, n=2000, n_graphs=2, noise="laplace",
                 kind="er", seed0=300),
        ]
    for s in settings:
        r = run_setting(**s)
        results.append(r)
        print(f"[recovery p={s['p']} {s['kind']} {s['noise']}] "
              f"ORACLE SHD={r['ORACLE']['shd']:.1f} skelF1={r['ORACLE']['skeleton_f1']:.2f} "
              f"orientF1={r['ORACLE']['orient_f1']:.2f} | "
              f"PCfz SHD={r['PC_fisherz']['shd'] if r['PC_fisherz'] else None} | "
              f"thr={r['throughput_obs_per_sec']:.0f} obs/s "
              f"| vs raw={r['varsortability_raw']:.2f} norm={r['varsortability_norm']:.2f}")

    save_json("recovery.json", results)

    # convergence figure for the first setting
    r0 = results[0]
    curve = r0["oracle_curve"]
    ts = sorted(int(k) for k in curve)
    if ts:
        fig, ax = newfig(figsize=(7, 4.5))
        ax.plot(ts, [curve[str(t)]["shd"] for t in ts], "-o", label="SHD")
        ax.plot(ts, [curve[str(t)]["sid"] for t in ts], "-s", label="SID")
        ax.set_xlabel("stream length t"); ax.set_ylabel("distance to ground truth")
        ax.set_title(f"ORACLE convergence (p={r0['p']}, {r0['kind']}, {r0['noise']})")
        ax.legend(); savefig(fig, "recovery_convergence.png")
    return results


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
