import time
import numpy as np
from oracle.discovery import ORACLE
from oracle.data import random_dag, sample_anm, normalize, varsortability
from oracle import metrics

rng = np.random.default_rng(0)
p = 5
A = random_dag(p, density=1.5, kind="er", rng=rng)
X, _ = sample_anm(A, 3000, noise="laplace", nonlinear=True, rng=rng)
print("true edges:", [(i, j) for i in range(p) for j in range(p) if A[i, j]])
print("varsortability raw:", round(varsortability(X, A), 3))
Xn = normalize(X)
print("varsortability norm:", round(varsortability(Xn, A), 3))

orc = ORACLE(p, alpha=0.05, k=2, n_features=64, warmup=40, recond_every=200, seed=1)
t0 = time.time()
for t in range(len(Xn)):
    orc.step(Xn[t])
dt = time.time() - t0
G = orc.graph()
print(f"\nthroughput: {len(Xn)/dt:.0f} obs/sec ({dt:.1f}s)")
print("est edges:", [(i, j) for i in range(p) for j in range(p) if G[i, j]])
print("skeleton edges:", [(i, j) for i in range(p) for j in range(i+1, p) if orc.skeleton()[i, j]])
print("metrics:", {k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in metrics.all_metrics(A, G).items()})
