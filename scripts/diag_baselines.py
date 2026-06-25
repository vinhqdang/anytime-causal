import numpy as np
from oracle.data import random_dag, sample_anm, normalize
from oracle import metrics
from baselines import batch
from baselines.naive_stopping import hsic_pvalue

rng = np.random.default_rng(0)
# HSIC sanity
u = rng.standard_normal(300); v = rng.standard_normal(300)
print("HSIC p (indep):", round(hsic_pvalue(u, v, 200, rng), 3))
w = np.sin(2 * u) + 0.3 * rng.standard_normal(300)
print("HSIC p (dep):  ", round(hsic_pvalue(u, w, 200, rng), 3))

print("causal-learn available:", batch.available())
A = random_dag(6, 1.5, 'er', rng)
X, _ = sample_anm(A, 1500, rng=rng)
Xn = normalize(X)
G = batch.run_pc(Xn, alpha=0.05, indep_test="fisherz")
m = metrics.all_metrics(A, G)
print("PC-fisherz SHD", m["shd"], "skelF1", round(m["skeleton_f1"], 2),
      "orientF1", round(m["orient_f1"], 2))
