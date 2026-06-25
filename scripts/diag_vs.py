import numpy as np
from oracle.data import random_dag, sample_anm, normalize, varsortability
rng = np.random.default_rng(0)
A = random_dag(5, 1.5, 'er', rng)
X, _ = sample_anm(A, 3000, rng=rng)
Xn = normalize(X)
print('raw var', np.var(X, axis=0).round(2))
print('norm var', np.var(Xn, axis=0).round(6))
print('vs raw', varsortability(X, A), 'vs norm', varsortability(Xn, A))
# average over many graphs
vs_raw, vs_norm = [], []
for s in range(30):
    r = np.random.default_rng(s)
    a = random_dag(10, 2.0, 'er', r)
    x, _ = sample_anm(a, 2000, rng=r)
    vs_raw.append(varsortability(x, a))
    vs_norm.append(varsortability(normalize(x), a))
print('mean vs raw', round(np.mean(vs_raw), 3), 'mean vs norm', round(np.mean(vs_norm), 3))
