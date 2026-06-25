import numpy as np
from oracle.skit import SKIT
from oracle.online_reg import OnlineRFFRidge

for mech_name, mech in [("sin1.5", lambda x: np.sin(1.5 * x)),
                         ("cubic", lambda x: x + x**3),
                         ("tanh2", lambda x: 2*np.tanh(2*x))]:
    rng = np.random.default_rng(7)
    n = 4000
    x = rng.standard_normal(n)
    y = mech(x) + 0.3 * rng.laplace(0, 1, n)

    reg_fwd = OnlineRFFRidge(1, n_features=128, warmup=50, seed=10)
    reg_rev = OnlineRFFRidge(1, n_features=128, warmup=50, seed=11)
    A = SKIT(1, 1, alpha=0.05, n_features=128, warmup=40, seed=12)
    B = SKIT(1, 1, alpha=0.05, n_features=128, warmup=40, seed=13)
    rf, rr = [], []
    for t in range(n):
        res_fwd = reg_fwd.update(x[t], y[t])
        res_rev = reg_rev.update(y[t], x[t])
        rf.append(res_fwd); rr.append(res_rev)
        A.update(res_fwd, x[t])
        B.update(res_rev, y[t])
    rf = np.array(rf); rr = np.array(rr)
    late = slice(n//2, n)
    print(f"[{mech_name}] A.logW={A.max_log_wealth:.2f} rej={A.rejected} | "
          f"B.logW={B.max_log_wealth:.2f} rej={B.rejected}")
    print(f"   corr(res_fwd,x)={abs(np.corrcoef(rf[late],x[late])[0,1]):.3f} "
          f"corr(res_rev,y)={abs(np.corrcoef(rr[late],y[late])[0,1]):.3f}")
