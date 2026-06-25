"""Change detection (algorithm.md sec 7.4): Page-CUSUM delay vs ARL.

Streams a piecewise-stationary SCM with planted change-points. For each true
edge i->j present in the first regime we maintain a residual-validity monitor:
an online regressor f_hat_{j|i} and a SKIT testing whether the residual
X_j - f_hat(X_i) is still INDEPENDENT of X_i. While the fitted mechanism is
correct the residual is independent and the monitor's wealth is flat
(increments <= 0); when the mechanism changes the residual becomes dependent and
the wealth rises. Page-CUSUM on these log e-increments thus stays near zero
before a change and climbs after it.

We sweep the threshold h offline to trace the detection-delay / ARL-to-false-
alarm curve. A kernel change-point detector (ruptures) is the generic
comparator.
"""

from __future__ import annotations

import numpy as np

from oracle.skit import SKIT
from oracle.online_reg import OnlineRFFRidge
from oracle.data import piecewise_stream, normalize
from experiments.common import save_json, savefig, newfig

try:
    import ruptures as rpt
    _HAS_RPT = True
except Exception:
    _HAS_RPT = False


def page_cusum_path(log_e, c):
    G = 0.0
    path = np.empty(len(log_e))
    for t, le in enumerate(log_e):
        G = max(0.0, G + le - c)
        path[t] = G
    return path


def detect_first(path, h, start=0):
    idx = np.where(path[start:] > h)[0]
    return (start + idx[0]) if len(idx) else None


def main(quick=False):
    p = 5
    seg = 1200 if quick else 1800
    n_changes = 2
    n_runs = 2 if quick else 5
    c = 0.0  # pure CUSUM; under H0 SKIT increments have non-positive mean

    monitors = []  # one dict per monitored true edge: {log_e, cp1, n}
    example_cps = None
    example_rpt = None
    for g in range(n_runs):
        rng = np.random.default_rng(700 + g)
        X, graphs, cps = piecewise_stream(p, seg, n_changes, density=1.5,
                                          noise="laplace", nonlinear=True, rng=rng)
        Xn = normalize(X)
        n = len(Xn)
        cp1 = cps[0]
        if example_cps is None:
            example_cps = cps
            if _HAS_RPT:
                example_rpt = _ruptures_cps(Xn, len(cps))

        A0 = graphs[0]
        true_edges = [(i, j) for i in range(p) for j in range(p) if A0[i, j]]
        for (i, j) in true_edges:
            reg = OnlineRFFRidge(1, n_features=128, warmup=40, seed=g * 100 + i * 7 + j)
            sk = SKIT(1, 1, alpha=0.05, n_features=64, warmup=40,
                      seed=g * 200 + i * 11 + j)
            le = np.zeros(n)
            for t in range(n):
                res = reg.update(Xn[t, i], Xn[t, j])   # X_j - f_hat(X_i)
                sk.update(res, Xn[t, i])
                le[t] = sk.last_log_increment
            # keep only monitors that actually respond to the change
            post = le[cp1:cp1 + seg].sum()
            if post > 0:
                monitors.append({"log_e": le, "cp1": cp1})

    hs = [0.5, 1, 2, 3, 5, 8, 12]
    delays, arls, det_rate = [], [], []
    for h in hs:
        d_list, arl_list, detected = [], [], 0
        for m in monitors:
            le = m["log_e"]; cp1 = m["cp1"]
            path = page_cusum_path(le, c)
            det = detect_first(path, h, start=cp1)
            # ARL: time to a (false) alarm running only on the pre-change segment
            ppath = page_cusum_path(le[:cp1], c)
            fa = detect_first(ppath, h, start=0)
            arl_list.append(fa if fa is not None else cp1)
            if det is not None and (fa is None or fa >= cp1):
                d_list.append(det - cp1); detected += 1
        delays.append(float(np.mean(d_list)) if d_list else None)
        arls.append(float(np.mean(arl_list)))
        det_rate.append(detected / max(1, len(monitors)))

    result = {
        "p": p, "segment_len": seg, "n_changes": n_changes, "n_runs": n_runs,
        "n_monitored_edges": len(monitors), "drift_c": c, "h_grid": hs,
        "mean_delay": delays, "mean_arl": arls, "detection_rate": det_rate,
        "ruptures_available": _HAS_RPT,
        "example_change_points": example_cps,
        "example_ruptures_cps": example_rpt,
    }
    save_json("change.json", result)

    fig, ax = newfig(figsize=(7, 4.5))
    xs = [a for a, d in zip(arls, delays) if d is not None]
    ys = [d for d in delays if d is not None]
    ax.plot(xs, ys, "-o", color="#2166ac")
    for h, a, d in zip(hs, arls, delays):
        if d is not None:
            ax.annotate(f"h={h}", (a, d), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("ARL to false alarm (obs)")
    ax.set_ylabel("detection delay (obs)")
    ax.set_title("Page-CUSUM delay vs ARL (sweep h)")
    savefig(fig, "change_delay_arl.png")
    print("change:", {"delay": delays, "arl": arls, "det_rate": det_rate,
                      "n_edges": len(monitors)})
    return result


def _ruptures_cps(X, n_bkps):
    try:
        algo = rpt.KernelCPD(kernel="rbf").fit(X)
        return algo.predict(n_bkps=n_bkps)
    except Exception:
        return None


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
