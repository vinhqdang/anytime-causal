"""Generate manuscript figures (PDF + PNG) from saved experiment JSON.

Decoupled from the heavy experiment compute: reads results/*.json and writes
publication-quality figures into manuscript/figures/. Re-runnable.

    python -m experiments.make_figures
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "manuscript", "figures")
os.makedirs(OUT, exist_ok=True)

DPI = 300  # high-resolution PNG output

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "figure.dpi": DPI, "savefig.bbox": "tight",
})

GREEN, RED, BLUE, PURPLE = "#1b7837", "#b2182b", "#2166ac", "#762a83"


def _load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _save(fig, stem):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{stem}.png"), dpi=DPI)
    plt.close(fig)
    print(f"  wrote {stem}.png ({DPI} dpi)")


def fig_validity(v):
    a = v["alpha"]
    s, n = v["skit"], v["naive"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(s["t"], s["per_edge_typeI"], "-o", ms=3, color=GREEN,
             label="ORACLE (SKIT wealth)")
    ax1.plot(n["t"], n["per_edge_typeI"], "-s", ms=3, color=RED,
             label="Naive optional stopping")
    ax1.axhline(a, ls="--", color="k", lw=1, label=fr"$\alpha={a}$")
    ax1.set_xlabel("stream length $t$"); ax1.set_ylabel("per-edge type-I error")
    ax1.set_title("(A) Per-edge type-I vs repeated looks")
    ax1.legend(); ax1.set_ylim(-0.02, max(0.5, max(n["per_edge_typeI"]) * 1.1))

    ax2.plot(s["t"], s["fwer_ebonf"], "-o", ms=3, color=GREEN,
             label="ORACLE (e-Bonferroni)")
    ax2.plot(s["t"], s["fwer_uncorrected"], "-^", ms=3, color=PURPLE, alpha=0.7,
             label="ORACLE (uncorrected union)")
    ax2.plot(n["t"], n["fwer_uncorrected"], "-s", ms=3, color=RED,
             label="Naive (uncorrected)")
    ax2.axhline(a, ls="--", color="k", lw=1, label=fr"$\alpha={a}$")
    ax2.set_xlabel("stream length $t$"); ax2.set_ylabel("FWER (any false edge)")
    ax2.set_title(f"(B) Family-wise error, $K={v['K']}$ edges")
    ax2.legend(); ax2.set_ylim(-0.02, max(0.5, max(n["fwer_uncorrected"]) * 1.1))
    _save(fig, "validity_fwer")


def fig_recovery_convergence(r):
    r0 = r[0]
    curve = r0["oracle_curve"]
    ts = sorted(int(k) for k in curve)
    if not ts:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ts, [curve[str(t)]["shd"] for t in ts], "-o", color=BLUE, label="SHD")
    ax.plot(ts, [curve[str(t)]["sid"] for t in ts], "-s", color=PURPLE, label="SID")
    ax.set_xlabel("stream length $t$"); ax.set_ylabel("distance to ground truth")
    ax.set_title(f"ORACLE convergence ($p={r0['p']}$, {r0['kind']}, {r0['noise']})")
    ax.legend()
    _save(fig, "recovery_convergence")


def fig_recovery_comparison(r):
    """Grouped bar: SHD and skeleton-F1 per method, for the first setting."""
    r0 = r[0]
    methods = ["ORACLE", "PC_fisherz", "PC_kci_window", "GES_BIC"]
    labels = ["ORACLE", "PC (Fisher-Z)", "PC (KCI)", "GES (BIC)"]
    present = [(lab, r0[m]) for m, lab in zip(methods, labels) if r0.get(m)]
    if not present:
        return
    labs = [p[0] for p in present]
    shd = [p[1]["shd"] for p in present]
    f1 = [p[1]["skeleton_f1"] for p in present]
    import numpy as np
    x = np.arange(len(labs))
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    colors = [GREEN if l == "ORACLE" else "#9e9e9e" for l in labs]
    b = ax1.bar(x - 0.2, shd, 0.4, color=colors, label="SHD (lower better)")
    ax1.set_ylabel("SHD"); ax1.set_xticks(x); ax1.set_xticklabels(labs, rotation=15)
    ax2 = ax1.twinx()
    ax2.plot(x, f1, "D-", color=BLUE, label="skeleton F1")
    ax2.set_ylabel("skeleton F1"); ax2.set_ylim(0, 1)
    ax1.set_title(f"Recovery vs baselines ($p={r0['p']}$, {r0['kind']}, "
                  f"{r0['noise']}, normalised)")
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, l1 + l2, loc="upper left", fontsize=9)
    _save(fig, "recovery_comparison")


def fig_anytime(a):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(a["grid"], a["oracle_false_curve"], "-o", ms=3, color=GREEN,
            label="ORACLE false skeleton edges")
    ax.plot(a["grid"], a["naive_false_curve"], "-s", ms=3, color=RED,
            label="Naive optional-stopping false edges")
    ax.set_xlabel("stream length $t$"); ax.set_ylabel("mean # false edges declared")
    ax.set_title(f"False-edge accumulation ($p={a['p']}$)")
    ax.legend()
    _save(fig, "anytime_false_edges")


def fig_change(c):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    arls, delays, hs = c["mean_arl"], c["mean_delay"], c["h_grid"]
    xs = [a for a, d in zip(arls, delays) if d is not None]
    ys = [d for d in delays if d is not None]
    ax.plot(xs, ys, "-o", color=BLUE)
    for h, a, d in zip(hs, arls, delays):
        if d is not None:
            ax.annotate(f"h={h}", (a, d), fontsize=8,
                        textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("ARL to false alarm (obs)"); ax.set_ylabel("detection delay (obs)")
    ax.set_title("Page-CUSUM delay vs ARL (sweep $h$)")
    _save(fig, "change_delay_arl")


def fig_systems(s):
    rows = s["rows"]
    xs = [r["p"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    ax1.plot(xs, [r["obs_per_sec"] for r in rows], "-o", color=BLUE)
    ax1.set_xlabel("variables $p$"); ax1.set_ylabel("throughput (obs/sec)")
    ax1.set_yscale("log"); ax1.set_title("Throughput vs $p$")
    ax2.plot(xs, [r["peak_mem_mb"] for r in rows], "-s", color=RED)
    ax2.set_xlabel("variables $p$"); ax2.set_ylabel("peak memory (MB)")
    ax2.set_title("Memory vs $p$")
    _save(fig, "systems_scaling")


def fig_ablation_k(ab):
    if "k" not in ab:
        return
    import numpy as np
    ks = sorted(ab["k"], key=lambda s: int(s))
    shd = [ab["k"][k]["shd"] for k in ks]
    f1 = [ab["k"][k]["orient_f1"] for k in ks]
    x = np.arange(len(ks))
    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax1.bar(x, shd, 0.5, color="#9e9e9e", label="SHD")
    ax1.set_xticks(x); ax1.set_xticklabels([f"k={k}" for k in ks])
    ax1.set_ylabel("SHD"); ax1.set_xlabel("degree cap $k$")
    ax2 = ax1.twinx()
    ax2.plot(x, f1, "D-", color=GREEN, label="orientation F1")
    ax2.set_ylabel("orientation F1"); ax2.set_ylim(0, 1)
    ax1.set_title("Degree cap $k$: pairwise ($k{=}0$) vs DAG")
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, l1 + l2, loc="upper left", fontsize=9)
    _save(fig, "ablation_degree_cap")


def main():
    print(f"Writing manuscript figures to {OUT}")
    v = _load("validity.json")
    if v:
        fig_validity(v)
    r = _load("recovery.json")
    if r:
        fig_recovery_convergence(r)
        fig_recovery_comparison(r)
    a = _load("anytime.json")
    if a:
        fig_anytime(a)
    c = _load("change.json")
    if c:
        fig_change(c)
    s = _load("systems.json")
    if s:
        fig_systems(s)
    ab = _load("ablations.json")
    if ab:
        fig_ablation_k(ab)
    print("done.")


if __name__ == "__main__":
    main()
