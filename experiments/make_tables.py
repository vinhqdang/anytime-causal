"""Generate LaTeX tables from saved experiment JSON.

Writes self-contained table floats into manuscript/tables/*.tex, which the
manuscript \\input. Re-runnable and decoupled from the experiment compute.

    python -m experiments.make_tables
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "manuscript", "tables")
os.makedirs(OUT, exist_ok=True)


def _load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _write(stem, body):
    with open(os.path.join(OUT, f"{stem}.tex"), "w") as f:
        f.write(body)
    print(f"  wrote tables/{stem}.tex")


def _f(x, nd=2):
    if x is None:
        return "--"
    return f"{x:.{nd}f}"


def _float(caption, label, header, rows, colspec):
    L = [r"\begin{table}[htbp]", r"\centering",
         rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         rf"\begin{{tabular}}{{{colspec}}}", r"\hline",
         " & ".join(header) + r" \\", r"\hline"]
    L += [" & ".join(r) + r" \\" for r in rows]
    L += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def tab_validity(v):
    rows = [
        ["max per-edge type-I", rf"\textbf{{{_f(v['skit_max_per_edge'],3)}}}",
         _f(v['naive_max_per_edge'], 3)],
        ["max FWER (any false edge)",
         rf"\textbf{{{_f(v['skit_max_fwer_ebonf'],3)}}}",
         _f(v['naive_max_fwer'], 3)],
    ]
    cap = (rf"Validity audit on a null stream ($p={v['p']}$, $K={v['K']}$ edges, "
           rf"target $\alpha={v['alpha']}$). ORACLE controls error uniformly over "
           r"the stream; naive optional stopping inflates. ORACLE column uses "
           r"e-Bonferroni for FWER.")
    _write("tab_validity", _float(cap, "tab:validity",
           ["metric", "ORACLE (SKIT)", "Naive opt.\\ stopping"], rows, "lcc"))


def tab_recovery(r):
    names = {"ORACLE": "ORACLE", "PC_fisherz": "PC (Fisher-Z)",
             "PC_kci_window": "PC (KCI)", "GES_BIC": "GES (BIC)"}
    rows = []
    for s in r:
        head = (rf"\multicolumn{{5}}{{l}}{{\emph{{$p={s['p']}$, {s['kind']}, "
                rf"{s['noise']} noise, density {s['density']}, $n={s['n']}$ }}}}"
                r" \\")
        rows.append((head, None))
        for key, lab in names.items():
            d = s.get(key)
            if not d:
                continue
            b = r"\textbf{" if key == "ORACLE" else ""
            e = "}" if key == "ORACLE" else ""
            rows.append((None, [f"\\quad {lab}", f"{b}{_f(d['shd'],1)}{e}",
                                _f(d['sid'], 1), _f(d['skeleton_f1']),
                                _f(d['orient_f1'])]))
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Graph recovery on normalised non-Gaussian ANM data: ORACLE "
         r"(online) versus batch baselines (checkpoint harness). SHD/SID lower is "
         r"better; F1 higher is better.}",
         r"\label{tab:recovery}",
         r"\begin{tabular}{lcccc}", r"\hline",
         r"method & SHD & SID & skel.\ F1 & orient.\ F1 \\", r"\hline"]
    for head, row in rows:
        if head:
            L.append(r"\hline")
            L.append(head)
            L.append(r"\hline")
        else:
            L.append(" & ".join(row) + r" \\")
    L += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    _write("tab_recovery", "\n".join(L))


def tab_change(c):
    rows = [[str(h), _f(d, 0), _f(a, 0), _f(dr, 2)]
            for h, d, a, dr in zip(c["h_grid"], c["mean_delay"], c["mean_arl"],
                                   c["detection_rate"])]
    cap = (rf"Change detection ($p={c['p']}$, segment $={c['segment_len']}$, "
           rf"{c['n_changes']} change-points, {c['n_monitored_edges']} monitored "
           r"edges). Page--CUSUM threshold $h$ trades delay against ARL to false "
           r"alarm.")
    _write("tab_change", _float(cap, "tab:change",
           ["$h$", "mean delay", "mean ARL", "det.\\ rate"], rows, "cccc"))


def tab_systems(s):
    rows = [[str(r["p"]), str(r["n_pairs"]), str(r["n_features"]),
             _f(r["obs_per_sec"], 0), _f(r["peak_mem_mb"], 1)] for r in s["rows"]]
    cap = (rf"Systems scaling (stream length $n={s['n']}$): throughput and peak "
           r"memory versus the number of variables $p$.")
    _write("tab_systems", _float(cap, "tab:systems",
           ["$p$", "pairs", "RFF $D$", "obs/sec", "mem (MB)"], rows, "ccccc"))


def _ablation_table(stem, label, caption, keyname, d):
    rows = [[k, _f(v["shd"], 1), _f(v["sid"], 1), _f(v["skeleton_f1"]),
             _f(v["orient_f1"])] for k, v in d.items()]
    _write(stem, _float(caption, label,
           [keyname, "SHD", "SID", "skel.\\ F1", "orient.\\ F1"], rows, "lcccc"))


def tab_ablations(ab):
    if "alpha" in ab:
        _ablation_table("tab_ablation_alpha", "tab:abl_alpha",
                        "Ablation: significance level $\\alpha$.", "$\\alpha$",
                        ab["alpha"])
    if "k" in ab:
        _ablation_table("tab_ablation_k", "tab:abl_k",
                        "Ablation: degree cap $k$ ($k=0$ is a pairwise "
                        "dependency graph, not a DAG).", "$k$", ab["k"])
    if "bet" in ab:
        _ablation_table("tab_ablation_bet", "tab:abl_bet",
                        "Ablation: bet sizing.", "bet", ab["bet"])
    if "n_features" in ab:
        _ablation_table("tab_ablation_features", "tab:abl_features",
                        "Ablation: RFF feature count $D$.", "$D$",
                        ab["n_features"])
    if "multiplicity" in ab:
        _ablation_table("tab_ablation_multiplicity", "tab:abl_mult",
                        "Ablation: multiplicity control.", "method",
                        ab["multiplicity"])
    if "noise" in ab:
        _ablation_table("tab_ablation_noise", "tab:abl_noise",
                        "Ablation: noise family (Gaussian is the "
                        "non-identifiable control).", "noise", ab["noise"])
    if "warmup" in ab:
        _ablation_table("tab_ablation_warmup", "tab:abl_warmup",
                        "Ablation: warm-up window length.", "warm-up",
                        ab["warmup"])
    if "recond_every" in ab:
        _ablation_table("tab_ablation_recond", "tab:abl_recond",
                        "Ablation: re-conditioning period (timesteps between "
                        "separating-set updates).", "recond.\\ period",
                        ab["recond_every"])
    if "density" in ab:
        _ablation_table("tab_ablation_density", "tab:abl_density",
                        "Ablation: graph density (expected edges per node).",
                        "density", ab["density"])
    if "stream_length" in ab:
        _ablation_table("tab_ablation_streamlen", "tab:abl_streamlen",
                        "Ablation: stream length $n$ (sample efficiency).", "$n$",
                        ab["stream_length"])
    if "standardization" in ab:
        st = ab["standardization"]
        rows = [[k, _f(st[k]["rejection_rate"], 2),
                 _f(st[k]["median_detection"], 0)]
                for k in st]
        _write("tab_ablation_standardization", _float(
            "Ablation: witness standardisation (SKIT-level power study on a "
            "fixed dependent stream). Standardisation turns the small HSIC "
            "signal into an order-one payoff.",
            "tab:abl_std",
            ["standardisation", "rejection rate", "median detection"],
            rows, "lcc"))


def main():
    print(f"Writing LaTeX tables to {OUT}")
    v = _load("validity.json")
    if v:
        tab_validity(v)
    r = _load("recovery.json")
    if r:
        tab_recovery(r)
    c = _load("change.json")
    if c:
        tab_change(c)
    s = _load("systems.json")
    if s:
        tab_systems(s)
    ab = _load("ablations.json")
    if ab:
        tab_ablations(ab)
    print("done.")


if __name__ == "__main__":
    main()
