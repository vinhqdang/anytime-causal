"""ORACLE: the online anytime-valid causal discovery algorithm.

Pipeline per timestep (mirrors algorithm.md sec 3.1):

  1. Skeleton: per unordered pair {i,j}, a sequential conditional-dependence
     test SKIT_CI(X_i, X_j | S) where S is the current estimated neighbour set
     (RESIT reduction, degree-capped at k). Conditioning on the parent/neighbour
     set is what makes the recovered graph a DAG rather than a dependency graph
     (sec 1 consistency claim). An edge is present iff the test currently rejects
     conditional independence; when S grows to include a true separator the test
     is reset and stops rejecting, so the spurious edge drops.

  2. Orientation: ANM residual asymmetry with the CORRECTED sign convention.
     For an adjacent pair, orient toward the direction whose residual is
     INDEPENDENT of the input; equivalently, REJECT the anti-causal direction
     (its residual stays dependent). Wealth A tests dependence of the forward
     residual on X_i, wealth B the reverse on X_j.

  3. Multiplicity: e-BH (Wang-Ramdas) over the per-pair CI wealths -> an
     FDR-controlled active edge set. e-Bonferroni available as the FWER option.

  4. DAG projection: add oriented surviving edges; if an edge closes a cycle,
     drop the cycle edge with the smallest log-wealth margin.

  5. Change detection: Page-CUSUM on each active edge's log e-increments; an
     alarm resets that pair's wealths and regressors.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import networkx as nx

from oracle.skit import SKIT, PageCUSUM
from oracle.skit_ci import SKIT_CI
from oracle.online_reg import OnlineRFFRidge
from oracle.ebh import ebh, e_bonferroni


class _Pair:
    """State for one unordered pair {i, j} with i < j."""

    def __init__(self, i, j, alpha_skel, alpha_orient, n_features, warmup, seed,
                 bet="mixture"):
        self.i = i
        self.j = j
        self.alpha_skel = alpha_skel
        self.alpha_orient = alpha_orient
        self.n_features = n_features
        self.warmup = warmup
        self.seed = seed
        self.bet = bet

        self.sepset = ()                  # current conditioning set
        self.ci = SKIT_CI(0, alpha=alpha_skel, n_features=n_features,
                          warmup=warmup, seed=seed, bet=bet)
        # orientation regressors & wealths (bivariate ANM, per direction)
        self.reg_fwd = OnlineRFFRidge(1, warmup=warmup, seed=seed + 11)  # i -> j
        self.reg_rev = OnlineRFFRidge(1, warmup=warmup, seed=seed + 12)  # j -> i
        self.A = SKIT(1, 1, alpha=alpha_orient, n_features=n_features,
                      warmup=warmup, bet=bet, seed=seed + 13)   # resid_j dep on X_i
        self.B = SKIT(1, 1, alpha=alpha_orient, n_features=n_features,
                      warmup=warmup, bet=bet, seed=seed + 14)   # resid_i dep on X_j
        self.cusum = PageCUSUM()

    def set_sepset(self, S):
        S = tuple(sorted(S))
        if S != self.sepset:
            self.sepset = S
            self.ci = SKIT_CI(len(S), alpha=self.alpha_skel,
                              n_features=self.n_features, warmup=self.warmup,
                              seed=self.seed, bet=self.bet)

    @property
    def adjacent(self) -> bool:
        return self.ci.rejected

    @property
    def ci_evalue(self) -> float:
        return self.ci.wealth

    def reset_after_alarm(self):
        self.ci.reset()
        self.reg_fwd.reset()
        self.reg_rev.reset()
        self.A.reset()
        self.B.reset()
        self.cusum.reset()


class ORACLE:
    def __init__(self, p, alpha=0.05, alpha_skel=None, alpha_orient=None,
                 k=2, n_features=64, warmup=40, recond_every=200,
                 cusum_c=1.0, cusum_h=12.0, multiplicity="ebh", bet="mixture",
                 seed=0):
        self.p = p
        self.alpha = alpha
        self.alpha_skel = alpha_skel if alpha_skel is not None else alpha
        self.alpha_orient = alpha_orient if alpha_orient is not None else alpha
        self.k = k
        self.recond_every = recond_every
        self.multiplicity = multiplicity
        self.cusum_c = cusum_c
        self.cusum_h = cusum_h
        self.t = 0
        self.alarms = []   # (t, i, j)

        rng = np.random.default_rng(seed)
        self.pairs = {}
        for i, j in combinations(range(p), 2):
            s = int(rng.integers(1, 1 << 30))
            pr = _Pair(i, j, self.alpha_skel, self.alpha_orient,
                       n_features, warmup, s, bet=bet)
            pr.cusum.c = cusum_c
            pr.cusum.h = cusum_h
            self.pairs[(i, j)] = pr

    # ------------------------------------------------------------------ #
    def _neighbors(self):
        nb = {v: set() for v in range(self.p)}
        for (i, j), pr in self.pairs.items():
            if pr.adjacent:
                nb[i].add(j)
                nb[j].add(i)
        return nb

    def _update_sepsets(self):
        nb = self._neighbors()
        for (i, j), pr in self.pairs.items():
            cand = (nb[i] | nb[j]) - {i, j}
            if not cand:
                S = ()
            else:
                # rank candidates by their edge evidence; keep top-k
                scored = sorted(cand, key=lambda v: self._edge_strength(v, i, j),
                                reverse=True)
                S = tuple(sorted(scored[: self.k]))
            pr.set_sepset(S)

    def _edge_strength(self, v, i, j) -> float:
        best = 0.0
        for other in (i, j):
            key = (min(v, other), max(v, other))
            if key in self.pairs:
                best = max(best, self.pairs[key].ci.log_wealth)
        return best

    # ------------------------------------------------------------------ #
    def step(self, x):
        """Process one observation x (length-p vector). Returns the current DAG."""
        x = np.asarray(x, dtype=float)
        self.t += 1

        if self.t % self.recond_every == 0:
            self._update_sepsets()

        for (i, j), pr in self.pairs.items():
            s = x[list(pr.sepset)] if pr.sepset else None
            pr.ci.update(x[i], x[j], s)

            # change detection on the CI e-increment of active edges
            if pr.adjacent:
                if pr.cusum.update(pr.ci.last_log_increment, self.t):
                    self.alarms.append((self.t, i, j))
                    pr.reset_after_alarm()
                    continue

            # orientation only for adjacent pairs
            if pr.adjacent:
                res_fwd = pr.reg_fwd.update(x[i], x[j])   # X_j - f(X_i)
                res_rev = pr.reg_rev.update(x[j], x[i])   # X_i - f(X_j)
                pr.A.update(res_fwd, x[i])   # dependence of fwd residual on X_i
                pr.B.update(res_rev, x[j])   # dependence of rev residual on X_j

        return self.graph()

    # ------------------------------------------------------------------ #
    def _orient(self, pr):
        """Return 'fwd' (i->j), 'rev' (j->i), or None (undirected)."""
        a_rej = pr.A.wealth >= 1.0 / self.alpha_orient
        b_rej = pr.B.wealth >= 1.0 / self.alpha_orient
        # true i->j  =>  fwd residual independent of X_i (A low) and
        #               rev residual dependent on X_j   (B high)
        if b_rej and not a_rej:
            return "fwd"
        if a_rej and not b_rej:
            return "rev"
        return None

    def graph(self, project=True):
        """Current estimated DAG adjacency (A[i,j]=1 => i->j)."""
        # 1. multiplicity-controlled active (undirected) edge set
        keys = list(self.pairs.keys())
        evalues = [self.pairs[k].ci_evalue for k in keys]
        K = self.p * (self.p - 1) // 2
        if self.multiplicity == "bonferroni":
            sel = e_bonferroni(evalues, self.alpha, n_hypotheses=K)
        elif self.multiplicity == "none":
            sel = [idx for idx, k in enumerate(keys) if self.pairs[k].adjacent]
        else:
            sel = ebh(evalues, self.alpha, n_hypotheses=K)
        active = [keys[idx] for idx in sel if self.pairs[keys[idx]].adjacent]

        # 2. orientation
        candidates = []  # (i, j, margin) directed
        A = np.zeros((self.p, self.p), dtype=int)
        for (i, j) in active:
            pr = self.pairs[(i, j)]
            d = self._orient(pr)
            margin = max(pr.A.log_wealth, pr.B.log_wealth)
            if d == "fwd":
                candidates.append((i, j, margin))
            elif d == "rev":
                candidates.append((j, i, margin))
            # undirected edges are left out of the directed DAG

        if not project:
            for (a, b, _) in candidates:
                A[a, b] = 1
            return A

        # 3. DAG projection: add edges by decreasing margin, skip cycle-closers
        candidates.sort(key=lambda x: -x[2])
        g = nx.DiGraph()
        g.add_nodes_from(range(self.p))
        for (a, b, _) in candidates:
            g.add_edge(a, b)
            if not nx.is_directed_acyclic_graph(g):
                g.remove_edge(a, b)  # smallest-margin-first drop via sort order
        for a, b in g.edges():
            A[a, b] = 1
        return A

    def skeleton(self):
        A = np.zeros((self.p, self.p), dtype=int)
        for (i, j), pr in self.pairs.items():
            if pr.adjacent:
                A[i, j] = A[j, i] = 1
        return A
