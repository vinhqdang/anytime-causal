# ORACLE: Online Anytime-Valid Causal Discovery — Developer Implementation & Evaluation Spec

*Corrected redesign of the original ORACLE draft (v0.1, June 2026). This document keeps the ORACLE name and the core idea (anytime-valid streaming causal discovery via e-processes) but fixes the broken statistical core. It is written for an ML engineer to implement and evaluate from scratch, and specifies baselines, algorithm, datasets, and metrics.*

---

## 0. What changed vs. the original draft (read this first)

| Original draft | Problem | Corrected ORACLE |
|---|---|---|
| `e = exp(HSIC·λ)` (always ≥ 1) | Not a valid e-value; wealth always diverges | **SKIT betting wealth** `K_t = K_{t-1}(1 + λ_t g_t)`, `E[g_t \| F_{t-1}] ≤ 0` under H₀ |
| Activate edge when wealth/dependence HIGH; orient toward higher dependence | Sign-inverted vs. ANM | Reject the **anti-causal** direction: orient toward the direction whose **residual is independent** of the input |
| Regress `X_j` on single `X_i` | Cannot identify a DAG (recovers dependency graph) | **Condition** on estimated parents via online residualization (RESIT reduction) or sequential CI-by-betting |
| λ, bandwidth "calibrated"/"incremental" | Breaks prequential validity | λ, bandwidth chosen from **past data only** (previsible) |
| Bonferroni over p(p−1) | Power collapse at scale | **e-BH** (Wang–Ramdas) for FDR; e-Bonferroni only as conservative option |
| `S_t = L_t − min L_s` | Degenerate under increasing wealth | **Page-CUSUM** `G_t = max(0, G_{t-1} + log e_t − c)` |
| SHD only; F1 on financial/ICU data | No causal ground truth there | Add **SID**; restrict graph metrics to ground-truthed data |
| — | No falsification test | Add a **validity audit** under simulated independence (H₀) |

---

## 1. Problem setting & guarantees to target

Observations `X_1, X_2, … ∈ ℝ^p` arrive sequentially from a (piecewise-)stationary SCM. At each `t` output an estimated DAG `Ĝ_t` and optional change alarms.

**Assumptions:** (A1) causal sufficiency; (A2) faithfulness; (A3) additive non-Gaussian noise `X_j = f_j(X_{Pa(j)}) + ε_j`, `ε_j ⊥ X_{Pa(j)}`; (A4) characteristic kernel.

**Guarantees to claim (and test):**
- **Anytime type-I / FDR control:** at any stopping time `τ` and level `α`, the false-edge rate is ≤ α — *with no correction for repeated looks*. Proven per-edge by Ville's inequality on a valid nonnegative supermartingale; combined across edges by e-BH or e-Bonferroni.
- **Sequential consistency:** `Ĝ_t → G*` a.s. under (A1)–(A4) **when conditioning is on the parent set**, not a single variable.
- **Shift detection:** Page-CUSUM delay `O(h / KL(P_new‖P_old))` with controlled average run length (ARL) to false alarm.

> Do **not** claim "no assumptions on the data-generating process." The error guarantee requires that each per-edge increment be a valid e-value; that is itself a (mild, construction-level) assumption.

---

## 2. The statistical core: a valid sequential test (SKIT betting wealth)

This replaces `exp(HSIC·λ)`. Reference implementation to mirror: **SKIT** (Podkopaev, Blöbaum, Kasiviswanathan, Ramdas, ICML 2023), repo `a-podkopaev/Sequential-Kernelized-Independence-Testing`.

### 2.1 Independence-by-betting primitive `SKIT(U, V)`
Tests `H₀: U ⊥ V` from a stream of paired samples `(u_t, v_t)`.

```
state: wealth K ← 1; previsible bet λ ∈ [0, 1); running estimates of kernel mean embeddings
on new pair (u_t, v_t):
    # payoff g_t built from a witness function evaluated on PAST data only
    ĝ ← μ̂_UV − μ̂_U ⊗ μ̂_V            # plug-in witness from samples < t  (previsible)
    g_t ← bounded_payoff(ĝ, u_t, v_t)  # E[g_t | F_{t-1}] ≤ 0 under H₀; |g_t| ≤ 1 (kernel bounded to [0,1])
    K ← K · (1 + λ · g_t)              # nonnegative supermartingale under H₀
    update μ̂_U, μ̂_UV, ...            # AFTER using them, so betting stays previsible
    λ ← aGRAPA_or_ONS_update(λ, g_t)   # previsible bet sizing (uses history ≤ t)
    return K
```

Key implementation rules (each is a place the original draft silently broke validity):
- **Previsibility:** the witness `ĝ`, the bet `λ`, and the **kernel bandwidth** must be functions of samples strictly before `t`. Use a warm-up window to initialise; never recompute bandwidth including the current point.
- **Bounded kernel:** normalise kernels to `[0,1]` so `|g_t| ≤ 1` and `K` stays nonnegative. Use SKIT's symmetry-based payoff (its Eq. 25) to drop the boundedness restriction if you need RBF without clipping.
- **Bet sizing:** aGRAPA or ONS (online Newton step) on `λ ∈ [0, 1−δ)`. Mixture-of-bets (grid over λ, average wealth) is a robust default.
- **Reject** `H₀` (declare dependence) the moment `K ≥ 1/α`. This is anytime-valid by Ville.

### 2.2 Conditional independence `SKIT_CI(U, V | S)`
Two acceptable constructions; pick one and document it:

**(a) RESIT residual reduction (default, scalable).** Maintain an online multivariate regressor `f̂_{U|S}` and `f̂_{V|S}` (online random-feature ridge, online sparse GP, or a small online MLP). Residualise `ũ_t = u_t − f̂_{U|S}(s_t)`, `ṽ_t = v_t − f̂_{V|S}(s_t)`, then run `SKIT(ũ, ṽ)`. Valid under additive structure (A3). Regressors must also be previsible (predict `t` from model fit on `< t`, then update).

**(b) Model-free CI-by-betting (rigorous).** Use Shaer, Maman & Romano, *Model-free sequential testing for conditional independence via testing by betting* (AISTATS 2023). Heavier but assumption-lean.

---

## 3. ORACLE algorithm

State per ordered pair and node:
- `Pa[j]` — current estimated parent set of node `j` (drives the regressors).
- `f̂_j` — online regressor `X_{Pa[j]} → X_j`.
- **Skeleton wealths** `K_skel[i][j][S]` — CI tests `X_i ⊥ X_j | S` for bounded-degree `S` (degree cap `k`).
- **Orientation wealths** `A[i][j]`, `B[i][j]` — residual-independence tests for the two directions (defined below).
- **Change-detection CUSUM** `G[i][j]` — Page statistic per active edge.

### 3.1 Per-timestep pipeline

```
Algorithm ORACLE(stream; α_skel, α_orient, k, h, c, m)

for each X_t = (X_{1,t},...,X_{p,t}):

  # ---- 1. Skeleton: anytime-valid conditional dependence ----
  for each unordered pair {i,j}, for each S ⊆ adj-candidates, |S| ≤ k:
      K_skel[i][j][S] ← SKIT_CI(X_i, X_j | S)      # using residual reduction
  # adjacency rule (PC-style, but sequential):
  #   {i,j} ADJACENT iff NO conditioning set has shown independence
  #   i.e. for all tested S, K_skel[i][j][S] has NOT yet "accepted" independence
  #   AND at least one test has rejected independence (K ≥ 1/α_skel).
  # Separator found (some S makes them independent) ⇒ remove edge, record sepset(i,j)=S.

  # ---- 2. Orientation: ANM residual asymmetry (CORRECTED SIGNS) ----
  for each ADJACENT pair {i,j}:
      a_t ← X_{j,t} − f̂_{j|i}(X_{i,t})            # residual, forward direction i→j
      b_t ← X_{i,t} − f̂_{i|j}(X_{j,t})            # residual, reverse direction j→i
      A[i][j] ← SKIT(a, X_i)   # wealth that residual_of_j is DEPENDENT on X_i
      B[i][j] ← SKIT(b, X_j)   # wealth that residual_of_i is DEPENDENT on X_j
      # ANM: true i→j ⇒ a ⊥ X_i (A stays low) and b ⊄⊥ X_j (B grows).
      if B[i][j] ≥ 1/α_orient and A[i][j] < 1/α_orient:  orient i→j
      elif A[i][j] ≥ 1/α_orient and B[i][j] < 1/α_orient: orient j→i
      else: leave undirected   # both grow ⇒ misspecification/confounding; both low ⇒ undecided

  # ---- 3. Multiplicity control across all candidate edges ----
  # collect the per-edge e-values (the rejecting wealth for each declared edge)
  # apply e-BH (Wang–Ramdas 2022) at level α  →  FDR-controlled active edge set
  # (e-Bonferroni: threshold each wealth at p(p-1)/α  →  conservative FWER option)

  # ---- 4. DAG projection (report BOTH pre- and post-projection graphs) ----
  add oriented, surviving edges to Ĝ_t
  if a new edge closes a cycle:
      drop the edge in the cycle with the SMALLEST rejection margin (lowest log-wealth)
  maintain incremental topological order

  # ---- 5. Change detection: Page-CUSUM on each active edge's e-increments ----
  for each edge (i→j) in Ĝ_t:
      G[i][j] ← max(0, G[i][j] + log e_t^{i→j} − c)   # c>0 drift compensation
      if G[i][j] > h:
          ALARM(i→j); reset wealths A,B and CUSUM for (i,j); reinitialise f̂_{j|i}
          flag Markov blanket of j for re-estimation

  output Ĝ_t
```

### 3.2 Notes the implementer must not skip
- **Sign convention is the whole point.** Edges are oriented by *rejecting* the wrong direction (residual still dependent), not by maximising dependence. Unit-test this on a 2-node ANM where you know the answer.
- **Degree cap `k`** controls the cost/correctness tradeoff: `k = 0` reduces to a pairwise dependency graph (fast, *not* a DAG); `k ≥ max-in-degree` is needed for correctness. Default `k = 2–3`, plus Markov-blanket pruning of candidate `S`.
- **Reset semantics after an alarm:** restart the affected wealths from `K = 1` so post-change evidence accumulates cleanly; do not carry stale wealth.

---

## 4. Complexity (honest accounting)

| Component | Cost / timestep | Memory |
|---|---|---|
| Residual regressors (all pairs/nodes, budget `m`) | `O(p² m)` | `O(p² m)` support points |
| SKIT wealth updates | `O(p²)` per conditioning set | windowed stats `O(p² w)` |
| Skeleton CI with degree cap `k` | `O(p² · C(p,k))` worst case | grows with tested sepsets |
| e-BH over active e-values | `O(p² log p)` | `O(p²)` |
| Page-CUSUM | `O(p²)` | `O(p²)` |

For real-time `p ≤ 50`, use `k ≤ 2`, Markov-blanket pruning, and **Random Fourier Features** (`D ≈ 512`) for the kernel statistics. Report measured throughput, not just asymptotics.

---

## 5. Baselines

Group baselines by the claim they stress-test. Each entry below gives **what it does / mechanism**, **`Impl:`** (library or repo and how to call it), and **`Streaming:`** (how to fit it into ORACLE's online evaluation). The single most important framing to preserve throughout: **none of these competitors provide finite-sample error control at every stopping time** — that anytime-validity axis is ORACLE's differentiator, so report it as a column in every comparison table, not just accuracy.

### 5.0 How to run a *batch* baseline in a *streaming* evaluation (read before implementing any of §5.1–5.3)
ORACLE outputs `Ĝ_t` at every `t`. Batch methods can't, so wrap them in a **checkpoint harness**:
1. Pick checkpoints `t ∈ {t₁, t₂, …}` (e.g. every 250 observations).
2. At each checkpoint, run the batch method on either the **expanding window** `X_{1:t}` (tests asymptotic quality) or a **sliding window** `X_{t−w+1:t}` of fixed width `w` (the only honest way to give a batch method a fighting chance at shift detection).
3. Record its output graph and wall-clock time; compare to ORACLE's `Ĝ_t` at the same `t`.
4. For "anytime FWER": run the batch method's internal test **at every observation with no multiplicity correction**, take the union of all edges it would have declared up to `t`. This is the optional-stopping inflation you are demonstrating (§5.4).

Use one CI-test backend everywhere (KCI for nonlinear, Fisher-Z for linear) so differences come from the algorithm, not the test. Install once: `pip install causal-learn gcastle tigramite ruptures`.

### 5.1 Classical / continuous-optimization (graph-quality floor — KEEP)
- **PC-stable** (Colombo & Maathuis 2014). Constraint-based: start from the complete undirected graph, remove edge `i–j` whenever a conditioning set `S` makes them conditionally independent, then orient v-structures and apply Meek's rules. Order-independent variant ("stable"). Outputs a CPDAG. `Impl:` `from causallearn.search.ConstraintBased.PC import pc; cg = pc(X, alpha=0.05, indep_test='kci')`. `Streaming:` checkpoint harness; this is also the algorithm whose *per-step, uncorrected* version is your optional-stopping straw man.
- **GES** (Chickering 2002). Score-based: greedily add edges to improve a decomposable score (BIC/BDeu), then greedily remove; searches over CPDAG space. `Impl:` `from causallearn.search.ScoreBased.GES import ges; rec = ges(X, score_func='local_score_BIC')`. `Streaming:` checkpoint harness.
- **RESIT** (Peters et al. 2014) — *the batch ANM method ORACLE is the online analogue of; include it as the primary apples-to-apples quality reference.* Mechanism: (1) regress each variable on **all others** (nonlinear regressor, e.g. GP/GAM); (2) the variable whose residual is **independent of all others** (HSIC test) is a **sink** — remove it; (3) recurse on the rest to get a topological order; (4) prune edges by testing whether each parent is still needed. `Impl:` no maintained pip package — port the authors' R code (`CAM`/`RESIT`) or implement directly: scikit-learn `GaussianProcessRegressor` + an HSIC test (`causal-learn`'s `Hsic`/`KCI`). `Streaming:` checkpoint harness; expect it to be the strongest non-amortized competitor on ANM data.
- **NOTEARS** (Zheng et al. 2018). Recasts DAG search as continuous optimization with a smooth acyclicity penalty `h(W)=tr(e^{W∘W})−d` solved by augmented Lagrangian; linear and MLP variants. `Impl:` `gcastle` (`from castle.algorithms import Notears`) or repo `xunzheng/notears`. `Streaming:` checkpoint harness; standardize inputs (varsortability caveat, §6).
- **DAG-GNN** (Yu et al. 2019). VAE whose encoder/decoder are graph-structured; optimizes an ELBO under the same acyclicity constraint. `Impl:` `gcastle` (`from castle.algorithms import DAG_GNN`) or repo `fishmoon1234/DAG-GNN`. `Streaming:` checkpoint harness.
- **DCDI** (Brouillard et al. 2020 — *not* DECI). Differentiable score over neural mechanisms (incl. normalizing flows) that can use interventional data; augmented-Lagrangian acyclicity. `Impl:` repo `slachapelle/dcdi` (PyTorch). `Streaming:` observational-only mode; checkpoint harness.
- **DiBS** (Lorch et al. 2021). Bayesian: variational inference over a latent graph embedding, giving a posterior over DAGs (report the posterior-mean/MAP edge probabilities). `Impl:` repo `larslorch/dibs` (JAX). `Streaming:` checkpoint harness; gives you a calibrated-probability comparator for ORACLE's wealth-derived confidences.

### 5.2 Nonstationary & change-point causal discovery (ORACLE's *direct* competitors — highest priority)
These target shifting structure and/or change-points, exactly ORACLE's niche. ORACLE must win on **validity and detection delay**, not just F1.
- **CD-NOD** (Huang et al., JMLR 2020). Adds a **surrogate time/domain index** `C` as an extra variable, then runs constraint-based discovery; edges into a variable that depend on `C` flag a changing mechanism, and the framework recovers both the graph and where mechanisms shift. `Impl:` directly available — `from causallearn.search.ConstraintBased.CDNOD import cdnod; cg = cdnod(X, c_indx, alpha=0.05, indep_test='kci')` where `c_indx` is the time index column. `Streaming:` run in the checkpoint harness with `c_indx = timestep`; its change flags are the natural comparator for ORACLE's CUSUM alarms.
- **CD-NOTS** (Sadeghi, Gopal, Fesanghary 2025). Extends CD-NOD to **time series**: builds a time-lagged variable set (lags up to `τ_max`), enforces graph consistency across time, handles pseudo-confounders, nonparametric so it captures nonlinear/non-Gaussian relations. `Impl:` check the authors' release; if unavailable, reconstruct as **CD-NOD + lag augmentation** (concatenate `X_{t}, X_{t−1}, …, X_{t−τ_max}` as columns, forbid future→past edges). `Streaming:` sliding window; natural comparator for the S&P case study (its original application is finance).
- **SpaceTime** (Mameche et al. 2025, arXiv 2501.10235) — **closest competitor; benchmark detection delay head-to-head.** Uses the **Minimum Description Length** principle to score, with nonparametric functional modeling + kernelized discrepancy testing, *jointly* discovering regime change-points and a (temporal) causal graph across heterogeneous environments. `Impl:` check the authors' GitHub for released code; if porting, the core is an MDL score that trades off model complexity against fit, optimized greedily over (graph, change-point) configurations. `Streaming:` offline-by-design — run via checkpoint harness and compare its recovered change-points to your planted `t*`.
- **Conditionally-stationary TS causal discovery** (Balsells-Rodas et al., ICML 2025). Models nonstationarity as stationarity *conditioned on latent state variables* (regime switching), and identifies the number of regimes without assuming it known. `Impl:` check authors' release. `Streaming:` checkpoint harness; report regime-count recovery vs. ORACLE's alarm count.
- **Causal-discovery-driven change-point detection** (Gao et al. 2024/2025, arXiv 2407.07290) and **Causal change-point detection & localization** (Huang, Peters & Pfister 2024, arXiv 2403.12677). Change-point-**first** framings: detect when the causal model changes and localize it. `Impl:` check authors' releases (Peters-group code is usually R/Python). `Streaming:` these are the cleanest comparators for the *change-detection* metrics specifically (delay/ARL), separate from graph recovery.
- **PCMCI+** (Runge 2020). Lagged + contemporaneous CI-based discovery for time series; the standard temporal-CI reference. `Impl:` `tigramite` (`from tigramite.pcmci import PCMCI`). `Streaming:` sliding window; batch but expected.

### 5.3 Amortized / foundation-model causal discovery (the new "SOTA" reviewers will ask for)
Train-on-synthetic, infer-in-one-forward-pass. Fast and strong on synthetic SCMs but **batch** and with **no per-instance error guarantee** — frame the comparison on that gap.
- **AVICI** (Lorch et al., NeurIPS 2022) — easiest drop-in. A permutation-invariant transformer trained on simulated SCMs that maps a data matrix `[n,d]` to a `[d,d]` matrix of edge probabilities. `Impl:` `pip install avici`; `import avici; model = avici.load_pretrained(download='scm-v0'); g_prob = model(X)`. `Streaming:` call at each checkpoint on `X_{1:t}` (or sliding window); threshold edge probs at 0.5. Pretrained domains include linear-Gaussian, RFF-Gaussian, and gene-network priors — pick the one matching your synthetic generator.
- **CSIvA** (Ke et al., 2023). Transformer for amortized structure inference, trained across many synthetic SCMs; mechanism similar to AVICI with a different attention/encoding scheme. `Impl:` check authors' release; AVICI is the practical substitute if code is unavailable.
- **SEA — Sample, Estimate, Aggregate** (2024, arXiv 2402.01929). A foundation-model *recipe*: run cheap classical estimators on many subsampled subsets, then learn to **aggregate** their outputs with a transformer — scales amortized discovery to larger `d`. `Impl:` check the paper's released code. `Streaming:` checkpoint harness.
- **CausalPFN** (Balazadeh et al., 2025) and **Do-PFN** (Robertson et al., 2025). Prior-data-fitted networks doing **in-context** causal inference (no per-dataset training); CausalPFN targets discovery/effects, Do-PFN targets interventional queries. `Impl:` check authors' releases. *Caveat worth reporting:* Do-PFN's target quantity is not identified from observational data, so its posterior may not concentrate — a concrete example of why ORACLE's explicit identifiability assumptions matter.
- **TabPFN-as-causal-discoverer** (Swelam et al., Nov 2025). Probes a **frozen TabPFN**'s mid-layer embeddings (which encode causal information) with lightweight adapters; reported to match AVICI and beat GIES/IGSP on synthetic SCMs. `Impl:` `pip install tabpfn` + the adapter from the paper. A cheap, strong baseline. `Streaming:` checkpoint harness.
- **Cite, don't just run:** Montagna et al. (*Demystifying amortized causal discovery with transformers*, TMLR 2024/2026) show these models still obey identifiability theory and fail to generalize to SCM classes unseen in training — the explanation for any case where ORACLE's assumption-explicit guarantees beat a foundation model on out-of-prior data.

### 5.4 Anytime-validity stress test (the core claim — build this yourself)
- **Naive optional stopping.** Run a standard p-value CI test (KCI or HSIC-permutation, both in `causal-learn`) on the residual/pair at **every** observation and "declare an edge" the first time `p < α`, with **no** multiplicity correction. `Impl:` ~30 lines: maintain the growing sample, recompute the test each step, record first-crossing time and the union of declared edges. `Expected result:` empirical type-I error far exceeds α and climbs with stream length — the exact failure ORACLE's martingale construction avoids. This is the most persuasive single plot for the anytime claim.

### 5.5 Change detection (generic — KEEP)
- **Kernel change-point detection (KCpD) / scan statistics.** Detect distribution shifts in a windowed stream via kernel MMD between adjacent windows. `Impl:` `ruptures` (`import ruptures; algo = ruptures.KernelCPD(kernel='rbf').fit(signal)`). `Streaming:` online windowed mode. Compare detected change-points to planted `t*`.
- **BOCPD** (Bayesian Online Change-point Detection, Adams & MacKay 2007). Maintains a run-length posterior; alarms when it collapses. `Impl:` `bayesian_changepoint_detection` package. `Streaming:` genuinely online — the fairest non-causal change-detection comparator for ORACLE's CUSUM.
- **Re-run-and-diff.** Re-run any §5.1 method per window and diff consecutive graphs; the "obvious" baseline that ORACLE should beat on both latency and false-alarm rate.

---

## 6. Datasets

**Ground-truthed (use for SHD/SID/F1 — required for the main claims):**
- **Synthetic random DAGs** — primary workhorse. Erdős–Rényi and scale-free, `p ∈ {10, 20, 50}`, density `d ∈ {1, 2, 4}`. ANM with **non-Gaussian** noise (Laplace, Student-t₃, Gumbel) and nonlinear `f_j` sampled from a GP. Stream length ≥ 10⁴. **Generate the data yourself so the ground-truth DAG and any planted change-points are known exactly.**
  - **Mandatory caveat (reviewers will check this):** standardize/normalize variables and verify low **varsortability** (Reisach et al., *Beware of the simulated DAG!*, NeurIPS 2021). Raw simulated SCMs leak the topological order through marginal variance, letting trivial methods "win." Report results on normalized data.
- **Piecewise-stationary synthetic DAGs** — for shift detection: plant `1–3` known change-points where edges are added/removed or mechanisms change; record `t*` for delay/ARL measurement.
- **New ground-truthed time-series benchmarks (add these — they fit a streaming method directly):**
  - **CausalRivers** (ICLR 2025) — largest in-the-wild time-series CD benchmark; real river-network data with ground-truth graphs, 1000+ nodes. Ideal stress test for streaming throughput + real structure.
  - **CausalDynamics** (NeurIPS 2025 D&B) — thousands of coupled ODE/SDE systems with true graphs; noisy, confounded, lagged regimes; includes two climate models. Good for the nonstationary claims.
  - **CausalTime** (ICLR 2024) — realistically generated time series with recoverable ground truth.
  - **causalAssembly** (CLeaR 2024) — realistic manufacturing/production data; complements your SmartOSC industrial angle.
- **DREAM3 / DREAM4** gene-regulatory in-silico challenges — known ground truth, standard.
- **Sachs** protein-signaling (consensus network) — classic; use windowed streaming.

**Robustness / misspecification (test where the ANM assumption breaks):**
- **TCD-Arena** / **CausalCompass** (under review, ICLR 2026) — assess time-series CD under assumption violations. Run these to show ORACLE's failure modes honestly (both residual tests growing = misspecification, which ORACLE surfaces by leaving edges undirected rather than guessing).

**No causal ground truth (qualitative case studies ONLY — do not report SHD/F1):**
- S&P 500 minute ETFs (illustrate detection of a known regime break, e.g. early 2020) and MIMIC-III vitals (shift around interventions). State explicitly these are case studies, not benchmarks. Compare shift-detection timing against CD-NOTS / SpaceTime here.

---

## 7. Metrics

### 7.1 Validity audit (run this FIRST — it falsifies the original design)
- **Null stream:** generate `p` mutually independent series (no edges). Plot empirical FWER/FDR as a function of `t` over many seeds. **Requirement:** the curve stays ≤ α at *all* `t` simultaneously. A method with the original `exp(HSIC·λ)` e-value fails here immediately (wealth diverges). This is the single most important experiment.
- **Calibration:** distribution of rejection times under H₀ should match the Ville bound.

### 7.2 Graph recovery (ground-truthed data)
- **SHD** (structural Hamming distance) and **SID** (structural intervention distance, Peters & Bühlmann) — SID is the causally meaningful one and was missing from the original.
- **Precision / Recall / F1**, reported **separately** for (a) skeleton adjacency and (b) edge orientation.
- Report **both** pre-projection and post-projection graphs (projection can delete true edges).

### 7.3 Anytime behaviour (the differentiator)
- **FWER(t) / FDR(t) curves** evaluated at every `t` simultaneously, with the α line overlaid.
- **Sample-to-detection:** how many observations until each true edge is correctly declared (efficiency of the bet).

### 7.4 Change detection
- **Expected detection delay** `E[τ̂ − t*]` vs. **average run length to false alarm (ARL)** — sweep `h` to trace the delay/ARL curve and compare against baselines.

### 7.5 Systems
- Throughput (obs/sec) and memory vs. `p` and budget `m`; verify the real-time claim empirically.

### 7.6 Ablations
- α ∈ {0.01, 0.05, 0.10}; degree cap `k ∈ {0,1,2,3}` (shows the pairwise-vs-DAG gap); bet sizing (fixed λ vs. aGRAPA vs. mixture); kernel (RBF vs. linear); RFF features `D`; with/without CUSUM.

---

## 8. Implementation roadmap

1. **Week 1–2:** Implement and unit-test `SKIT(U,V)` against the reference repo on a Gaussian linear model; confirm type-I control on a null stream (validity audit). *Do not proceed until this passes.*
2. **Week 3:** Add `SKIT_CI` via residual reduction; verify on a 3-node chain that conditioning removes the spurious edge.
3. **Week 4:** ANM orientation on 2-node and 3-node graphs; unit-test the **sign convention** explicitly.
4. **Week 5–6:** Full pipeline + e-BH + DAG projection; synthetic benchmark vs. batch baselines.
5. **Week 7:** Page-CUSUM + piecewise-stationary experiments (delay/ARL).
6. **Week 8:** DREAM/Sachs; scalability; qualitative financial/ICU case studies.

---

## 9. Corrected key references

- Podkopaev, Blöbaum, Kasiviswanathan, Ramdas (2023). *Sequential Kernelized Independence Testing.* ICML. — **the valid HSIC betting wealth process** (code available).
- Shaer, Maman, Romano (2023). *Model-free sequential testing for conditional independence via testing by betting.* AISTATS.
- Shekhar & Ramdas (2023). *Nonparametric two-sample testing by betting.* IEEE Trans. Inf. Theory.
- Peters, Mooij, Janzing, Schölkopf (2014). *Causal discovery with continuous additive noise models* (RESIT). JMLR.
- Gretton, Bousquet, Smola, Schölkopf (2005). *Measuring statistical dependence with Hilbert–Schmidt norms* (HSIC origin). ALT. — (the 2012 JMLR paper is the *two-sample* test; don't conflate.)
- Wang & Ramdas (2022). *False discovery rate control with e-values.* JRSS-B (e-BH).
- Ramdas, Grünwald, Vovk, Shafer (2023). *Game-theoretic statistics and safe anytime-valid inference.* Statistical Science.
- Brouillard, Lachapelle, Lacoste, Lacoste-Julien, Drouin (2020). *Differentiable Causal Discovery from Interventional Data* (**DCDI**). NeurIPS. — (DECI = Geffner et al. 2022, a different paper.)
- Peters & Bühlmann (2015). *Structural intervention distance (SID).* Neural Computation.
- Lorden (1971). *Procedures for reacting to a change in distribution.* Ann. Math. Stat. (CUSUM optimality lower bound).
- Ville (1939). *Étude critique de la notion de collectif.* (anytime validity).

**2024–2026 baselines, benchmarks, and cautions:**
- Lorch et al. (2022). *Amortized Inference for Causal Structure Learning* (AVICI). NeurIPS. — pretrained model + code.
- Ke et al. (2023). *CSIvA: transformer for amortized causal discovery.*
- *Sample, Estimate, Aggregate (SEA): a recipe for causal discovery foundation models* (2024). arXiv 2402.01929.
- Balazadeh et al. (2025). *CausalPFN*; Robertson et al. (2025). *Do-PFN.* — in-context (PFN) causal models.
- Swelam et al. (2025). *Causal discovery with frozen TabPFN adapters.*
- Montagna, Cairney-Leeming, Sridhar, Locatello (2024/2026). *Demystifying amortized causal discovery with transformers.* TMLR. — identifiability limits of amortized methods (cite as caveat).
- Huang, Zhang, Zhang, Ramsey, Sanchez-Romero, Glymour, Schölkopf (2020). *Causal discovery from heterogeneous/nonstationary data* (CD-NOD). JMLR.
- Sadeghi, Gopal, Fesanghary (2025). *Causal discovery from nonstationary time series* (CD-NOTS). Int. J. Data Sci. Anal.
- Mameche et al. (2025). *SpaceTime: Causal Discovery from Non-Stationary Time Series.* arXiv 2501.10235. — closest shift-detection competitor.
- Balsells-Rodas et al. (2025). *Causal Discovery from Conditionally Stationary Time Series.* ICML.
- Gao, Addanki, Yu, Rossi, Kocaoglu (2024/2025). *Causal Discovery-Driven Change Point Detection in Time Series.* arXiv 2407.07290.
- Huang, Peters, Pfister (2024). *Causal change point detection and localization.* arXiv 2403.12677.
- Reisach, Seiler, Weichwald (2021). *Beware of the simulated DAG!* NeurIPS. — varsortability caveat for synthetic benchmarks.
- Benchmarks: **CausalRivers** (ICLR 2025), **CausalDynamics** (NeurIPS 2025), **CausalTime** (ICLR 2024), **causalAssembly** (CLeaR 2024), **OCDB** (2024). Tooling: **causal-learn**, **gCastle**.

---

*ORACLE spec — corrected redesign for implementation. The original ORACLE's research framing is sound; its statistical core needed replacement, not tuning.*