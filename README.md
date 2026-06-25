# ORACLE — Online Anytime-Valid Causal Discovery

Implementation of the corrected ORACLE design specified in
[`algorithm.md`](algorithm.md): streaming causal discovery from a
(piecewise-)stationary SCM with **finite-sample error control at every stopping
time** (no correction for repeated looks), built on a valid sequential test
(SKIT betting wealth), ANM residual-asymmetry orientation, e-BH multiplicity
control, DAG projection, and Page-CUSUM change detection.

## What this fixes vs. the original draft

| Original draft | Problem | This implementation |
|---|---|---|
| `e = exp(HSIC·λ)` (always ≥ 1) | not a valid e-value; wealth always diverges | **SKIT betting wealth** — a nonnegative (super)martingale under H₀ via an exchangeability-based swap construction |
| orient toward *higher* dependence | sign-inverted vs. ANM | orient by **rejecting the anti-causal direction** (residual stays dependent) |
| regress `Xⱼ` on a single `Xᵢ` | recovers a dependency graph, not a DAG | **condition on the estimated parent/neighbour set** (RESIT reduction) |
| λ / bandwidth recomputed on current point | breaks prequential validity | λ, bandwidth, witness, regressors are all **previsible** (fit on data `< t`) |
| Bonferroni over `p(p−1)` | power collapse | **e-BH** (Wang–Ramdas), e-Bonferroni optional |
| `Sₜ = Lₜ − min Lₛ` | degenerate | **Page-CUSUM** on log e-increments |

## Layout

```
oracle/
  skit.py        SKIT independence-by-betting wealth + Page-CUSUM
  online_reg.py  online RFF-ridge regressor (RLS, previsible residuals)
  skit_ci.py     conditional independence by RESIT residual reduction
  ebh.py         e-BH and e-Bonferroni multiplicity control
  discovery.py   the ORACLE algorithm (skeleton -> orient -> e-BH -> DAG -> CUSUM)
  metrics.py     SHD, SID (parent-adjustment), skeleton/orientation P/R/F1
  data.py        random DAGs, non-Gaussian ANM, piecewise-stationary, null streams
baselines/
  naive_stopping.py  optional-stopping straw man (HSIC p-value, repeated looks)
  batch.py           PC-stable / GES via causal-learn (checkpoint harness)
experiments/         the six evaluation experiments + run_all
tests/               critical unit tests (run these first)
```

## Statistical core: why the SKIT wealth is valid

We process the paired stream two observations at a time. Given two pairs
`(uₐ,vₐ)`, `(u_b,v_b)`, the **joint** points `{(uₐ,vₐ),(u_b,v_b)}` and the
**product** (swapped) points `{(uₐ,v_b),(u_b,vₐ)}` are exchangeable under
H₀: U ⊥ V. The bet payoff

```
g = ½ ( mean_P f_t − mean_Q f_t ),   f_t = previsible standardised witness ∈ [−1,1]
```

has `E[g | F_{t−1}] = 0` under H₀ for **any** previsible witness `f_t`, so
`K_t = K_{t−1}(1 + λ g_t)` is a nonnegative martingale and `K_t ≥ 1/α` is an
anytime-valid rejection by Ville's inequality. Validity does not depend on
witness quality — only power does. The witness is standardised by its running
scale (estimated from past evaluations only) to turn the small HSIC signal into
an order-1 payoff.

## Setup & running

```bash
conda activate py313          # numpy, scipy, scikit-learn, networkx, torch, causal-learn
export PYTHONPATH=.           # (Windows Git-Bash: PYTHONPATH=. conda run -n py313 ...)

# critical unit tests first (sec 8 of the spec: do not proceed until green)
python -m pytest tests/test_core.py -v

# full evaluation suite -> results/RESULTS.md + results/figures/*.png
python -m experiments.run_all          # add --quick for a fast smoke config
```

## Guarantees targeted (and tested)

- **Anytime type-I / FDR control** — per-edge by Ville on a valid martingale;
  combined by e-BH / e-Bonferroni. Falsified-or-confirmed by the validity audit
  (sec 7.1): SKIT stays ≤ α at all t, naive optional stopping inflates.
- **Sequential consistency** — `Ĝ_t → G*` when conditioning on the parent set.
- **Shift detection** — Page-CUSUM delay vs ARL trade-off (sec 7.4).

See `results/RESULTS.md` (generated) for the measured numbers and figures.
