# Regime-Switching Optimal Market Making Engine

A stochastic control framework for inventory-aware market making under
regime-switching volatility, with explicit adverse selection modelling
and online Bayesian regime inference.

---

## What this project is (and is not)

**What it is:**
- A **reduced-form inventory control model** under a 2-state Markov chain volatility process
- A **coupled HJB solver**: after the ansatz reduction the S-diffusion vanishes and the
  problem becomes a coupled *ODE system* in time, indexed by discrete inventory and
  regime, integrated backward via implicit/trapezoidal time-stepping
- A **stochastic simulation** of Poisson order arrivals with regime-dependent rates and adverse selection
- An **online Wonham HMM filter** that infers the latent regime from observed price innovations

**What it is not:**
- A full limit order book simulator (no queue model, no depth levels, no cancellations)
- A 4D HJB solver — the substitution `V(x,q,S,t) = x + qS + h(q,t)` eliminates S, leaving a 2D problem per regime

The correct description is: *inventory-aware market making simulation under regime-switching volatility*.

---

## Results (5000 simulations, with adverse selection)

| Agent | Sharpe | CVaR 5% | PnL σ | Max Inventory |
|---|---|---|---|---|
| **Regime-Switching** | 0.2230 | −0.107 | **2.00** | **1.61** |
| Naive Constant-Vol | 0.2461 | −0.161 | 2.82 | 1.96 |
| Symmetric Fixed | 0.2115 | −0.309 | 3.80 | 2.84 |

**Regime-Switching vs Naive:**

| Metric | Improvement |
|---|---|
| CVaR 5% | **+33.5%** (less left-tail risk) |
| PnL std deviation | **+29.1%** (tighter outcome distribution) |
| Max inventory held | **+17.8%** (better adverse selection avoidance) |
| Sharpe ratio | −9.4% (intentional: regime agent is more conservative in chaos) |

The Sharpe difference is expected and economically correct. The regime-switching
agent trades less aggressively during chaotic periods (wider spreads = fewer fills)
to avoid informed order flow. The payoff is a 29.1% reduction in PnL variance and
33.5% better tail risk — the metrics that matter for a risk-managed trading desk.

### Confidence intervals (5000 simulations, 95% CI = mean ± 1.96 × SE)

| Agent | Sharpe | ±95% CI | CVaR 5% | ±95% CI | PnL σ |
|---|---|---|---|---|---|
| **Regime-Switching** | 0.2230 | ±0.0027 | −0.107 | ±0.0025 | 2.00 |
| Naive Constant-Vol | 0.2461 | ±0.0031 | −0.161 | ±0.0045 | 2.82 |
| Symmetric Fixed | 0.2115 | ±0.0034 | −0.309 | ±0.0061 | 3.80 |

### Ablation study — contribution of each component

Each row adds one component to the previous. 5000 simulations per row.

| Model | Sharpe | ±SE | CVaR 5% | ±SE | PnL σ |
|---|---|---|---|---|---|
| Base A-S (constant vol, no adv. selection) | 0.2567 | ±0.0016 | −0.155 | ±0.0023 | 3.05 |
| + Regime-switching spreads | 0.2223 | ±0.0014 | −0.105 | ±0.0013 | 2.11 |
| + Adverse selection model | 0.2195 | ±0.0014 | −0.105 | ±0.0013 | 1.98 |
| + Bayesian filter (hidden regime) | 0.2260 | ±0.0014 | −0.109 | ±0.0014 | 2.07 |

**Reading the table:**
- Regime switching alone drives most of the CVaR improvement (−0.155 → −0.105, **32% better**).
- Adding adverse selection tightens the PnL distribution further (σ: 2.11 → 1.98) without hurting Sharpe materially.
- The Bayesian filter (which infers the regime from price increments rather than observing it directly)
  achieves CVaR within 4% of the oracle regime-switching agent and actually has the *highest* Sharpe of
  the risk-aware models — its blended spreads are slightly less aggressive than the oracle's hard regime
  switch, smoothing transitions at regime boundaries.
- Sharpe drops from the base model to the regime-aware models because each makes the agent
  *more conservative* — intentionally trading less in dangerous conditions. The risk reduction
  (CVaR, PnL variance) is the goal, not raw return.

*Note: arrival probabilities use the exact form `1 − e^{−λΔt}` rather than the small-Δt
approximation `λΔt`, which matters at the high arrival rates (A₂=50/min) used in the
chaotic regime — see `docs/model_limitations.md`.*

---

## Adverse Selection Model

The key addition over standard A-S: in the chaotic regime, 35% of market orders
are "informed" — after the trade, the price moves adversely by `α × spread`.
The naive agent ignores this and gets picked off; the regime-switching agent widens
spreads in chaotic periods to compensate.

```
P(informed | calm)    = 5%   → small price impact
P(informed | chaotic) = 35%  → significant adverse selection
Price move after informed trade: ΔS = α · δ*  (α = 0.30)
```

---

## Research Figures

### Figure 1 — Hidden Regime vs Inferred Regime
![Fig1](results/plots/fig1_regime_inference.png)
*Discrete-time Bayesian regime filter (see naming note below) achieves 95–99%
regime classification accuracy across 20 independent 2-hour simulated paths
(7200 one-second steps each) at σ₂/σ₁ ratios of 1.2×–8×, with transition rates
q₁₂=1/300 s⁻¹ (avg 5 min calm) and q₂₁=1/120 s⁻¹ (avg 2 min chaotic). Accuracy
increases monotonically with volatility separation — full sensitivity table below.*

**Naming note:** This is *not* a classical Wonham (1964) filter. The Wonham
filter applies to hidden Markov chains observed through a **drift** signal
(dY = h(k)dt + dW). Our model has the hidden state modulating **volatility**
(dS = σ(k)dW) with zero drift under both regimes — a fundamentally different
filtering problem where drift-based innovations carry no information. We
implement the exact discrete-time Bayesian predict-update (Hamilton 1989
type), which converges to the true continuous-time filter as dt→0. See
`docs/math_derivation.md` §11 and the docstring in
`hmm_filter/src/wonham_filter.py` (class `BayesVolFilter`) for the full
derivation. The `WonhamFilter` name is kept only as a backward-compatible alias.

### Figure 2 — Posterior Belief π_t
![Fig2](results/plots/fig2_posterior_belief.png)
*π_t = P(chaotic | observations). Agent widens spreads when π_t > 0.5.*

### Filter Sensitivity (classification accuracy vs volatility separation)

| σ₁ | σ₂ | σ₂/σ₁ | Accuracy | Paths | Path length |
|---|---|---|---|---|---|
| 0.005 | 0.006 | 1.2× | 81.2% ± 2.9% | 20 | 7200 s |
| 0.005 | 0.0075 | 1.5× | 91.3% ± 1.4% | 20 | 7200 s |
| 0.005 | 0.010 | 2× | 95.7% ± 0.9% | 20 | 7200 s |
| 0.005 | 0.0125 | 2.5× | 97.1% ± 0.6% | 20 | 7200 s |
| 0.005 | 0.015 | 3× | 97.8% ± 0.4% | 20 | 7200 s |
| 0.005 | 0.020 | 4× *(paper setting)* | 98.4% ± 0.3% | 20 | 7200 s |
| 0.005 | 0.025 | 5× | 98.8% ± 0.2% | 20 | 7200 s |
| 0.005 | 0.030 | 6× | 98.9% ± 0.2% | 20 | 7200 s |
| 0.005 | 0.040 | 8× | 99.2% ± 0.2% | 20 | 7200 s |

Transition rates fixed: q₁₂ = 1/300 s⁻¹ (avg 5 min calm), q₂₁ = 1/120 s⁻¹ (avg
2 min chaotic). Accuracy = fraction of 1-second steps where MAP regime (π ≥ 0.5)
matches true regime, mean ± std over 20 paths. With the corrected discrete-time
Bayesian filter (which includes the Gaussian normalising constants — an earlier
version omitted these and was biased), accuracy increases **monotonically** with
σ₂/σ₁, as expected from identifiability theory: more separated regimes are
strictly easier to distinguish.


### Figure 3 — PnL Distributions
![Fig3](results/plots/fig3_pnl_distributions.png)
*Regime-switching agent has 32.9% lower PnL variance and 38.7% better CVaR.*

### Figure 4 — Inventory Trajectories
![Fig4](results/plots/fig4_inventory_trajectories.png)
*Regime agent holds 19.7% less peak inventory — avoids toxic order flow.*

### Figure 5 — Full Risk-Adjusted Comparison
![Fig5](results/plots/fig5_sharpe_comparison.png)
*CVaR, PnL std, regime-decomposed PnL, and per-simulation ΔSharpe distribution.*

---

## Benchmarks

| Component | Performance |
|---|---|
| HJB ODE-system solver | 82,000 nodes in **0.10 ± 0.002 s** | 3 timed runs, 1000t × 41q × 2 regimes |
| Wonham HMM filter | **224k belief updates/sec** | 5 timed runs on 100k-step paths, pure Python |
| Full simulation step | **~410k steps/sec** | Spread calc + Poisson draw + PnL update, pure Python |
| Full backtest | 5000 sims × 60 steps × 3 agents in **~45 s** | Single-threaded Python |

> The inner simulation loop runs in Python (~410k steps/sec). A C++ port of this loop would reach 10–50M steps/sec; the skeleton in `simulator/` is the starting point for that.

---

## Key Math

### Reduced HJB (after substitution V = x + qS + h)

```
∂h^k/∂t
  + max_{δᵃ} { A_k e^{−κδᵃ} [h^k(q−1) − h^k(q) + δᵃ − γσ_k²q·τ] }
  + max_{δᵇ} { A_k e^{−κδᵇ} [h^k(q+1) − h^k(q) + δᵇ + γσ_k²q·τ] }
  + Σⱼ≠ₖ q_{kj}·(h^j − h^k) = 0

Terminal: h^k(q,T) = −(φ/2)q²
```

### Wonham Filter (chi-squared innovation)

```
dπ_t = [q₁₂(1−π) − q₂₁π] dt
       + π(1−π) · (σ₂²−σ₁²)/(2σ(π)²) · (dS²/σ(π)²dt − 1) dt

σ(π) = π·σ₂ + (1−π)·σ₁  (blended volatility)
```

Innovation `(dS²/σ²dt − 1)` is positive when observed variance exceeds expectation
under current belief → π pushed toward 1 (chaotic). Negative → π pulled toward 0.

---

## Quick Start

```bash
pip install -r requirements.txt
python pde_solver/src/hjb_solver.py   # solve HJB, save spread tables (~0.1s)
python generate_figures.py            # full backtest + all 5 figures (~45s)
```

---

## Parameters (`configs/default.yaml`)

| Parameter | Default | Meaning |
|---|---|---|
| σ₁ / σ₂ | 0.5 / 3.0 | Calm / chaotic regime volatility |
| q₁₂ / q₂₁ | 2.0 / 8.0 | Regime transition rates |
| γ | 0.3 | Risk aversion |
| κ | 2.0 | Order flow sensitivity to spread |
| A₁ / A₂ | 5 / 50 per min | Baseline arrival rates |
| α | 0.30 | Adverse selection price impact |
| P(informed\|chaos) | 35% | Fraction of informed orders in chaotic regime |

---

## References

1. Avellaneda & Stoikov (2008) — *High-frequency trading in a limit order book*
2. Cartea, Jaimungal & Penalva (2015) — *Algorithmic and High-Frequency Trading*
3. Wonham (1964) — *Some applications of SDEs to optimal nonlinear filtering*
4. Glosten & Milgrom (1985) — *Bid, ask and transaction prices* (adverse selection model)

Full annotated references in `docs/references.md`.
