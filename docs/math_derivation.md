# Mathematical Derivation — Regime-Switching Optimal Market Making

This document derives everything from first principles: state dynamics, the value function,
the HJB PDE, the optimal spread formulas, and the Wonham HMM filter.

---

## 1. Setup and State Variables

We work on a filtered probability space (Ω, F, {F_t}, P).

**State at time t:**
- `x_t` — cash (wealth from completed trades)
- `q_t` — inventory (shares held; can be negative)
- `S_t` — mid-price of the asset
- `k_t ∈ {1, 2}` — current volatility regime (hidden in Phase 4)

**Total wealth:** `W_t = x_t + q_t · S_t`

---

## 2. Mid-Price Dynamics

The mid-price follows a regime-switching arithmetic Brownian motion:

```
dS_t = σ_{k_t} dW_t
```

where:
- `W_t` is a standard Brownian motion
- `σ_k` is the volatility in regime k (σ₁ < σ₂)
- No drift: at the intraday timescale (seconds to minutes) drift is negligible vs volatility

---

## 3. Regime Dynamics

`k_t` is a continuous-time Markov chain on {1, 2} with generator matrix:

```
Q = [ -q₁₂   q₁₂ ]
    [  q₂₁  -q₂₁ ]
```

Transition probabilities over dt:
```
P(k_{t+dt} = 2 | k_t = 1) = q₁₂ · dt + o(dt)
P(k_{t+dt} = 1 | k_t = 2) = q₂₁ · dt + o(dt)
```

Stationary distribution:
```
π₁* = q₂₁ / (q₁₂ + q₂₁)
π₂* = q₁₂ / (q₁₂ + q₂₁)
```

---

## 4. Order Flow Model

The market maker posts bid at `S_t − δᵇ` and ask at `S_t + δᵃ`.

Market orders arrive as regime-dependent Poisson processes:
```
Ask hit rate: λᵃ_k(δᵃ) = A_k · exp(−κ · δᵃ)
Bid hit rate: λᵇ_k(δᵇ) = A_k · exp(−κ · δᵇ)
```

When ask is hit: `x → x + (S_t + δᵃ)`,  `q → q − 1`
When bid is hit: `x → x − (S_t − δᵇ)`,  `q → q + 1`

---

## 5. Objective Function

The market maker maximises expected terminal wealth minus a terminal inventory penalty:

```
J(x, q, S, t, k) = E[ x_T + q_T · S_T − (φ/2) · q_T²  |  x_t=x, q_t=q, S_t=S, k_t=k ]
```

The term `(φ/2)q_T²` penalises holding a large position at end of day.

**Note on two risk-aversion parameters:** The objective has terminal penalty φ, but
the ansatz in Section 9 introduces a separate running-inventory term γ. These are
not redundant — they serve different purposes:

- **φ (terminal penalty)**: appears in the boundary condition h(q,T) = −(φ/2)q².
  It penalises inventory *held at the end of the session* and is part of the
  formal optimisation objective.

- **γ (running risk aversion)**: appears in the ansatz V = x + qS − ½γσₖ²q²τ + h.
  This term accounts for the *ongoing mark-to-market risk* of holding inventory q
  in a market with volatility σₖ. It is an Avellaneda-Stoikov approximation that
  separates the inventory risk from the trading optimisation, enabling the PDE
  reduction. It does not appear in J directly.

In the single-regime unconstrained limit, φ and γ can be made consistent by
choosing γ = φ/(σ²T). In the regime-switching case with coupling, they are
independent parameters: φ governs end-of-day liquidation risk, γ governs
intraday spread widening. An interviewer asking "why both?" should receive
this explanation.

---

## 6. Value Function

Define the value function as the supremum of J over all admissible controls:

```
Vᵏ(x, q, S, t) = sup_{δᵃ, δᵇ} J(x, q, S, t, k)
```

**Terminal condition:**
```
Vᵏ(x, q, S, T) = x + qS − (φ/2)q²    for k ∈ {1, 2}
```

---

## 7. Deriving the Coupled HJB System

Apply Itô's lemma to Vᵏ(x_t, q_t, S_t, t) over [t, t+dt].

**From the Brownian motion of S:**
```
dVᵏ|_diffusion = (∂Vᵏ/∂t) dt + (∂Vᵏ/∂S) dS_t + ½(∂²Vᵏ/∂S²)(dS_t)²
              = (∂Vᵏ/∂t) dt + σₖ(∂Vᵏ/∂S) dW_t + ½σₖ²(∂²Vᵏ/∂S²) dt
```

**From an ask being hit (Poisson jump, rate λᵃ):**
```
Jump contribution = λᵃ_k(δᵃ) · [Vᵏ(x + S + δᵃ, q−1, S, t) − Vᵏ(x, q, S, t)] dt
```

**From a bid being hit (Poisson jump, rate λᵇ):**
```
Jump contribution = λᵇ_k(δᵇ) · [Vᵏ(x − S + δᵇ, q+1, S, t) − Vᵏ(x, q, S, t)] dt
```

**From the regime switching (Markov chain jump):**
```
Regime contribution = Σⱼ≠ₖ qₖⱼ · [Vʲ(x, q, S, t) − Vᵏ(x, q, S, t)] dt
```

By the principle of optimality (Bellman), the total expected change in V must be zero
under the optimal policy. Setting the drift of V to zero:

```
∂Vᵏ/∂t + ½σₖ² ∂²Vᵏ/∂S²
  + max_{δᵃ} { λᵏᵃ(δᵃ) · [Vᵏ(x+S+δᵃ, q−1, S, t) − Vᵏ] }
  + max_{δᵇ} { λᵏᵇ(δᵇ) · [Vᵏ(x−S+δᵇ, q+1, S, t) − Vᵏ] }
  + Σⱼ≠ₖ qₖⱼ · (Vʲ − Vᵏ)  =  0
```

This is a **system of two coupled PDEs** — coupled via the last line.

---

## 8. Solving the Ask Optimisation (First-Order Condition)

For the ask term, maximise over δᵃ:
```
f(δᵃ) = A_k · exp(−κδᵃ) · [Vᵏ(x+S+δᵃ, q−1, S, t) − Vᵏ(x, q, S, t)]
```

After substituting the ansatz Vᵏ = x + qS − ½γσₖ²q²τ + hᵏ(q,t), the value
difference computes exactly (all x, S terms cancel cleanly):

```
ΔVᵃ = [x+(S+δᵃ) + (q−1)S − ½γσₖ²(q−1)²τ + hᵏ(q−1,t)]
     − [x + qS   −  ½γσₖ²q²τ              + hᵏ(q,t)  ]
    = δᵃ  +  hᵏ(q−1,t) − hᵏ(q,t)  −  γσₖ²qτ + ½γσₖ²τ
    ≡ δᵃ  +  C(q, t, k)
```

where C(q,t,k) = hᵏ(q−1,t) − hᵏ(q,t) − γσₖ²qτ + ½γσₖ²τ collects everything
independent of δᵃ. So:

```
f(δᵃ) = A_k · e^{−κδᵃ} · (δᵃ + C)
```

Taking df/dδᵃ = 0:
```
−κ · A_k · e^{−κδᵃ} · (δᵃ + C)  +  A_k · e^{−κδᵃ} · 1  =  0
⟹  −κ(δᵃ + C) + 1  =  0
⟹  δᵃ*  =  1/κ − C
          =  1/κ − hᵏ(q−1,t) + hᵏ(q,t) + γσₖ²qτ − ½γσₖ²τ
```

Second-order condition: d²f/d(δᵃ)² = −κ²(δᵃ+C)e^{−κδᵃ}A_k < 0 when δᵃ* > 0,
confirming this is a maximum. The same derivation gives δᵇ* symmetrically.

---

## 9. Change of Variables (Ansatz)

Assume:
```
Vᵏ(x, q, S, t) = x + q·S − γ·q²·σₖ²·(T−t)/2 + hᵏ(q, t)
```

The `x + qS` part is current wealth. The second term is the inventory penalty accumulated
from now to T (assuming regime k persists — the coupling corrects for regime changes).
`hᵏ(q,t)` captures residual value from optimal trading.

Substituting into the HJB and simplifying (the S terms cancel by construction):

```
∂hᵏ/∂t − γσₖ²q²/2
  + max_{δᵃ} { A_k·e^{−κδᵃ} · [hᵏ(q−1,t) − hᵏ(q,t) + δᵃ − γσₖ²q(T−t)] }
  + max_{δᵇ} { A_k·e^{−κδᵇ} · [hᵏ(q+1,t) − hᵏ(q,t) + δᵇ + γσₖ²q(T−t)] }
  + Σⱼ≠ₖ qₖⱼ·(hʲ − hᵏ)  =  0
```

---

## 10. Optimal Spreads (Closed Form Approximation)

Taking first-order conditions on δᵃ and δᵇ:

```
δᵏ·ᵃ* = (1/κ) + [hᵏ(q,t) − hᵏ(q−1,t)] + γσₖ²q(T−t)
δᵏ·ᵇ* = (1/κ) − [hᵏ(q,t) − hᵏ(q+1,t)] − γσₖ²q(T−t)
```

For the unconstrained single-regime case (no inventory limits, no coupling),
and approximating hᵏ(q,t) ≈ hᵏ(q,t)|_{symmetric} so that h(q−1)−h(q) ≈ −γσ²qτ,
this reduces to the **Avellaneda-Stoikov approximation**:

```
δ* ≈ 1/κ  +  γσ²(T−t)/2        [A-S approximation, not exact for coupled system]
```

**Important:** once regime switching and coupling terms are present, this formula
no longer follows directly from the HJB. The exact optimal spread requires the full
numerical solution for hᵏ(q,t). The A-S formula serves as an initialisation and
sanity check, but the simulator uses the numerically computed spreads.

The **reservation price** (A-S approximation):
```
r ≈ S − q · γ · σ² · (T−t)      [A-S approximation]
```

Total approximate spread = `2/κ + γσ²(T−t)` per regime.

---

## 11. Wonham HMM Filter

In practice, `k_t` is not observed. We maintain the posterior belief:
```
π_t = P(k_t = 2 | F_t^Y)
```
where `F_t^Y` is the filtration generated by observed price moves and order arrivals.

**Innovation process:** The "surprise" in observed price moves given current belief:
```
σ(π_t) = π_t·σ₂ + (1−π_t)·σ₁   (blended volatility under current belief)
dI_t = dS_t / σ(π_t) − dW_t^P   (standardised innovation)
```

**Wonham filter SDE:**
```
dπ_t = [q₁₂(1−π_t) − q₂₁π_t] dt
       + π_t(1−π_t) · [(σ₂−σ₁)/σ(π_t)] · dI_t
```

**First term:** deterministic drift from regime dynamics (mean-reverting toward stationary π*)
**Second term:** stochastic Bayesian update — large price moves (big dI_t) push π_t toward 1

**Blended optimal spread:**
```
δ_blended* = (1−π_t)·δ¹* + π_t·δ²*
```

---

## 12. Numerical Solution Plan

Since no clean closed form exists for the coupled constrained system, we solve numerically.

**Important structural note:** After the ansatz reduction, the ½σₖ²∂²V/∂S² diffusion
term has been completely eliminated from the PDE — it lives in S-space, and S drops out
of the reduced equation entirely. What remains is a coupled system of ODEs in time on
a discrete inventory grid, with jump terms between neighbouring inventory states q−1, q,
q+1. There is no diffusion operator in inventory. Crank-Nicolson is used only for the
time integration, not as a spatial discretisation of any diffusion.

The reduced system at each inventory node q is:

```
dhᵏ/dt = jump_a(q, hᵏ, hᵏ, k, τ)
        + jump_b(q, hᵏ, hᵏ, k, τ)
        + Σⱼ≠ₖ qₖⱼ·(hʲ(q,t) − hᵏ(q,t))
```

This is an ODE system (no spatial derivative in q), coupled across k ∈ {1,2}.

**Numerical procedure:**

1. **Discretise** the inventory grid: q ∈ {−Q, ..., 0, ..., Q}, n=41 points
2. **Time-step backward** from T to 0 using trapezoidal (Crank-Nicolson) rule
   for the time derivative only:
   ```
   hᵏ(q, tₙ) = hᵏ(q, tₙ₊₁) + ½·dt·[RHS(tₙ₊₁) + RHS(tₙ)]
   ```
3. **At each (q, t) node**, solve the max over δᵃ and δᵇ analytically via FOC
4. **Simultaneously update** h¹ and h² at each step (coupling requires joint solve)
5. **Store** the optimal δᵃ*(q,t,k) and δᵇ*(q,t,k) lookup tables

The Crank-Nicolson label refers only to the second-order accurate trapezoidal
time-stepping scheme. There is no finite-difference diffusion operator anywhere
in the reduced problem.
