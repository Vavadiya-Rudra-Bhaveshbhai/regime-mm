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

After substituting the ansatz Vᵏ = x + qS − ½γσₖ²q²τ + hᵏ(q,t), and using
q² − (q−1)² = 2q−1, the value difference (all x, S terms cancel) is:

```
ΔVᵃ = [x+(S+δᵃ) + (q−1)S − ½γσₖ²(q−1)²τ + hᵏ(q−1,t)]
     − [x + qS   −  ½γσₖ²q²τ              + hᵏ(q,t)  ]
    = δᵃ  +  [hᵏ(q−1,t) − hᵏ(q,t)]  +  γσₖ²qτ − ½γσₖ²τ
    ≡ δᵃ  +  D(q, t, k)
```

where D(q,t,k) = [hᵏ(q−1,t) − hᵏ(q,t)] + γσₖ²qτ − ½γσₖ²τ collects everything
independent of δᵃ. So f(δᵃ) = A_k·e^{−κδᵃ}·(δᵃ + D). Taking df/dδᵃ = 0:

```
⟹  −κ(δᵃ + D) + 1  =  0
⟹  δᵃ*  =  1/κ − D
          =  1/κ + [hᵏ(q,t) − hᵏ(q−1,t)] − γσₖ²qτ + ½γσₖ²τ
```

Second-order condition: d²f/d(δᵃ)² = −κ²(δᵃ+D)e^{−κδᵃ}A_k < 0 when δᵃ* > 0,
confirming this is a maximum.

**Bid side**, by the symmetric expansion (q+1)² − q² = 2q+1:
```
ΔVᵇ = [x+(−S+δᵇ) + (q+1)S − ½γσₖ²(q+1)²τ + hᵏ(q+1,t)]
     − [x + qS    −  ½γσₖ²q²τ              + hᵏ(q,t)  ]
    = δᵇ + [hᵏ(q+1,t) − hᵏ(q,t)] − γσₖ²qτ − ½γσₖ²τ ≡ δᵇ + E(q,t,k)

⟹  δᵇ*  =  1/κ − E
          =  1/κ + [hᵏ(q,t) − hᵏ(q+1,t)] + γσₖ²qτ + ½γσₖ²τ
```

**Implementation note:** D and E above are exactly `dV_a` and `dV_b` in
`pde_solver/src/hjb_solver.py`, and `δ* = 1/κ − dV` (clipped to a minimum
tick size). *(An earlier draft of this section had the signs on the
γσₖ²qτ and ½γσₖ²τ terms inverted relative to the code; the forms above are
the corrected, code-consistent versions.)*

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
  + max_{δᵃ} { A_k·e^{−κδᵃ} · [hᵏ(q−1,t) − hᵏ(q,t) + δᵃ + γσₖ²qτ − ½γσₖ²τ] }   (= D, §8)
  + max_{δᵇ} { A_k·e^{−κδᵇ} · [hᵏ(q+1,t) − hᵏ(q,t) + δᵇ − γσₖ²qτ − ½γσₖ²τ] }   (= E, §8)
  + Σⱼ≠ₖ qₖⱼ·(hʲ − hᵏ)  =  0
```

---

## 10. Optimal Spreads (Closed Form Approximation)

Taking first-order conditions on δᵃ and δᵇ:

```
δᵏ·ᵃ* = (1/κ) + [hᵏ(q,t) − hᵏ(q−1,t)] − γσₖ²q(T−t) + ½γσₖ²(T−t)
δᵏ·ᵇ* = (1/κ) + [hᵏ(q,t) − hᵏ(q+1,t)] + γσₖ²q(T−t) + ½γσₖ²(T−t)
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

The **reservation price** r is the price at which the MM is indifferent
between buying and selling one more unit — i.e. r = S − ½(δᵇ*−δᵃ*) using
the exact spreads. In the A-S flat-h approximation:
```
r ≈ S − q · γ · σ² · (T−t)      [A-S approximation]
```

Total approximate spread = `2/κ + γσ²(T−t)` per regime.

**Validity:** Both the A-S spread formula and the reservation price are
approximations that hold exactly only in the unconstrained, single-regime,
flat-h limit. With regime switching, coupling, and inventory bounds, the
exact reservation price is implicitly defined by the value-function
differences h(q±1,t) − h(q,t) and must be read off the numerical solution.

---

## 11. Regime Filter — Discrete-Time Bayesian Update

**Naming clarification.** The classical Wonham (1964) filter applies to a
hidden Markov chain observed through a **drift** signal:
```
dY_t = h(k_t) dt + dW_t        [drift-signal model — classical Wonham]
```
Our model has the hidden state modulating **volatility**, not drift:
```
dS_t = σ_{k_t} dW_t            [volatility-switching model — our case]
E[dS_t | k_t] = 0 for both regimes
```
These are different filtering problems. A drift-based innovation
dI_t = dS_t/σ(π_t) − dW_t^P carries no information here because the
expected drift is zero under both regimes — there is no "surprise in the
mean" to exploit. The correct approach filters on the **quadratic
variation** (squared increments), which is where the volatility
information lives. We therefore implement an **exact discrete-time
Bayesian filter** rather than a continuous-time Wonham SDE. (An earlier
draft of this document used a Wonham-style SDE with a chi-squared
innovation as a heuristic approximation; that has been replaced.)

**Posterior belief:**
```
π_t = P(k_t = 2 | S_0, S_1, ..., S_t)
```

**Predict-update cycle** (exact for the discrete-time model, converges to
the true continuous-time filter as dt → 0):

*Prediction* (Markov chain propagation):
```
π_pred = π_{t-1} + [q₁₂(1−π_{t-1}) − q₂₁π_{t-1}] · dt
```

*Likelihood* (Gaussian observation model — note the normalising constants
are required and do NOT cancel since σ₁ ≠ σ₂):
```
L_k = (2πσₖ²dt)^{-1/2} · exp( −dS² / (2σₖ²dt) ),   k ∈ {1,2}
```

*Update* (Bayes' rule):
```
π_t = π_pred · L₂ / (π_pred · L₂ + (1−π_pred) · L₁)
```

This is implemented as `BayesVolFilter` in `hmm_filter/src/wonham_filter.py`
(the `WonhamFilter` name is retained as an alias for backward compatibility
only — see the docstring for the full naming discussion).

**Blended volatility** (variance-weighted, not linearly weighted — see
Section on filter properties below):
```
σ²(π_t) = π_t·σ₂² + (1−π_t)·σ₁²
```

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

---

## 13. Explicitly Stated Assumptions

This section collects modelling assumptions that are used throughout but
were not previously stated explicitly.

**Unit trade size.** Each fill is assumed to execute exactly one unit of
inventory (q → q±1 per arrival). Extending to variable trade sizes would
require a compound Poisson jump model where jump sizes are drawn from a
distribution, and the value-function differences h(q±n,t)−h(q,t) for
n>1 would need to be computed.

**Inventory boundary conditions.** At q = +Q (maximum long), the outward
sell intensity λᵇ is set to zero — the MM cannot increase inventory further.
At q = −Q (maximum short), the outward buy intensity λᵃ is set to zero.
In `hjb_solver.py` this is implemented by zeroing `dV_b[-1]` and `dV_a[0]`
(the boundary jump terms), which forces δᵇ*(Q,t)→∞ and δᵃ*(−Q,t)→∞ in the
spread formula, equivalent to refusing to quote on that side.

**Stationary distribution interpretation.** Statements like "fraction of
time chaotic ≈ 26%" refer to the **stationary probability** of the Markov
chain, q₁₂/(q₁₂+q₂₁). For any finite simulated session, the realised
fraction of time spent in the chaotic regime is itself a random variable
that converges to this stationary value only as session length → ∞.

**Calibration is synthetic, not from raw tick data.** The parameters in
`results/calibrated_params.json` are estimated from **synthetic** 1-minute
return series generated to match documented statistical properties of SPY
(Christoffersen 2011, Andersen et al. 2001, Ang & Bekaert 2002) — specifically
calm/chaotic volatility levels and average regime durations reported in the
literature. This is calibration to literature-reported moments, not
estimation from raw exchange tick data. See `docs/MODEL_LIMITATIONS.md`
Section 5 for what would be required to use real tick data.

**Symmetric order flow.** Ask and bid arrivals are modelled as independent
Poisson processes with the same functional form λ_k(δ) = A_k e^{−κδ} on
both sides. Real order flow exhibits buy/sell correlation (order flow
imbalance) that this model does not capture.

**Independence assumptions.** Conditional on the current regime k_t, price
increments dS_t and order arrivals are independent across time steps
(Markov property) and independent of each other within a step.

**Risk-aversion parameters φ and γ.** The terminal objective uses φ (the
formal terminal inventory penalty in J). The ansatz introduces a separate
running parameter γ (the per-unit-time inventory risk used to construct
h(q,t)). These are not redundant: φ enters only the boundary condition
h(q,T) = −(φ/2)q², while γ enters the running dynamics via the D and E
terms in Section 8. In the single-regime unconstrained limit they can be
made consistent via γ = φ/(σ²T); in the regime-switching case they are
independent and play different roles (end-of-day liquidation risk vs.
intraday spread widening).

---

## 14. Audit Response — Rigour Caveats

This section addresses high-severity issues raised by external review that
go beyond notation/labelling and concern genuine gaps in rigour. We state
each honestly rather than claim a fix that does not exist.

**(1) The discrete-time Bayes filter (Section 11) is not a Wonham filter,
and its continuous-time limit is NOT established here.** Section 11's
predict-update recursion is exact for the discrete-time observation model
S_i − S_{i-1} | k_i ~ N(0, σ_{k_i}²dt). It does **not** claim to be (or
converge to) a finite-dimensional continuous-time SDE filter. For
diffusion-coefficient switching, the rigorous continuous-time filter is
given by the Kushner-Stratonovich/Zakai equation and is generally
infinite-dimensional (Elliott, Aggoun & Moore 1995, Ch. 6). The
discrete-time recursion used here is a standard, well-defined, and exactly
correct *discrete-time* hidden Markov model filter (Hamilton 1989); we make
no claim about a closed-form continuous-time analogue. Anyone requiring the
continuous-time filter should treat dt as a fixed sampling interval (1
second in our experiments) and use the discrete filter as-is — this is
what the simulator does.

**(2) Reservation price under regime switching.** The formula
r ≈ S − qγσ²(T−t) (Section 10) is the single-regime A-S form and is
labelled an approximation. The exact reservation price in the
regime-switching model is r = S + ∂h^k/∂q(q,t) (a one-sided finite
difference h^k(q+1,t)−h^k(q,t) or h^k(q,t)−h^k(q-1,t) on the discrete
grid), evaluated from the numerically solved h. The simulator uses the
numerically computed δᵃ*, δᵇ* directly (Section 8's exact D, E
expressions), not the reservation-price shortcut — the A-S reservation
price formula is presented only as classical context, not as what is
implemented.

**(3) φ vs γ — possible double-counting.** We do not derive φ from γ (e.g.
via φ = ½γσ²Δt_liq). They are treated as two independent tunable
parameters with distinct roles (Section 13): γ shapes intraday spreads via
h(q,t), φ sets the terminal boundary h(q,T) = −(φ/2)q². This is a genuine
simplification relative to a fully consistent single-utility derivation.
We do not claim γ and φ are derived from one risk-aversion primitive; both
are calibration knobs. A fully rigorous treatment would fix one from the
other via the agent's terminal liquidation problem.

**(4) Ansatz validity across regime switches is assumed, not proven.** The
−½γσ_k²q²τ term is regime-dependent and therefore discontinuous in k at
switch times. We do not provide a verification theorem (viscosity solution
argument) showing the candidate V^k = x+qS−½γσ_k²q²τ+h^k(q,t) actually
equals the true value function. The numerical solution is a candidate
solution to the reduced system of Section 9's equations; standard
verification arguments (Fleming & Soner 2006) would be needed for a fully
rigorous claim of optimality. We present this as the standard
Avellaneda-Stoikov-style ansatz extended heuristically to the
regime-switching case, consistent with the broader literature's practice
(Cartea & Jaimungal 2015) but without an independent verification proof.

**(5) σ(π) usage is now consistent.** Following the variance-blend fix
(Section 11, `blended_sigma`), σ²(π) = πσ₂²+(1−π)σ₁² is used wherever a
blended volatility appears (only in the HMM-blended spread formula at the
end of Section 11). The per-regime spread formulas (Section 8/10) use σ_k
directly (k known/assumed) and never reference σ(π) — there is no
remaining ambiguity between linear and RMS forms.

**(6) Continuous control on a discrete inventory grid.** Following standard
A-S practice (Avellaneda & Stoikov 2008), q is treated as continuous for
the ansatz/ODE derivation (Sections 8-9) and the resulting expressions are
then evaluated on the integer grid {−Q,...,Q}. The "derivative" terms that
appear are, in fact, already the finite differences h(q±1,t)−h(q,t) coming
from the unit-size jump terms — i.e. the derivation is finite-difference
from the start, and no continuous ∂h/∂q is separately invoked or required.

**(7) Additional unstated assumptions (extending Section 13):**
- **A, κ are regime-independent**; only the baseline rate A_k differs by
  regime (A₁=5, A₂=50/min) — the price-sensitivity κ is shared. A fully
  regime-dependent κ_k would be a natural extension.
- **No price impact**: the agent's own fills do not move S_t. The book is
  assumed infinitely deep beyond the agent's quotes.
- **Regime k_t and driving Brownian motion W_t are independent processes.**
- **Continuous-time Poisson/Markov dynamics, discretised at dt=1/60h
  (1 min) for the simulator** and dt=1s for the filter experiments —
  treated as small enough that the discrete-time approximations
  (Section 7's ODE system, Section 11's filter) are accurate.
- **Finite horizon T** is a modelling choice (matches a trading session);
  an infinite-horizon/stationary formulation is a standard alternative
  (Guéant, Lehalle & Fernandez-Tapia 2013) not pursued here.
- **No formal verification theorem** is provided connecting the HJB
  candidate to the true value function (see (4)); the controls derived are
  candidates consistent with the formal optimality conditions (FOC + SOC),
  not proven optimal in a fully rigorous stochastic-control sense.
