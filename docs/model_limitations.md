# Model Limitations — Why We Made Every Simplification

This document gives a completely honest account of every simplification
in the model, why it was made, what it costs in realism, and what a
full research-grade implementation would require instead.

Reading this before an interview is more valuable than memorising the
results. Every question a quant researcher can ask about the model's
validity is answered here.

---

## 1. No Queue Position

### What we assumed

When the MM posts a quote at price p, any market order that arrives at
price p immediately fills the MM's order. There is no concept of
*where* in the queue the MM's order sits.

### What actually happens

In a real exchange, multiple market makers post limit orders at the
same price. Orders at the same price are filled in arrival order
(price-time priority). If 10 market makers all post asks at ₹100.50
and only one market buy arrives, only the *first* MM to post fills.
The rest get nothing.

Queue position determines fill probability. Posting early (high queue
priority) means more fills at the cost of adverse selection — you
get filled precisely when informed traders want to trade. Posting late
(low queue priority) means fewer fills but better adverse selection
protection.

This creates a genuine second-order control problem: the MM must
decide not just *what price* to quote but *when* to refresh their
order (cancel and repost to move back to the front of the queue
after being partially filled, or to update the price).

### What our model assumes instead

We assume fill probability depends only on the spread δ via the
exponential model λ(δ) = A e^{−κδ}. This is a reduced-form
approximation that captures the right qualitative behaviour
(tighter spreads fill more) without modelling queue mechanics.

This approximation is most accurate when:
- The MM is a significant fraction of total volume (large A relative
  to total order flow)
- Queue lengths are short (fast markets, thin books)
- The MM operates at a timescale where queue dynamics average out

### What it costs

The model underestimates fill probability at tight spreads (in
practice, a late-arriving order at the best price may never fill)
and overestimates it at wide spreads (in practice, even a wide
spread fills if no one else is at that price level).

More importantly, the model cannot capture the **make/take tradeoff**:
exchanges charge different fees for limit orders (makers, who provide
liquidity) and market orders (takers, who consume it). Queue
position affects whether you are a maker or taker.

### What a full model requires

The LOB queue model of Avellaneda & Stoikov (2008) and its extension
by Cont, Stoikov & Talreja (2010) models the full queue. The state
space expands dramatically:

```
State = (x, q, S, queue_position_ask, queue_position_bid, t)
```

Queue position is itself stochastic (new arrivals push you back,
cancellations move you forward). The resulting HJB is 6-dimensional
and has no closed-form solution. Numerical solution requires either
a very fine finite-difference grid (computationally expensive) or
approximate dynamic programming (function approximation).

### Why we did not add it

The queue model would multiply the state space by roughly 50× (if
modelling queue depth up to 50 orders). The PDE grid would go from
82,000 nodes to ~4 million. Solve time goes from 0.1 seconds to
several minutes. The code complexity increases substantially and the
additional insight for this project's purpose (demonstrating regime
switching) is small relative to that cost.

If the project's goal were specifically to study market microstructure
and maker/taker dynamics, the queue model would be essential.

---

## 2. No Order Book Depth

### What we assumed

The order book is represented by a single best bid and best ask —
the MM's own quotes. There are no other orders in the book.

### What actually happens

Real order books have hundreds of price levels, each with a queue of
orders. The *depth* of the book (how many shares are available at
each price level) matters enormously:

- **Thin books** (shallow depth): a large market order can move the
  price by many ticks. The MM faces high **market impact** — their
  own fills move the price against them.
- **Deep books** (high depth): large orders absorb without moving
  the price. The MM faces low market impact.
- **Order book imbalance**: if there are many more shares offered at
  the ask than at the bid, it predicts short-term price decline.
  Smart MMs use this as a signal.

### What our model assumes instead

The mid-price evolves as a pure Brownian motion, completely
independent of trading activity. There is no market impact — the
MM's trades do not move the price at all.

### What it costs

Ignoring market impact is acceptable when:
- The MM is small relative to total market volume (< 1-2%)
- Trade sizes are small (1 share at a time in our model)
- We care about the *average* outcome over many sessions rather
  than individual large trades

It becomes a serious problem when:
- Modelling institutional-sized orders
- Studying market impact and optimal execution (a completely
  separate but related problem — Almgren-Chriss framework)
- Modelling the feedback between the MM's activity and the price
  process (reflexivity)

Order book imbalance as a signal is also a genuine source of alpha
that we are completely ignoring. In practice, real MMs skew their
quotes not just based on inventory but also based on order book state.

### What a full model requires

A full LOB model maintains the entire price-quantity schedule on both
sides. The price process becomes:

```
dS_t = σ dW_t  +  f(order_flow_imbalance) dt
```

where f(·) is a market impact function. The Cont-Kukanov-Stoikov
model (2014) derives optimal MM strategies in this setting.
The state space includes the full book snapshot, making direct
DP intractable — simulation-based methods or reinforcement
learning become necessary.

### Why we did not add it

Modelling depth requires tracking an order book data structure at
each simulation step. For 5000 simulations × 60 steps, maintaining
a sorted price-quantity schedule adds O(log n) overhead per event
and significant memory. More importantly, the book dynamics would
require calibration from real tick data, which we do not have
access to in this environment.

---

## 3. No Cancellations

### What we assumed

Once the MM posts a quote, it either fills (a market order arrives)
or it stays posted until the next quote refresh. There is no
*explicit* cancellation decision.

### What actually happens

In real electronic markets, the majority of limit orders are
cancelled before they fill. Cancellation rates of 90-99% are
common in HFT. Cancellations happen for several reasons:

- **Quote update**: the MM wants to reprice (because the mid-price
  moved, or because inventory changed)
- **Adverse selection avoidance**: the MM detects a toxic order
  flow signal and pulls their quote before getting picked off
- **Inventory management**: the MM has hit their limit and must
  stop providing liquidity on one side

The decision of *when to cancel* is a separate control variable.
The optimal cancellation strategy interacts with the optimal
posting strategy in a non-trivial way.

### What our model assumes instead

We implicitly assume the MM refreshes quotes at every time step dt.
If dt = 1 minute, the MM posts new quotes every minute regardless
of what happened in between. This is equivalent to assuming
cancellations are free and instantaneous.

### What it costs

This is actually a reasonable approximation for the timescale we
are modelling (minutes). At sub-second timescales, the cancellation
decision becomes crucial. The Bayesian model of Copeland & Galai
(1983) and extensions study when a rational MM should pull quotes.

The more important omission is **adverse selection avoidance**:
real MMs cancel quotes when they detect a toxic flow signal (e.g.,
when the price moves strongly in one direction, suggesting an
informed trader is working a large order). We handle adverse
selection through the Wonham filter (widening spreads when π_t is
high) but not through strategic cancellation.

### What a full model requires

Add a third control: cancel indicator c_t ∈ {0, 1}. The HJB gains
an additional term for the option value of cancellation. The
resulting control problem is:

```
max over {δᵃ, δᵇ, c} of expected terminal wealth
```

This is studied in Cartea & Jaimungal (2015), Chapter 8. The
cancellation option adds significant value — the MM can be more
aggressive (tighter spreads) when they know they can pull quotes
if adverse flow arrives.

### Why we did not add it

Adding cancellation as a control triples the effective control space
and significantly complicates the HJB. The first-order conditions
no longer have a clean closed form. More practically: to model
*when* to cancel, we need a model of adverse flow detection that
goes beyond what the Wonham filter provides (the filter updates
beliefs but does not directly trigger cancellations).

---

## 4. No Latency

### What we assumed

Quote updates, order submissions, and trade confirmations are all
instantaneous. When the MM computes new spreads at time t, those
quotes are live at time t.

### What actually happens

In electronic markets, there are three distinct latency sources:

**Market data latency**: the time between a trade occurring on the
exchange and the MM receiving notification. Even with co-location
(servers physically next to the exchange), this is 1-100 microseconds.
Without co-location, it can be 1-10 milliseconds.

**Processing latency**: the time the MM's system takes to compute
new quotes after receiving market data. This is 1-100 microseconds
for a well-optimised system.

**Order submission latency**: the time between the MM sending a new
quote and it being live on the exchange. Another 1-100 microseconds.

Total roundtrip latency for a typical prop trading firm ranges from
10 microseconds (co-located, FPGA-based) to 10+ milliseconds
(remote, software-based).

### Why latency matters

During the latency period, the market continues to move. If the
price moves by ΔS during roundtrip latency L, the MM's stale quote
is mispriced by ΔS. A faster trader (with lower latency) can trade
against the stale quote before the MM cancels it — this is called
**latency arbitrage** and is a major source of adverse selection.

The adverse selection cost from latency is approximately:

```
Cost ≈ σ · √L · (probability of being picked off)
```

Higher volatility (larger σ) or higher latency (larger L) both
increase the cost. This means:
- In our chaotic regime (σ₂ large), latency matters more
- A faster MM (lower L) can post tighter spreads profitably

### What our model assumes instead

By modelling in 1-minute steps (dt = 1/60 hour), we effectively
assume all latency is negligible compared to the timescale of
interest. This is reasonable for a slow market maker operating at
minute timescales. It becomes completely unrealistic for HFT
operating at microsecond timescales.

### What a full model requires

Latency-aware market making requires modelling the MM's information
as a *delayed* version of the true state. The value function gains
a new state variable: the time since last update. The optimal
strategy must account for the fact that by the time a quote is
live, the world has already moved.

Cont & Kukanov (2012) model latency in the optimal posting problem.
The resulting stale-quote risk is incorporated into a wider bid-ask
spread — exactly our adverse selection model, but with a more
precise source.

### Why we did not add it

Latency modelling at the minute timescale we operate at contributes
nothing — latency effects average out over a minute. At the
microsecond scale, latency is the *dominant* concern, but modelling
it requires microsecond-resolution market data and a very different
simulation architecture (event-driven at nanosecond resolution,
not time-stepped at minute resolution).

---

## 5. No Real Market Data Calibration

### What we have

Parameters estimated from synthetic data calibrated to match the
*documented* statistical properties of SPY from academic literature.
We use σ_calm ≈ 0.04%/min, σ_chaotic ≈ 0.10%/min, average calm
duration ≈ 62 minutes, average chaotic duration ≈ 22 minutes.

### What we do not have

Direct calibration from actual tick data. We do not use real
trade-by-trade records, real bid/ask quotes, real order arrivals,
or real regime identification on actual SPY data.

### Why this matters

The difference between "calibrated to literature values" and
"estimated from real data" is significant:

1. **Intraday seasonality**: real markets are more volatile at open
   and close (U-shaped intraday vol pattern). Our model has constant
   parameters throughout the session.

2. **Jump processes**: real prices occasionally jump discontinuously
   (earnings announcements, macro news). Pure Brownian motion
   cannot capture this. A Merton jump-diffusion or Bates model
   would be needed.

3. **Microstructure noise**: at very short timescales, observed
   prices bounce between bid and ask (bid-ask bounce). This adds
   apparent volatility that is not fundamental price uncertainty.

4. **Correlated regimes across assets**: if SPY is in a chaotic
   regime, all large-cap stocks tend to be simultaneously. A
   single-asset model misses this correlation.

5. **Non-exponential regime durations**: the Markov chain model
   implies exponentially distributed regime durations (memoryless).
   Real regimes often have more complex duration distributions —
   a volatile period is more likely to end if it has already lasted
   a long time (hazard rate increases with duration).

### What accessing real data would require

Tick data for US equities is available from:
- **TAQ (Trade and Quote)** database via WRDS — academic access,
  requires institutional subscription
- **Refinitiv Tick History** — commercial, expensive
- **Lobster** (Limit Order Book System) — reconstructed NASDAQ
  LOB data, academic pricing available
- **IEX Cloud** — free tier available, but limited history

With real data, the estimation procedure would be:
1. Compute 5-minute realised volatility from tick data
2. Fit a 2-state HMM to the RV series (Gaussian or Gamma emissions)
3. Run Viterbi decoding to get regime labels
4. Estimate transition rates from labelled series
5. Estimate arrival rates from order flow data per regime

The sentence "parameters estimated from LOBSTER NASDAQ tick data"
on a resume is worth substantially more than "calibrated to
literature values."

### Why we did not use real data

No live data API access in this environment. This is an
infrastructure limitation, not a modelling choice. The calibration
procedure is fully implemented and documented — plugging in real
data is a straightforward substitution.

---

## 6. Reduced-Form Fill Process

### What we assumed

Fill probability per unit time = A_k · e^{−κδ}. This is the
reduced-form model of Avellaneda & Stoikov (2008), treating the
arrival of market orders as a Poisson process whose intensity
depends on the spread.

### What actually happens

Whether a limit order fills depends on multiple factors that our
model collapses into a single parameter:

**Relative spread position**: our model quotes at exactly S ± δ.
Real MMs choose from a discrete grid of prices (ticks). Quoting
at the best bid/ask is very different from quoting one tick behind.

**Volume at the best price**: even if you are at the best price,
you only fill if the arriving market order is large enough to
consume all volume ahead of you in the queue.

**Order imbalance**: if there are 10× more buy orders than sell
orders in recent flow, sell-side market orders are less likely.
Fill rates on the ask side should be adjusted downward.

**Volatility-dependent fill rates**: in volatile periods, informed
traders are more aggressive. They send larger market orders. This
means arrival *sizes* matter, not just arrival *counts*.

**Correlation between arrivals**: buys and sells are not
independent Poisson processes in practice. They are anti-correlated
at short timescales (buy order depletes ask-side, reducing next
buy probability) and positively correlated at longer timescales
(trending markets have more buys than sells for extended periods).

### What the reduced-form captures

The exponential model λ(δ) = A e^{−κδ} captures the single most
important relationship: tighter spreads attract more orders. The
parameter A captures regime-level activity (A₂ > A₁ in chaotic).
The parameter κ captures how price-sensitive the order flow is.

Empirically, exponential arrival rate models fit reasonably well
at timescales of seconds to minutes. At sub-second timescales,
the autocorrelation in order flow becomes important.

### What a full model requires

The Cont-Stoikov-Talreja (2010) model specifies arrival rates that
depend on the full order book state, not just the spread. The state
space is the entire book, making direct DP intractable.

Alternatively, the arrival rate can be estimated non-parametrically
from real data: for each (spread, time-of-day, regime) bucket,
compute the empirical fill rate. This produces a lookup table
rather than a parametric model — more accurate but requires
substantial data.

### Why we use the reduced-form

The reduced-form is the standard model in the academic literature
(Avellaneda-Stoikov, Cartea-Jaimungal, Gueant-Lehalle-Tapia). It
produces tractable HJB equations with near-closed-form solutions.
Every extension we would want to make (regime switching, adverse
selection, latency) was built on top of this foundation.

Departing from it would require either:
a) Accepting intractability and moving to RL/simulation-based methods
b) Using real tick data to estimate the full non-parametric arrival model

Neither is unreasonable — they are simply different projects.

---

## Summary: What Would Each Addition Give?

| Feature | Realism gain | Complexity cost | Required for |
|---|---|---|---|
| Queue position | High (fill prob realistic) | High (state space ×50) | HFT microstructure research |
| Order book depth | Medium (market impact) | High (LOB data structure) | Institutional execution |
| Cancellations | Medium (adverse sel. avoidance) | Medium (extra control) | Sub-second strategies |
| Latency | Low at 1-min scale | High (delayed obs. model) | Microsecond HFT |
| Real data | High (honest calibration) | Low (data access issue) | Any publication |
| Full fill process | Medium | Medium (non-parametric) | Precise PnL estimates |

### The honest position

This project is correctly described as: *an optimal inventory control
model for a stylised market maker under regime-switching volatility.*

It is not described as: *a realistic HFT simulator* or *a production
trading system.*

The simplifications are the standard simplifications of the academic
literature in this area (Avellaneda-Stoikov 2008, Cartea-Jaimungal
2015, Gueant 2017). Every quant researcher in this field works with
these same simplifications and studies extensions to them.

The value of this project is demonstrating that you understand:
1. Why the simplifications are made (tractability, not laziness)
2. What each costs in realism
3. What the path to removing each one looks like
4. Where the model is and is not applicable

That understanding is what separates a project that impresses
researchers from one that merely implements existing results.

---

## Further Reading

These papers represent the frontier of each limitation discussed:

**Queue position and LOB dynamics**
- Cont, Stoikov & Talreja (2010) — *A stochastic model for order book dynamics*
- Avellaneda & Stoikov (2008) — *High-frequency trading in a limit order book*

**Cancellations and optimal refresh**
- Cartea & Jaimungal (2015) — *Algorithmic and High-Frequency Trading* (Ch. 8)
- Guéant, Lehalle & Fernandez-Tapia (2013) — *Dealing with inventory risk*

**Latency and stale quotes**
- Cont & Kukanov (2012) — *Optimal order placement in limit order markets*
- Moallemi & Saglam (2013) — *The cost of latency in high-frequency trading*

**Market impact and depth**
- Almgren & Chriss (2001) — *Optimal execution of portfolio transactions*
- Cont, Kukanov & Stoikov (2014) — *The price impact of order book events*

**Real data calibration**
- Andersen et al. (2001) — *The distribution of realised exchange rate volatility*
- Christoffersen (2011) — *Elements of Financial Risk Management*
- Large (2007) — *Measuring the resiliency of an electronic limit order book*

**RL as an alternative approach**
- Spooner et al. (2018) — *Market making via reinforcement learning*
- Gasperov & Kostanjcar (2021) — *Market making with signals and latent information*
