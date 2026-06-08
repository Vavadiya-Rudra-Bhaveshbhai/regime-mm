# Regime-Switching Optimal Market Making Engine

A full implementation of a stochastic control framework for electronic market making under regime-switching volatility — from mathematical formulation to backtested C++ simulator.

---

## What This Project Does

Standard market making models (Avellaneda-Stoikov) assume constant volatility. Real markets flip between **calm** and **chaotic** regimes in seconds. This project:

1. Models volatility as a **2-state continuous-time Markov chain** (σ₁ ≪ σ₂)
2. Derives and numerically solves a **coupled Hamilton-Jacobi-Bellman (HJB) PDE system** to compute optimal bid/ask spreads per regime
3. Implements a **high-performance discrete-event LOB simulator in C++** with regime-switching Poisson order arrivals
4. Runs an **online Wonham HMM filter** to infer the latent regime from observed order flow and dynamically blend spreads

---

## Repository Structure

```
regime_mm/
│
├── docs/                        # Math derivations, notes, references
│   ├── math_derivation.md       # Full HJB derivation from scratch
│   └── references.md            # Key papers (Avellaneda-Stoikov, Cartea-Jaimungal)
│
├── pde_solver/                  # Phase 2: Numerical HJB solver (Python)
│   ├── include/                 # (reserved for C++ port)
│   ├── src/
│   │   ├── hjb_solver.py        # Crank-Nicolson solver for coupled HJB system
│   │   ├── optimal_spreads.py   # Extract optimal δᵃ*, δᵇ* from solved V
│   │   └── visualize_pde.py     # Plot value function and spreads
│   └── tests/
│       └── test_hjb_solver.py
│
├── simulator/                   # Phase 2/3: C++ discrete-event LOB simulator
│   ├── include/
│   │   ├── order_book.hpp       # Limit order book data structure
│   │   ├── market_maker.hpp     # MM agent interface
│   │   ├── regime_model.hpp     # Markov chain regime state
│   │   └── event_queue.hpp      # Priority queue for events
│   ├── src/
│   │   ├── main.cpp             # Entry point
│   │   ├── order_book.cpp
│   │   ├── market_maker.cpp     # Regime-switching MM strategy
│   │   ├── naive_agent.cpp      # Baseline constant-vol agent
│   │   └── regime_model.cpp
│   ├── tests/
│   │   └── test_simulator.cpp
│   └── CMakeLists.txt
│
├── hmm_filter/                  # Phase 4: Online Wonham HMM filter
│   ├── include/
│   │   └── wonham_filter.hpp
│   ├── src/
│   │   ├── wonham_filter.py     # Continuous-time belief update dπ_t
│   │   └── regime_inference.py  # Attach filter to live order flow
│   └── tests/
│       └── test_wonham.py
│
├── backtest/                    # Phase 3: Backtesting and metrics
│   ├── include/
│   ├── src/
│   │   ├── backtest_engine.py   # Run agents through simulator output
│   │   └── metrics.py           # Sharpe, max inventory drawdown, halt freq
│
├── scripts/
│   ├── run_pde_solver.sh        # End-to-end: solve HJB → save spread tables
│   ├── run_backtest.sh          # Run all agents, generate comparison report
│   └── install_deps.sh          # Install Python + C++ dependencies
│
├── configs/
│   └── default.yaml             # All model parameters in one place
│
├── results/                     # Auto-generated plots and CSV outputs (gitignored)
│
├── CMakeLists.txt               # Root C++ build config
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

---

## Phases

| Phase | What | Files |
|-------|------|-------|
| 1 | Math formulation | `docs/math_derivation.md` |
| 2 | HJB PDE solver + C++ simulator skeleton | `pde_solver/`, `simulator/` |
| 3 | Full backtest + metrics | `backtest/` |
| 4 | HMM filter + live regime inference | `hmm_filter/` |

---

## Quick Start

### Python (PDE solver)
```bash
pip install -r requirements.txt
python pde_solver/src/hjb_solver.py
python pde_solver/src/visualize_pde.py
```

### C++ (Simulator)
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
./regime_mm_sim --config ../configs/default.yaml
```

---

## Key Math

### Coupled HJB System (2 regimes)

For k ∈ {1, 2}:

```
∂Vᵏ/∂t + ½σₖ² ∂²Vᵏ/∂S²
  + max_{δᵃ} { λₖᵃ(δᵃ) · [Vᵏ(x+S+δᵃ, q−1, S, t) − Vᵏ] }
  + max_{δᵇ} { λₖᵇ(δᵇ) · [Vᵏ(x−S+δᵇ, q+1, S, t) − Vᵏ] }
  + Σⱼ≠ₖ qₖⱼ · (Vʲ − Vᵏ)  =  0
```

Terminal condition: `Vᵏ(x, q, S, T) = x + qS − (φ/2)q²`

### Wonham HMM Filter

```
dπ_t = [q₁₂(1−π_t) − q₂₁π_t] dt
       + π_t(1−π_t) · [(σ₂−σ₁)/σ(π_t)] · dI_t
```

### Optimal Spread (per regime, approximate closed form)

```
δₖ*  =  1/κ  +  γσₖ²(T−t)/2  ±  inventory skew
```

---

## Parameters (`configs/default.yaml`)

| Parameter | Symbol | Default | Meaning |
|-----------|--------|---------|---------|
| `sigma_1` | σ₁ | 0.5 | Low-vol regime volatility |
| `sigma_2` | σ₂ | 2.0 | High-vol regime volatility |
| `q_12` | q₁₂ | 0.5 | Calm → chaotic rate (per hour) |
| `q_21` | q₂₁ | 2.0 | Chaotic → calm rate (per hour) |
| `gamma` | γ | 0.1 | Risk aversion |
| `kappa` | κ | 1.5 | Order flow sensitivity to spread |
| `A` | A | 10.0 | Baseline arrival rate |
| `T` | T | 1.0 | Trading horizon (hours) |
| `q_max` | — | 20 | Max inventory (±20 shares) |
| `phi` | φ | 0.01 | Terminal inventory penalty |

---

## Results (after running backtest)

| Agent | Sharpe Ratio | Max Inv Drawdown | Halt Frequency |
|-------|-------------|-----------------|----------------|
| Regime-switching (ours) | TBD | TBD | TBD |
| Naive constant-vol | TBD | TBD | TBD |
| Symmetric fixed spread | TBD | TBD | TBD |

---

## References

- Avellaneda & Stoikov (2008) — *High-frequency trading in a limit order book*
- Cartea, Jaimungal & Penalva (2015) — *Algorithmic and High-Frequency Trading*
- Elliott, Aggoun & Moore (1995) — *Hidden Markov Models: Estimation and Control*
- Wonham (1964) — *Some applications of stochastic differential equations to optimal nonlinear filtering*
