"""
backtest_engine.py
==================
Python backtest engine — runs three agents through a regime-switching
simulated market and reports full metrics.

Run:
    python backtest/src/backtest_engine.py --config configs/default.yaml
"""

import numpy as np
import yaml
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../pde_solver/src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../hmm_filter/src"))

from optimal_spreads import SpreadTable
from wonham_filter import WonhamFilter
from metrics import compute_metrics, print_comparison_table, AgentMetrics


# ─────────────────────────────────────────────────────────────────────
#  Simulator helpers
# ─────────────────────────────────────────────────────────────────────

def simulate_regime_path(cfg: dict, rng: np.random.Generator, n_steps: int, dt: float):
    """Simulate regime switches and price path."""
    r = cfg["regime"]
    q12, q21 = r["q_12"], r["q_21"]
    s1, s2   = r["sigma_1"], r["sigma_2"]

    S = np.zeros(n_steps)
    regimes = np.zeros(n_steps, dtype=int)
    S[0] = cfg["simulator"]["mid_price_init"]
    regime = 0

    for i in range(1, n_steps):
        # Regime switch
        if regime == 0:
            if rng.uniform() < q12 * dt:
                regime = 1
        else:
            if rng.uniform() < q21 * dt:
                regime = 0
        regimes[i] = regime

        sigma = s2 if regime == 1 else s1
        S[i] = S[i - 1] + sigma * rng.normal() * np.sqrt(dt)

    return S, regimes


def optimal_spread_approx(q, tau, sigma, gamma, kappa, q_max):
    """A-S closed form spread — used for all agents as base."""
    q_cl = float(np.clip(q, -q_max, q_max))
    base  = 1.0 / kappa + 0.5 * gamma * sigma**2 * tau
    skew  = gamma * sigma**2 * tau * q_cl
    da    = max(base - skew, 1e-4)
    db    = max(base + skew, 1e-4)
    return da, db


# ─────────────────────────────────────────────────────────────────────
#  Agent runners
# ─────────────────────────────────────────────────────────────────────

def run_agent(agent_type, S, regimes, cfg, rng, spread_table=None, hmm_filter=None):
    """
    Simulate one agent on a given price/regime path.

    agent_type:
      "oracle" - CLAIVOYANT ORACLE: uses the TRUE hidden regime k_t directly
                 to pick sigma_k each step. NOT implementable in practice
                 (regime is hidden) - this is an upper-bound benchmark only.
      "hmm"    - uses the Bayesian filter's belief pi_t (no access to true k_t),
                 blending sigma via sqrt(pi*sigma2^2+(1-pi)*sigma1^2). This is
                 the practically implementable regime-aware agent.
      "naive"  - constant blended sigma (RMS-weighted by stationary P(regime)).
      "fixed"  - constant symmetric spread 1/kappa.
    Returns (pnl_series, inv_series, spread_earned, n_trades)
    """
    n  = cfg["order_flow"]
    a  = cfg["agent"]
    r  = cfg["regime"]
    t_cfg = cfg["time"]

    kappa  = n["kappa"]
    gamma  = a["gamma"]
    q_max  = a["q_max"]
    T      = t_cfg["T"]
    dt     = t_cfg["dt_pde"]
    n_steps = len(S)

    # For naive: RMS (variance-weighted) blend by stationary distribution.
    # Using the linear blend (1-pi)*s1+pi*s2 understates volatility by Jensen's
    # inequality and is inconsistent with the RMS blend used by the HMM filter
    # (hmm_filter/src/wonham_filter.py: blended_sigma). RMS used here for
    # consistency between the "naive" and "hmm" agents' volatility inputs.
    pi_star   = r["q_12"] / (r["q_12"] + r["q_21"])
    sigma_avg = np.sqrt((1 - pi_star) * r["sigma_1"]**2 + pi_star * r["sigma_2"]**2)

    # Arrival rates per regime
    A = [n["A_1"], n["A_2"]]

    cash = 0.0
    q    = 0
    pnl_series = [0.0]
    inv_series = [0]
    spread_earned = 0.0
    n_trades = 0

    if hmm_filter is not None:
        hmm_filter.reset()

    for i in range(1, n_steps):
        t   = i * dt
        tau = max(T - t, 1e-6)
        k   = int(regimes[i])
        mid = S[i]

        if abs(q) >= q_max:
            pnl_series.append(cash + q * mid)
            inv_series.append(q)
            continue

        # Get spreads based on agent type
        if agent_type == "oracle":
            # CLAIRVOYANT: uses true k_t. Upper-bound benchmark, not implementable.
            sigma_k = r["sigma_2"] if k == 1 else r["sigma_1"]
            da, db  = optimal_spread_approx(q, tau, sigma_k, gamma, kappa, q_max)

        elif agent_type == "hmm":
            # Practical regime-aware agent: infers pi_t from price increments only.
            dS = S[i] - S[i - 1]
            pi = hmm_filter.update(dS)
            sigma_blend = np.sqrt(pi*r["sigma_2"]**2 + (1-pi)*r["sigma_1"]**2)
            da, db = optimal_spread_approx(q, tau, sigma_blend, gamma, kappa, q_max)

        elif agent_type == "naive":
            da, db = optimal_spread_approx(q, tau, sigma_avg, gamma, kappa, q_max)

        else:  # fixed
            da, db = 0.5, 0.5

        # Poisson arrivals (exact form 1-exp(-lam*dt), not the lam*dt approximation)
        A_k    = A[k]
        rate_a = 1.0 - np.exp(-A_k * np.exp(-kappa * da) * dt)
        rate_b = 1.0 - np.exp(-A_k * np.exp(-kappa * db) * dt)

        ask_hit = rng.uniform() < rate_a
        bid_hit = rng.uniform() < rate_b

        # Adverse selection: with regime-dependent probability, an informed
        # trader hits the quote and the price moves against the MM by
        # alpha * spread (see README "Adverse Selection Model" / configs
        # adv_selection block). Applied identically to all agent types so
        # the comparison isolates the *spread-setting policy*, not exposure.
        adv = cfg.get("adv_selection", {})
        alpha   = adv.get("alpha", 0.0)
        p_inf_k = adv.get("p_informed_calm", 0.0) if k == 0 else adv.get("p_informed_chaotic", 0.0)

        if ask_hit:
            cash    += mid + da
            q       -= 1
            spread_earned += da
            n_trades += 1
            if alpha > 0 and rng.uniform() < p_inf_k:
                S[i] += alpha * da   # informed buy -> price moves up against MM

        if bid_hit and abs(q) < q_max:
            cash    -= mid - db
            q       += 1
            spread_earned += db
            n_trades += 1
            if alpha > 0 and rng.uniform() < p_inf_k:
                S[i] -= alpha * db   # informed sell -> price moves down against MM

        mid = S[i]  # reflect any adverse-selection move in this step's mark
        pnl_series.append(cash + q * mid)
        inv_series.append(q)

    return (
        np.array(pnl_series),
        np.array(inv_series),
        spread_earned,
        n_trades,
    )


# ─────────────────────────────────────────────────────────────────────
#  Main backtest loop
# ─────────────────────────────────────────────────────────────────────

def run_backtest(cfg_path: str, results_dir: str = "results") -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    n_sims = cfg["simulator"]["n_simulations"]
    seed   = cfg["simulator"]["seed"]
    dt     = cfg["time"]["dt_pde"]
    T      = cfg["time"]["T"]
    n_steps = int(T / dt)

    rng = np.random.default_rng(seed)

    # Try to load pre-solved spread table; fall back gracefully
    spread_table = None
    try:
        spread_table = SpreadTable.load(results_dir)
    except FileNotFoundError:
        print("  [info] No spread table found — using closed-form spreads for all agents.")

    # Wonham filter for HMM agent
    hmm = WonhamFilter(
        sigma_1=cfg["regime"]["sigma_1"],
        sigma_2=cfg["regime"]["sigma_2"],
        q_12=cfg["regime"]["q_12"],
        q_21=cfg["regime"]["q_21"],
        pi_init=cfg["hmm_filter"]["pi_init"],
        dt=dt,
    )

    agent_types = ["oracle", "hmm", "naive", "fixed"]
    agent_names = ["Oracle (true regime)", "HMM-Filter (Bayesian, hidden regime)",
                   "Naive-ConstantVol", "SymmetricFixed"]

    # Collect results
    all_pnl       = {a: [] for a in agent_types}
    all_pnl_series = {a: [] for a in agent_types}
    all_inv_series = {a: [] for a in agent_types}
    all_spread    = {a: [] for a in agent_types}
    all_trades    = {a: [] for a in agent_types}

    print(f"Running {n_sims} simulations, {n_steps} steps each...")

    for sim in range(n_sims):
        S, regimes = simulate_regime_path(cfg, rng, n_steps, dt)

        for atype in agent_types:
            pnl_s, inv_s, sp, nt = run_agent(
                # Issue K: pass a copy - adverse-selection mutates S in place,
                # which would otherwise contaminate later agents with earlier
                # agents' price impact and create order-dependent results.
                atype, S.copy(), regimes, cfg, rng,
                spread_table=spread_table,
                hmm_filter=hmm if atype == "hmm" else None,
            )
            all_pnl[atype].append(pnl_s[-1])
            all_pnl_series[atype].append(pnl_s)
            all_inv_series[atype].append(inv_s)
            all_spread[atype].append(sp)
            all_trades[atype].append(nt)

        if (sim + 1) % max(1, n_sims // 10) == 0:
            print(f"  {sim+1}/{n_sims} done")

    # Compute and print metrics
    metrics_list = []
    q_max = cfg["agent"]["q_max"]
    for atype, aname in zip(agent_types, agent_names):
        m = compute_metrics(
            name=aname,
            pnl_runs=np.array(all_pnl[atype]),
            pnl_series_runs=all_pnl_series[atype],
            inv_series_runs=all_inv_series[atype],
            spread_earned_runs=np.array(all_spread[atype]),
            trade_counts=np.array(all_trades[atype]),
            q_max=q_max,
        )
        metrics_list.append(m)

    print_comparison_table(metrics_list)

    # Save terminal PnLs
    os.makedirs(results_dir, exist_ok=True)
    for atype, aname in zip(agent_types, agent_names):
        fname = os.path.join(results_dir, f"pnl_{atype}.npy")
        np.save(fname, np.array(all_pnl[atype]))
    print(f"\nPnL arrays saved to {results_dir}/")


# ─────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    run_backtest(args.config, args.results)
