"""
metrics.py
==========
Computes all performance metrics used to evaluate market making agents.

Metrics:
  - Sharpe Ratio (of PnL increments)
  - Maximum Inventory Drawdown
  - Inventory Halt Frequency
  - PnL mean, std, total
  - Trade count and fill rate
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentMetrics:
    name:                  str
    sharpe_ratio:          float
    mean_terminal_pnl:     float
    std_terminal_pnl:      float
    max_inventory:         float
    halt_frequency:        float      # fraction of steps where |q| >= q_max
    total_trades:          int
    avg_spread_earned:     float
    pnl_series:            np.ndarray  # shape (n_sims,) terminal PnLs


def sharpe_ratio(pnl_series: np.ndarray, risk_free: float = 0.0) -> float:
    """
    Sharpe ratio of PnL increments.

    pnl_series: time series of mark-to-market PnL values (not increments).
    Returns annualised-like Sharpe (we don't scale since intraday).
    """
    returns = np.diff(pnl_series)
    if returns.std() < 1e-12:
        return 0.0
    excess = returns - risk_free
    return float(excess.mean() / excess.std())


def max_inventory_drawdown(inventory_series: np.ndarray) -> float:
    """
    Maximum absolute inventory held at any point.
    Measures peak inventory risk exposure.
    """
    return float(np.max(np.abs(inventory_series)))


def halt_frequency(inventory_series: np.ndarray, q_max: int) -> float:
    """
    Fraction of time steps where |inventory| >= q_max.
    High halt frequency = agent is getting toxic order flow and running out of room.
    """
    return float(np.mean(np.abs(inventory_series) >= q_max))


def compute_metrics(
    name:              str,
    pnl_runs:          np.ndarray,   # shape (n_sims,) — terminal PnL per run
    pnl_series_runs:   list[np.ndarray],  # list of per-step PnL series
    inv_series_runs:   list[np.ndarray],  # list of per-step inventory series
    spread_earned_runs: np.ndarray,  # shape (n_sims,) — total spread earned
    trade_counts:      np.ndarray,   # shape (n_sims,)
    q_max:             int,
) -> AgentMetrics:
    """
    Aggregate metrics across all simulation runs.
    """
    # Per-run Sharpe ratios
    sharpes = [sharpe_ratio(series) for series in pnl_series_runs]
    avg_sharpe = float(np.mean(sharpes))

    # Per-run max inventory
    max_invs = [max_inventory_drawdown(inv) for inv in inv_series_runs]
    avg_max_inv = float(np.mean(max_invs))

    # Per-run halt frequency
    halts = [halt_frequency(inv, q_max) for inv in inv_series_runs]
    avg_halt = float(np.mean(halts))

    return AgentMetrics(
        name=name,
        sharpe_ratio=avg_sharpe,
        mean_terminal_pnl=float(np.mean(pnl_runs)),
        std_terminal_pnl=float(np.std(pnl_runs)),
        max_inventory=avg_max_inv,
        halt_frequency=avg_halt,
        total_trades=int(np.sum(trade_counts)),
        avg_spread_earned=float(np.mean(spread_earned_runs)),
        pnl_series=pnl_runs,
    )


def print_comparison_table(metrics_list: list[AgentMetrics]) -> None:
    """
    Print a formatted comparison table of all agents.
    """
    header = (
        f"{'Agent':<24} {'Sharpe':>10} {'Mean PnL':>10} "
        f"{'Std PnL':>10} {'Max Inv':>10} {'Halt Freq':>12} {'Trades':>8}"
    )
    print("\n" + "=" * len(header))
    print("BACKTEST RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for m in metrics_list:
        print(
            f"{m.name:<24} {m.sharpe_ratio:>10.4f} {m.mean_terminal_pnl:>10.4f} "
            f"{m.std_terminal_pnl:>10.4f} {m.max_inventory:>10.2f} "
            f"{m.halt_frequency:>12.4f} {m.total_trades:>8d}"
        )
    print("=" * len(header))

    # Highlight improvement of regime-switching agent over naive
    if len(metrics_list) >= 2:
        rs = metrics_list[0]
        naive = metrics_list[1]
        sharpe_improvement = (rs.sharpe_ratio - naive.sharpe_ratio) / (abs(naive.sharpe_ratio) + 1e-12) * 100
        inv_reduction = (naive.max_inventory - rs.max_inventory) / (naive.max_inventory + 1e-12) * 100
        print(f"\n  Regime-switching vs Naive:")
        print(f"    Sharpe improvement : {sharpe_improvement:+.1f}%")
        print(f"    Inventory reduction: {inv_reduction:+.1f}%")
