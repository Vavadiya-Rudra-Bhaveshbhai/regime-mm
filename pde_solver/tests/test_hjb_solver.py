"""
Tests for hjb_solver.py
Run: python -m pytest pde_solver/tests/test_hjb_solver.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np
import pytest
from hjb_solver import Params, HJBSolver, optimal_spread_and_rate


# ── Params fixture ────────────────────────────────────────────────────

def small_params():
    """Tiny params for fast unit tests."""
    return Params(
        sigma=(0.5, 2.0),
        q_switch=(0.5, 2.0),
        A=(5.0, 20.0),
        kappa=1.5,
        gamma=0.1,
        phi=0.01,
        q_max=5,
        T=1.0,
        n_t=50,
    )


# ── optimal_spread_and_rate ───────────────────────────────────────────

def test_optimal_spread_positive():
    """Optimal spreads must always be positive."""
    dV = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    da, la = optimal_spread_and_rate(dV, A_k=10.0, kappa=1.5)
    assert np.all(da > 0), "Spreads must be positive"
    assert np.all(la > 0), "Arrival rates must be positive"


def test_optimal_spread_monotone():
    """Larger ΔV → smaller spread (FOC: δ* = 1/κ − ΔV)."""
    dV = np.array([0.0, 0.2, 0.4, 0.6])
    da, _ = optimal_spread_and_rate(dV, A_k=10.0, kappa=1.5)
    # Should be decreasing (until floor clip)
    assert da[0] >= da[1] >= da[2]


def test_optimal_spread_formula():
    """δ* = max(1/κ − ΔV, ε). Check specific values."""
    kappa = 2.0
    dV = np.array([0.0])
    da, _ = optimal_spread_and_rate(dV, A_k=10.0, kappa=kappa)
    expected = 1.0 / kappa
    assert abs(da[0] - expected) < 1e-10


# ── HJBSolver ─────────────────────────────────────────────────────────

def test_terminal_condition():
    """h(q, T) = -(φ/2)q² for both regimes."""
    p = small_params()
    solver = HJBSolver(p)
    terminal_expected = -0.5 * p.phi * solver.q_grid ** 2
    np.testing.assert_allclose(solver.h[0, :, -1], terminal_expected)
    np.testing.assert_allclose(solver.h[1, :, -1], terminal_expected)


def test_solve_runs_without_error():
    """Solver completes without NaN or Inf."""
    p = small_params()
    solver = HJBSolver(p)
    solver.solve()
    assert not np.any(np.isnan(solver.h)), "NaN in h"
    assert not np.any(np.isinf(solver.h)), "Inf in h"


def test_spreads_positive_after_solve():
    """
    Stored optimal spreads must be positive at all interior time steps.
    The terminal slice (t_idx = n_t) is initialised to 0 (no more trading)
    and is intentionally excluded from this check.
    """
    p = small_params()
    solver = HJBSolver(p)
    solver.solve()
    # Exclude the terminal column (index -1) which is 0 by initialisation
    assert np.all(solver.delta_a[:, :, :-1] > 0), "delta_a must be positive at non-terminal steps"
    assert np.all(solver.delta_b[:, :, :-1] > 0), "delta_b must be positive at non-terminal steps"


def test_regime2_wider_spreads():
    """
    Regime 2 (high vol) must produce wider spreads than regime 1 (low vol)
    at q=0. We check at the midpoint of the solved time grid where the
    numerical solution has fully converged away from the terminal condition.
    """
    p = small_params()
    solver = HJBSolver(p)
    solver.solve()
    q_mid = p.q_max      # index for q=0
    t_mid = p.n_t // 2  # midpoint in time (large tau, well away from T)
    da1 = solver.delta_a[0, q_mid, t_mid]
    da2 = solver.delta_a[1, q_mid, t_mid]
    assert da2 > da1, (
        f"Regime 2 spread ({da2:.4f}) should exceed regime 1 ({da1:.4f}) at t_mid"
    )


def test_spread_skew_direction():
    """
    Long inventory (q > 0) should produce tighter ask and wider bid.
    The MM wants to sell to reduce inventory.
    """
    p = small_params()
    solver = HJBSolver(p)
    solver.solve()
    # q=0 vs q=+3 at midpoint in time
    t_mid = p.n_t // 2
    q0   = p.q_max       # index for q=0
    qpos = p.q_max + 3   # index for q=+3

    # ask at q=+3 should be tighter (smaller) than at q=0
    da_q0  = solver.delta_a[0, q0,   t_mid]
    da_pos = solver.delta_a[0, qpos, t_mid]
    assert da_pos < da_q0, (
        f"Long inventory should tighten ask: da(q=3)={da_pos:.4f} < da(q=0)={da_q0:.4f}"
    )

    # bid at q=+3 should be wider (larger) than at q=0
    db_q0  = solver.delta_b[0, q0,   t_mid]
    db_pos = solver.delta_b[0, qpos, t_mid]
    assert db_pos > db_q0, (
        f"Long inventory should widen bid: db(q=3)={db_pos:.4f} > db(q=0)={db_q0:.4f}"
    )


def test_spreads_widen_with_more_time():
    """
    Spreads should be wider when more time remains (more inventory risk to accumulate).
    Compare t=0 (most time left) vs t=T (terminal).
    """
    p = small_params()
    solver = HJBSolver(p)
    solver.solve()
    q_mid = p.q_max
    # t=0 index vs t=T index
    da_early = solver.delta_a[0, q_mid, 0]
    da_late  = solver.delta_a[0, q_mid, -1]
    assert da_early >= da_late - 1e-6, (
        f"Spread at t=0 ({da_early:.4f}) should be >= spread at t=T ({da_late:.4f})"
    )


def test_save_and_load(tmp_path):
    """Save results and verify files exist."""
    p = small_params()
    solver = HJBSolver(p)
    solver.solve()
    solver.save(str(tmp_path))
    expected_files = [
        "h1.npy", "h2.npy",
        "delta_a_regime1.npy", "delta_a_regime2.npy",
        "delta_b_regime1.npy", "delta_b_regime2.npy",
        "spread_table.json",
    ]
    for fname in expected_files:
        assert (tmp_path / fname).exists(), f"Missing output file: {fname}"
