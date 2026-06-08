"""
Tests for wonham_filter.py
Run: python -m pytest hmm_filter/tests/test_wonham.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np
import pytest
from wonham_filter import WonhamFilter, simulate_regime_switching_prices


# ── Basic properties ──────────────────────────────────────────────────

def make_filter(**kwargs):
    defaults = dict(sigma_1=0.5, sigma_2=2.0, q_12=0.5, q_21=2.0, pi_init=0.2, dt=1.0)
    defaults.update(kwargs)
    return WonhamFilter(**defaults)


def test_initial_belief():
    f = make_filter(pi_init=0.3)
    assert abs(f.pi - 0.3) < 1e-10


def test_belief_stays_in_01():
    """After many updates, π must stay in (0, 1)."""
    f = make_filter()
    rng = np.random.default_rng(0)
    for _ in range(10000):
        dS = rng.normal(0, 0.02)
        f.update(dS)
    assert 0 < f.pi < 1


def test_large_moves_increase_pi():
    """
    Price moves consistently larger than what calm regime predicts should
    push π upward on average. We feed random shocks scaled to σ₂ (chaotic)
    and verify that after many steps the mean π exceeds the initial value.
    """
    rng = np.random.default_rng(42)
    f = make_filter(pi_init=0.05, sigma_1=0.5, sigma_2=2.0, dt=1.0)
    # Draw shocks from N(0, σ₂²) — consistent with chaotic regime
    shocks = rng.normal(0, f.sigma_2 * np.sqrt(f.dt), size=500)
    pi_history = []
    for dS in shocks:
        pi_history.append(f.update(dS))
    mean_pi = np.mean(pi_history[100:])  # skip early transient
    assert mean_pi > 0.05, f"Chaotic-scale shocks should raise mean π above 0.05, got {mean_pi:.4f}"


def test_small_moves_decrease_pi():
    """Repeatedly tiny price moves should pull π down (toward calm)."""
    f = make_filter(pi_init=0.9)
    for _ in range(200):
        f.update(0.0001)
    assert f.pi < 0.9


def test_reset():
    f = make_filter(pi_init=0.2)
    for _ in range(100):
        f.update(0.5)
    f.reset()
    assert abs(f.pi - 0.2) < 1e-10
    assert len(f.history) == 1


def test_stationary_pi():
    f = make_filter(q_12=0.5, q_21=2.0)
    expected = 0.5 / (0.5 + 2.0)
    assert abs(f.stationary_pi - expected) < 1e-10


def test_blended_sigma():
    f = make_filter(sigma_1=0.5, sigma_2=2.0, pi_init=0.4)
    expected = 0.4 * 2.0 + 0.6 * 0.5
    assert abs(f.blended_sigma() - expected) < 1e-10


def test_batch_update_shape():
    f = make_filter()
    prices = np.cumsum(np.random.default_rng(1).normal(0, 0.01, 500)) + 100
    pi_series = f.update_batch(prices)
    assert pi_series.shape == (500,)
    assert np.all((pi_series > 0) & (pi_series < 1))


def test_regime_estimate():
    f = make_filter(pi_init=0.3)
    assert f.regime_estimate == 0   # calm
    f.pi = 0.7
    assert f.regime_estimate == 1   # chaotic


# ── Statistical accuracy ──────────────────────────────────────────────

def test_filter_mean_close_to_stationary():
    """
    Over a long simulation, mean π should approximate the stationary probability.
    """
    T, dt = 100000, 1.0
    q_12, q_21 = 1/600, 1/150
    _, S, _ = simulate_regime_switching_prices(
        T_seconds=T, dt=dt,
        sigma_1=0.005, sigma_2=0.02,
        q_12=q_12, q_21=q_21,
        seed=7,
    )
    f = WonhamFilter(sigma_1=0.005, sigma_2=0.02, q_12=q_12, q_21=q_21,
                     pi_init=q_12/(q_12+q_21), dt=dt)
    pi_series = f.update_batch(S)
    mean_pi = pi_series[1000:].mean()   # skip burn-in
    stationary = q_12 / (q_12 + q_21)
    assert abs(mean_pi - stationary) < 0.05, (
        f"Mean π={mean_pi:.3f} not close to stationary π*={stationary:.3f}"
    )
