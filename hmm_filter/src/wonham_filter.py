"""
wonham_filter.py
================
Implements the continuous-time Wonham filter for online regime inference.

The filter maintains the posterior belief:
    π_t = P(k_t = 2 | observations up to t)

It updates using two sources of information:
  1. Deterministic drift from the Markov chain dynamics
  2. Stochastic Bayesian update from observed price innovations

The filter SDE (Wonham, 1964) for a volatility-switching model:
    dπ_t = [q₁₂(1−π_t) − q₂₁·π_t] dt
           + π_t(1−π_t) · [(σ₂²−σ₁²) / (2·σ²(π_t))] · innovation_t dt

where the blended variance (not standard deviation) enters:
    σ²(π_t) = π_t·σ₂² + (1−π_t)·σ₁²   (blended variance — correct for vol switching)
    σ(π_t)  = √σ²(π_t)                  (blended vol, for normalisation only)

NOTE: For a volatility-switching model, the observation variance is
π_t·σ₂² + (1−π_t)·σ₁², not the square of the linearly blended σ.
The linear blend σ(π) = π·σ₂ + (1−π)·σ₁ is an approximation used in some
textbook treatments (Liptser & Shiryaev) but is not exact for this model.
We implement the chi-squared innovation form which is exact.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WonhamFilter:
    """
    Online Wonham HMM filter for 2-regime volatility.

    Parameters
    ----------
    sigma_1 : volatility in calm regime
    sigma_2 : volatility in chaotic regime
    q_12    : transition rate calm → chaotic
    q_21    : transition rate chaotic → calm
    pi_init : initial belief P(regime=2)
    dt      : simulation time step (seconds)
    """

    sigma_1:  float
    sigma_2:  float
    q_12:     float
    q_21:     float
    pi_init:  float = 0.2
    dt:       float = 1.0    # seconds

    # Runtime state (post-init)
    pi:       float = field(init=False)
    history:  list  = field(init=False, repr=False)

    def __post_init__(self):
        self.pi = float(np.clip(self.pi_init, 1e-6, 1 - 1e-6))
        self.history = [self.pi]

    # ── Core update ───────────────────────────────────────────────────

    def blended_sigma(self, pi: Optional[float] = None) -> float:
        """
        Blended volatility for a variance-switching model.

        The correct blending for a two-state volatility model is over variance:
            σ²(π) = π·σ₂² + (1−π)·σ₁²   →   σ(π) = √(π·σ₂² + (1−π)·σ₁²)

        The linear blend π·σ₂ + (1−π)·σ₁ is an approximation that understates
        the blended vol whenever σ₁ ≠ σ₂ (by Jensen's inequality, √E[σ²] ≥ E[σ]).
        We use the variance-correct form throughout.
        """
        p = pi if pi is not None else self.pi
        blended_var = p * self.sigma_2**2 + (1.0 - p) * self.sigma_1**2
        return float(np.sqrt(blended_var))

    def update(self, dS: float) -> float:
        """
        Update belief given an observed price move dS over time step dt.

        For a volatility-switching model (dS = σ_k dW), the natural sufficient
        statistic for distinguishing regimes is dS², not dS (price moves have
        zero mean under both regimes, so the sign carries no information).

        We use the chi-squared innovation form, which is exact for this model:
          innovation = dS²/(σ²(π)·dt) − 1

        Expected value = 0 under current belief π (no surprise on average).
        Positive when |dS| > σ(π)·√dt → evidence for chaotic regime → π rises.
        Negative when |dS| < σ(π)·√dt → evidence for calm regime → π falls.

        The gain term π(1−π)·(σ₂²−σ₁²)/(2·σ²(π)) is the likelihood-ratio
        sensitivity — how much a unit innovation should update the belief.
        This uses variance differences σ₂²−σ₁² (not σ₂−σ₁), which is the
        correct form for a variance-switching observation model.

        Parameters
        ----------
        dS : observed change in mid-price over dt

        Returns
        -------
        Updated π_t
        """
        pi = self.pi
        sigma_b = self.blended_sigma(pi)

        # Chi-squared innovation: (observed variance / expected variance) - 1
        # Positive when |dS| larger than expected → bullish on chaotic regime
        realized_var  = dS ** 2
        expected_var  = sigma_b ** 2 * self.dt
        innovation    = realized_var / (expected_var + 1e-30) - 1.0

        # Gain: how much the chi-sq innovation updates the belief
        # Derived from the likelihood ratio between regime 2 and regime 1
        sigma_ratio   = (self.sigma_2 ** 2 - self.sigma_1 ** 2) / (2.0 * sigma_b ** 2 + 1e-30)
        gain          = pi * (1.0 - pi) * sigma_ratio

        # Wonham SDE — Euler-Maruyama
        drift         = (self.q_12 * (1.0 - pi) - self.q_21 * pi) * self.dt
        diffusion     = gain * innovation * self.dt

        pi_new = pi + drift + diffusion
        self.pi = float(np.clip(pi_new, 1e-6, 1 - 1e-6))
        self.history.append(self.pi)
        return self.pi

    def update_batch(self, price_series: np.ndarray) -> np.ndarray:
        """
        Run the filter over a full price series.

        Parameters
        ----------
        price_series : array of mid-prices, shape (n,)

        Returns
        -------
        pi_series : posterior belief at each step, shape (n,)
        """
        self.reset()
        pi_series = np.zeros(len(price_series))
        pi_series[0] = self.pi
        for i in range(1, len(price_series)):
            dS = price_series[i] - price_series[i - 1]
            pi_series[i] = self.update(dS)
        return pi_series

    def reset(self, pi: Optional[float] = None) -> None:
        """Reset belief to initial value (or given pi)."""
        self.pi = float(np.clip(pi if pi is not None else self.pi_init, 1e-6, 1 - 1e-6))
        self.history = [self.pi]

    # ── Properties ───────────────────────────────────────────────────

    @property
    def regime_estimate(self) -> int:
        """MAP regime estimate: 0 (calm) or 1 (chaotic)."""
        return 1 if self.pi >= 0.5 else 0

    @property
    def stationary_pi(self) -> float:
        """Stationary probability of regime 2 = q₁₂ / (q₁₂ + q₂₁)."""
        return self.q_12 / (self.q_12 + self.q_21)

    # ── Diagnostics ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"WonhamFilter(π={self.pi:.4f}, "
            f"regime={'chaotic' if self.pi >= 0.5 else 'calm'}, "
            f"σ_blended={self.blended_sigma():.4f})"
        )


# ── Simulation helper for testing ─────────────────────────────────────

def simulate_regime_switching_prices(
    T_seconds: int,
    dt: float,
    sigma_1: float,
    sigma_2: float,
    q_12: float,
    q_21: float,
    S0: float = 100.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a regime-switching price path for testing the filter.

    Returns
    -------
    t_grid   : time array
    S        : mid-price path
    regimes  : true regime at each step (0 or 1)
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T_seconds / dt)
    S = np.zeros(n_steps)
    regimes = np.zeros(n_steps, dtype=int)

    S[0] = S0
    regime = 0  # start calm

    for i in range(1, n_steps):
        # Regime switching
        if regime == 0:
            if rng.uniform() < q_12 * dt:
                regime = 1
        else:
            if rng.uniform() < q_21 * dt:
                regime = 0
        regimes[i] = regime

        # Price move
        sigma_k = sigma_2 if regime == 1 else sigma_1
        dW = rng.normal(0, np.sqrt(dt))
        S[i] = S[i - 1] + sigma_k * dW

    t_grid = np.arange(n_steps) * dt
    return t_grid, S, regimes


# ── Entry point / demo ────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Simulate a 1-hour price path (3600 seconds, dt=1s)
    T = 3600
    dt = 1.0
    print("Simulating regime-switching price path...")
    t, S, true_regimes = simulate_regime_switching_prices(
        T_seconds=T, dt=dt,
        sigma_1=0.005, sigma_2=0.02,
        q_12=1/600, q_21=1/150,   # avg 10 min calm, 2.5 min chaotic
        seed=42,
    )

    # Run Wonham filter
    filt = WonhamFilter(
        sigma_1=0.005, sigma_2=0.02,
        q_12=1/600, q_21=1/150,
        pi_init=1/5, dt=dt,
    )
    pi_series = filt.update_batch(S)

    print(f"Filter final state: {filt}")
    print(f"Stationary π*: {filt.stationary_pi:.3f}")
    print(f"Mean π: {pi_series.mean():.3f}  (should be ~{filt.stationary_pi:.3f})")

    # Accuracy: how often does MAP regime match true regime?
    predicted = (pi_series >= 0.5).astype(int)
    accuracy = (predicted == true_regimes).mean()
    print(f"Regime classification accuracy: {accuracy:.1%}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(t / 60, S, lw=0.6, color="#185FA5", label="Mid-price")
    axes[0].set_ylabel("Price S_t")
    axes[0].set_title("Wonham Filter — Regime Inference from Price Path")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(t / 60, true_regimes, alpha=0.4, color="#D85A30",
                         label="True regime 2 (chaotic)")
    axes[1].set_ylabel("True regime")
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["calm", "chaotic"])
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(t / 60, pi_series, lw=0.8, color="#5DCAA5", label="π_t = P(chaotic)")
    axes[2].axhline(0.5, color="gray", lw=0.8, ls="--", label="Decision threshold")
    axes[2].axhline(filt.stationary_pi, color="#EF9F27", lw=0.8, ls=":",
                    label=f"Stationary π* = {filt.stationary_pi:.2f}")
    axes[2].set_xlabel("Time (minutes)")
    axes[2].set_ylabel("π_t")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/wonham_filter_demo.png", dpi=150, bbox_inches="tight")
    print("Plot saved: results/plots/wonham_filter_demo.png")
    plt.show()
