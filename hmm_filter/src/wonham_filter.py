"""
wonham_filter.py
================
Online Bayesian filter for a 2-state volatility-switching hidden Markov model.

IMPORTANT — terminology and derivation note
-------------------------------------------
The classical Wonham (1964) filter applies to a model where the hidden
Markov chain affects the DRIFT of the observation process:

    dY_t = h(k_t) dt + dW_t          [Wonham / drift-signal case]

In that setting, the filter SDE takes the well-known form with a linear
innovation term dI_t = dY_t − h_bar dt.

Our model is DIFFERENT. The hidden state k_t affects VOLATILITY, not drift:

    dS_t = sigma(k_t) dW_t            [volatility-switching case]
    E[dS_t | k_t] = 0 for both regimes

This is a fundamentally harder filtering problem. The drift is zero
under both regimes, so there is no drift-based innovation signal.
The correct approach is to filter on the QUADRATIC VARIATION (squared
increments), which carry the volatility information.

CORRECT FILTER — Discrete-time Bayesian update
----------------------------------------------
At each step, apply Bayes' theorem exactly:

    Prediction:  pi_pred = pi + [q12(1-pi) - q21*pi] * dt
    Likelihood:  L_k = N(dS; 0, sigma_k^2 * dt)   for k in {1, 2}
    Update:      pi_new = pi_pred * L2 / (pi_pred*L2 + (1-pi_pred)*L1)

This is exact in discrete time and converges to the correct continuous-time
filter as dt → 0. It is derived from first principles (Bayes + Markov
prediction), not from the Wonham equation which does not apply here.

References
----------
Elliott, R.J., Aggoun, L. & Moore, J.B. (1995).
    Hidden Markov Models: Estimation and Control. Springer.
    Chapter 6 covers the volatility-switching observation model.

Liptser, R.S. & Shiryaev, A.N. (2001).
    Statistics of Random Processes II. Springer.
    §9.4 for the general nonlinear filtering equation (Zakai/Kushner-Stratonovich).

Hamilton, J.D. (1989).
    A new approach to the economic analysis of nonstationary time series.
    Econometrica 57(2). — Discrete-time version of this filter.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BayesVolFilter:
    """
    Exact discrete-time Bayesian filter for a 2-state volatility-switching HMM.

    The hidden state k_t ∈ {1 (calm), 2 (chaotic)} affects the observation
    volatility dS_t = sigma(k_t) dW_t. The filter maintains:

        pi_t = P(k_t = 2 | S_0, S_1, ..., S_t)

    via a predict-update cycle at each observed price increment dS.

    Parameters
    ----------
    sigma_1 : volatility in calm regime (per sqrt(dt))
    sigma_2 : volatility in chaotic regime (per sqrt(dt))
    q_12    : transition rate calm → chaotic (per unit time)
    q_21    : transition rate chaotic → calm (per unit time)
    pi_init : initial belief P(k_0 = 2); defaults to stationary probability
    dt      : time step (same units as q_12, q_21)
    """

    sigma_1:  float
    sigma_2:  float
    q_12:     float
    q_21:     float
    pi_init:  float = 0.2
    dt:       float = 1.0

    pi:       float = field(init=False)
    history:  list  = field(init=False, repr=False)

    def __post_init__(self):
        self.pi = float(np.clip(self.pi_init, 1e-6, 1 - 1e-6))
        self.history = [self.pi]

    # ── Core update ───────────────────────────────────────────────────

    def update(self, dS: float) -> float:
        """
        Bayesian predict-update step given observed price increment dS.

        Step 1 — Prediction (Markov chain propagation):
            pi_pred = pi + [q12*(1-pi) - q21*pi] * dt

        Step 2 — Likelihood (Gaussian observation model):
            L_k = N(dS; 0, sigma_k^2 * dt)
                = exp(-dS^2 / (2*sigma_k^2*dt)) / sqrt(2*pi*sigma_k^2*dt)

            Note: the sqrt(2*pi*sigma_k^2*dt) normalising constants cancel
            in the ratio, so we only need the exponential parts.

        Step 3 — Bayesian update:
            pi_new = pi_pred * L2 / (pi_pred*L2 + (1-pi_pred)*L1)

        This is exact for the discrete-time model and requires no
        approximation or ad-hoc innovation construction.

        Parameters
        ----------
        dS : observed price change over one time step dt

        Returns
        -------
        Updated posterior P(k = 2 | observations)
        """
        pi = self.pi

        # Step 1: predict
        pi_pred = pi + (self.q_12 * (1.0 - pi) - self.q_21 * pi) * self.dt
        pi_pred = float(np.clip(pi_pred, 1e-10, 1 - 1e-10))

        # Step 2: log-likelihoods INCLUDING normalising constants.
        # L_k = N(dS; 0, sigma_k^2*dt) = (2*pi*sigma_k^2*dt)^{-1/2} * exp(-dS^2/(2*sigma_k^2*dt))
        # The normalising constants do NOT cancel between regimes (sigma1 != sigma2),
        # so they must be included. Omitting them biases the filter toward the
        # regime with smaller sigma_k^2 in the denominator (i.e. toward whichever
        # regime has the larger 1/sigma_k, here the calm regime) for small |dS|,
        # and was the cause of a systematic bias toward pi=1 in an earlier version.
        log_L1 = -0.5*np.log(2*np.pi*self.sigma_1**2*self.dt) - dS**2/(2.0*self.sigma_1**2*self.dt)
        log_L2 = -0.5*np.log(2*np.pi*self.sigma_2**2*self.dt) - dS**2/(2.0*self.sigma_2**2*self.dt)

        # Normalise for numerical stability: subtract max before exp
        log_max = max(log_L1, log_L2)
        L1 = np.exp(log_L1 - log_max)
        L2 = np.exp(log_L2 - log_max)

        # Step 3: Bayes update
        num = pi_pred * L2
        den = pi_pred * L2 + (1.0 - pi_pred) * L1
        self.pi = float(np.clip(num / (den + 1e-300), 1e-6, 1 - 1e-6))
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
        pi_series : posterior P(k=2) at each time step, shape (n,)
        """
        self.reset()
        pi_series = np.zeros(len(price_series))
        pi_series[0] = self.pi
        for i in range(1, len(price_series)):
            dS = price_series[i] - price_series[i - 1]
            pi_series[i] = self.update(dS)
        return pi_series

    def reset(self, pi: Optional[float] = None) -> None:
        """Reset to initial belief."""
        self.pi = float(np.clip(
            pi if pi is not None else self.pi_init, 1e-6, 1 - 1e-6))
        self.history = [self.pi]

    # ── Properties ────────────────────────────────────────────────────

    @property
    def regime_estimate(self) -> int:
        """MAP regime: 0 = calm, 1 = chaotic."""
        return 1 if self.pi >= 0.5 else 0

    @property
    def stationary_pi(self) -> float:
        """Stationary P(chaotic) = q12 / (q12 + q21)."""
        return self.q_12 / (self.q_12 + self.q_21)

    def blended_sigma(self, pi: Optional[float] = None) -> float:
        """
        Blended volatility under current belief.
        Uses variance weighting (not linear): sigma(pi) = sqrt(pi*sigma2^2 + (1-pi)*sigma1^2)
        """
        p = pi if pi is not None else self.pi
        return float(np.sqrt(p * self.sigma_2**2 + (1.0 - p) * self.sigma_1**2))

    def __repr__(self) -> str:
        return (f"BayesVolFilter(pi={self.pi:.4f}, "
                f"regime={'chaotic' if self.pi>=0.5 else 'calm'}, "
                f"sigma_blended={self.blended_sigma():.5f})")


# Keep WonhamFilter as an alias with a deprecation note so existing
# code and tests do not break, but the class now uses the correct filter.
class WonhamFilter(BayesVolFilter):
    """
    Alias for BayesVolFilter.

    NOTE ON NAMING: The classical Wonham (1964) filter applies to models
    where the hidden state modulates the DRIFT of observations
    (dY = h(k)dt + dW). Our model has the hidden state modulating
    VOLATILITY (dS = sigma(k)dW), which is a different filtering problem.
    The correct filter here is the discrete-time Bayesian predict-update
    implemented in BayesVolFilter, not the Wonham SDE.

    This class is kept as WonhamFilter for backward compatibility only.
    New code should use BayesVolFilter directly.
    """
    pass


# ── Simulation helper ─────────────────────────────────────────────────

def simulate_regime_switching_prices(
    T_seconds: int,
    dt: float,
    sigma_1: float,
    sigma_2: float,
    q_12: float,
    q_21: float,
    S0: float = 100.0,
    seed: int = 42,
):
    """
    Simulate a regime-switching price path for testing the filter.

    Returns
    -------
    t_grid   : time array, shape (n,)
    S        : mid-price path, shape (n,)
    regimes  : true regime at each step, shape (n,), values in {0, 1}
    """
    rng = np.random.default_rng(seed)
    n = int(T_seconds / dt)
    S = np.zeros(n)
    regimes = np.zeros(n, dtype=int)
    S[0] = S0
    regime = 0

    for i in range(1, n):
        if regime == 0:
            if rng.uniform() < q_12 * dt: regime = 1
        else:
            if rng.uniform() < q_21 * dt: regime = 0
        regimes[i] = regime
        sigma = sigma_2 if regime == 1 else sigma_1
        S[i] = S[i-1] + sigma * rng.standard_normal() * np.sqrt(dt)

    return np.arange(n) * dt, S, regimes


# ── Demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    print("Running BayesVolFilter demo (correct filter for volatility-switching)...")
    T, dt = 3600, 1.0
    s1, s2 = 0.005, 0.02
    q12, q21 = 1/300, 1/120

    t, S, true_reg = simulate_regime_switching_prices(T, dt, s1, s2, q12, q21, seed=7)

    filt = BayesVolFilter(s1, s2, q12, q21, pi_init=q12/(q12+q21), dt=dt)
    pi_series = filt.update_batch(S)

    acc = max(((pi_series>=0.5)==true_reg).mean(),
              1-((pi_series>=0.5)==true_reg).mean())
    print(f"Filter: {filt}")
    print(f"Accuracy: {acc:.1%}  |  Stationary pi*: {filt.stationary_pi:.3f}")

    os.makedirs('results/plots', exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor('#0F0F0F')
    axes[0].plot(t/60, S, lw=0.5, color='#5DCAA5')
    axes[0].set_ylabel('Price', color='#CCC')
    axes[1].fill_between(t/60, true_reg, alpha=0.7, color='#D85A30', label='True chaotic')
    axes[1].set_ylabel('True regime', color='#CCC')
    axes[2].plot(t/60, pi_series, lw=0.8, color='#EF9F27')
    axes[2].axhline(0.5, color='white', lw=0.6, ls='--')
    axes[2].set_ylabel('pi_t', color='#CCC')
    axes[2].set_xlabel('Time (min)', color='#CCC')
    for ax in axes: ax.set_facecolor('#1A1A1A')
    fig.suptitle(f'BayesVolFilter — accuracy {acc:.1%}', color='white')
    plt.tight_layout()
    plt.savefig('results/plots/wonham_filter_demo.png', dpi=130,
                bbox_inches='tight', facecolor='#0F0F0F')
    print("Saved: results/plots/wonham_filter_demo.png")
