"""
optimal_spreads.py
==================
Loads the pre-solved spread tables from hjb_solver.py and provides a
clean interface for the simulator and HMM filter to query optimal spreads.

At runtime the simulator calls:
    spreads = SpreadTable.load("results/")
    da, db = spreads.lookup(q, t, regime)      # regime ∈ {0, 1}

For the HMM agent:
    da, db = spreads.lookup_blended(q, t, pi)  # pi = P(regime=2)
"""

import numpy as np
import os
from dataclasses import dataclass


@dataclass
class SpreadTable:
    q_grid:   np.ndarray   # shape (n_q,)  — inventory values
    t_grid:   np.ndarray   # shape (n_t+1,) — time values in [0, T]
    delta_a:  np.ndarray   # shape (2, n_q, n_t+1) — ask half-spreads
    delta_b:  np.ndarray   # shape (2, n_q, n_t+1) — bid half-spreads
    T:        float
    q_max:    int

    # ── Loaders ──────────────────────────────────────────────────────

    @classmethod
    def load(cls, results_dir: str = "results") -> "SpreadTable":
        """Load spread tables saved by hjb_solver.py."""
        da1 = np.load(os.path.join(results_dir, "delta_a_regime1.npy"))
        da2 = np.load(os.path.join(results_dir, "delta_a_regime2.npy"))
        db1 = np.load(os.path.join(results_dir, "delta_b_regime1.npy"))
        db2 = np.load(os.path.join(results_dir, "delta_b_regime2.npy"))

        n_q, n_t_plus1 = da1.shape
        q_max = (n_q - 1) // 2
        T = 1.0  # default; could be stored alongside npy files

        # Try to infer T from a metadata file if present
        meta_path = os.path.join(results_dir, "meta.npz")
        if os.path.exists(meta_path):
            meta = np.load(meta_path)
            T = float(meta["T"])

        q_grid = np.arange(-q_max, q_max + 1, dtype=float)
        t_grid = np.linspace(0, T, n_t_plus1)

        delta_a = np.stack([da1, da2], axis=0)  # (2, n_q, n_t+1)
        delta_b = np.stack([db1, db2], axis=0)

        print(f"Loaded spread table: q ∈ [{-q_max}, {q_max}], "
              f"{n_t_plus1} time steps, T={T}")
        return cls(q_grid, t_grid, delta_a, delta_b, T, q_max)

    # ── Core lookup ───────────────────────────────────────────────────

    def _q_idx(self, q: int) -> int:
        """Clamp q to grid and return index."""
        q_clamped = int(np.clip(q, -self.q_max, self.q_max))
        return q_clamped + self.q_max

    def _t_idx(self, t: float) -> int:
        """Find nearest time grid index for time t."""
        return int(np.searchsorted(self.t_grid, t, side="left"))

    def lookup(self, q: int, t: float, regime: int) -> tuple[float, float]:
        """
        Return (δᵃ*, δᵇ*) for a given inventory q, time t, and regime (0 or 1).

        Parameters
        ----------
        q      : current inventory
        t      : current time (in [0, T])
        regime : 0 = calm, 1 = chaotic

        Returns
        -------
        (delta_ask, delta_bid) : optimal half-spreads
        """
        qi = self._q_idx(q)
        ti = self._t_idx(t)
        da = float(self.delta_a[regime, qi, ti])
        db = float(self.delta_b[regime, qi, ti])
        return da, db

    def lookup_blended(self, q: int, t: float, pi: float) -> tuple[float, float]:
        """
        Return HMM-blended spreads: (1−π)·δ¹* + π·δ²*

        Parameters
        ----------
        q  : current inventory
        t  : current time
        pi : P(regime=2) from Wonham filter ∈ [0, 1]

        Returns
        -------
        (delta_ask_blended, delta_bid_blended)
        """
        da1, db1 = self.lookup(q, t, regime=0)
        da2, db2 = self.lookup(q, t, regime=1)
        da = (1.0 - pi) * da1 + pi * da2
        db = (1.0 - pi) * db1 + pi * db2
        return da, db

    # ── Closed-form fallback (no pre-solved table needed) ─────────────

    @staticmethod
    def closed_form(
        q: int,
        tau: float,
        sigma: float,
        gamma: float,
        kappa: float,
    ) -> tuple[float, float]:
        """
        Approximate optimal spreads from the A-S closed form.
        Used as a fallback or sanity check.

        δᵃ* = 1/κ + γσ²τ/2 − γσ²τ·q      (ask: tighten if long)
        δᵇ* = 1/κ + γσ²τ/2 + γσ²τ·q      (bid: tighten if short)

        Both clipped to minimum 1e-4.
        """
        base = 1.0 / kappa + 0.5 * gamma * sigma**2 * tau
        skew = gamma * sigma**2 * tau * q
        da = max(base - skew, 1e-4)
        db = max(base + skew, 1e-4)
        return da, db

    # ── Diagnostics ──────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a summary of spread behaviour across regimes and inventory."""
        print("\n=== Spread Table Summary ===")
        print(f"  q range : [{-self.q_max}, {self.q_max}]")
        print(f"  T       : {self.T}")
        print(f"  Shape   : {self.delta_a.shape}")

        t_mid = self.T / 2
        print(f"\n  Spreads at t = T/2 = {t_mid:.2f}, varying inventory:")
        print(f"  {'q':>5} | {'δᵃ(regime1)':>12} {'δᵇ(regime1)':>12} | "
              f"{'δᵃ(regime2)':>12} {'δᵇ(regime2)':>12}")
        print("  " + "-" * 65)
        for q in range(-self.q_max, self.q_max + 1, self.q_max // 4 or 1):
            da1, db1 = self.lookup(q, t_mid, 0)
            da2, db2 = self.lookup(q, t_mid, 1)
            print(f"  {q:>5} | {da1:>12.4f} {db1:>12.4f} | {da2:>12.4f} {db2:>12.4f}")
        print()


# ── Quick test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test closed-form without needing a solved table
    print("Closed-form spread check (σ=0.5, regime 1, T-t=0.5):")
    for q in [-5, -2, 0, 2, 5]:
        da, db = SpreadTable.closed_form(q, tau=0.5, sigma=0.5, gamma=0.1, kappa=1.5)
        print(f"  q={q:+3d}  δᵃ={da:.4f}  δᵇ={db:.4f}  ask=S+{da:.4f}  bid=S-{db:.4f}")

    print("\nClosed-form spread check (σ=2.0, regime 2, T-t=0.5):")
    for q in [-5, -2, 0, 2, 5]:
        da, db = SpreadTable.closed_form(q, tau=0.5, sigma=2.0, gamma=0.1, kappa=1.5)
        print(f"  q={q:+3d}  δᵃ={da:.4f}  δᵇ={db:.4f}  ask=S+{da:.4f}  bid=S-{db:.4f}")
