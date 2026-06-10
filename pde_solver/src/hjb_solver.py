"""
hjb_solver.py
=============
Solves the coupled Hamilton-Jacobi-Bellman (HJB) PDE system for the
regime-switching market making problem using a Crank-Nicolson finite
difference scheme.

The system (for k=1,2):

  ∂Vᵏ/∂t + ½σₖ² ∂²Vᵏ/∂S²
    + max_{δᵃ} { λₖᵃ(δᵃ) · [Vᵏ(q−1) − Vᵏ(q) + δᵃ] }
    + max_{δᵇ} { λₖᵇ(δᵇ) · [Vᵏ(q+1) − Vᵏ(q) + δᵇ] }
    + Σⱼ≠ₖ qₖⱼ · (Vʲ − Vᵏ) = 0

After the change of variables Vᵏ(x,q,S,t) = x + qS + hᵏ(q,t), the
S-dependence drops out and we solve for hᵏ(q,t) on a grid over q and t.

Usage:
    python hjb_solver.py                  # uses configs/default.yaml
    python hjb_solver.py --config path    # custom config
"""

import numpy as np
import yaml
import argparse
import os
import json
from dataclasses import dataclass
from typing import Tuple


# ─────────────────────────────────────────────
#  Parameters
# ─────────────────────────────────────────────

@dataclass
class Params:
    # Regimes
    sigma: Tuple[float, float]   # (σ₁, σ₂)
    q_switch: Tuple[float, float] # (q₁₂, q₂₁) transition rates
    # Order flow
    A: Tuple[float, float]        # (A₁, A₂) baseline arrival rates
    kappa: float                  # spread sensitivity
    # Agent
    gamma: float                  # risk aversion
    phi: float                    # terminal inventory penalty
    q_max: int                    # maximum |inventory|
    # Time
    T: float                      # horizon
    n_t: int                      # number of time steps

    @classmethod
    def from_yaml(cls, path: str) -> "Params":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        r = cfg["regime"]
        o = cfg["order_flow"]
        a = cfg["agent"]
        t = cfg["time"]
        p = cfg["pde"]
        return cls(
            sigma=(r["sigma_1"], r["sigma_2"]),
            q_switch=(r["q_12"], r["q_21"]),
            A=(o["A_1"], o["A_2"]),
            kappa=o["kappa"],
            gamma=a["gamma"],
            phi=a["phi"],
            q_max=a["q_max"],
            T=t["T"],
            n_t=int(t["T"] / t["dt_pde"]),
        )


# ─────────────────────────────────────────────
#  Optimal spread helper
# ─────────────────────────────────────────────

def optimal_spread_and_rate(
    delta_V: np.ndarray,  # shape (n_q,) — value difference for one side
    A_k: float,
    kappa: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each inventory level, compute optimal half-spread δ* and the
    corresponding arrival rate λ*(δ*).

    First-order condition of:  f(δ) = A · e^{−κδ} · (δ + C)
    where C = h(q−1)−h(q) + γσₖ²qτ − ½γσₖ²τ  (passed in as delta_V)
    Setting df/dδ = 0:
        −κ(δ* + C) + 1 = 0  ⟹  δ* = 1/κ − C

    If δ* < 0, we clip to a minimum spread (e.g. 1 tick = 0.01).
    """
    delta_opt = np.maximum(1.0 / kappa - delta_V, 1e-4)
    lambda_opt = A_k * np.exp(-kappa * delta_opt)
    return delta_opt, lambda_opt


# ─────────────────────────────────────────────
#  Main solver
# ─────────────────────────────────────────────

class HJBSolver:
    """
    Solves for h¹(q,t) and h²(q,t) backward in time.

    The full value function is reconstructed as:
        Vᵏ(x, q, S, t) = x + qS − ½γσₖ²q²(T−t) + hᵏ(q,t)
    """

    def __init__(self, params: Params):
        self.p = params
        n_q = 2 * params.q_max + 1         # grid: q ∈ {-q_max, ..., q_max}
        self.n_q = n_q
        self.q_grid = np.arange(-params.q_max, params.q_max + 1, dtype=float)
        self.dt = params.T / params.n_t

        # Storage: h[k, q_idx, t_idx]  (k=0 → regime 1, k=1 → regime 2)
        self.h = np.zeros((2, n_q, params.n_t + 1))

        # Storage for optimal spreads
        self.delta_a = np.zeros((2, n_q, params.n_t + 1))  # ask
        self.delta_b = np.zeros((2, n_q, params.n_t + 1))  # bid

        # Terminal condition: hᵏ(q, T) = −(φ/2)q²
        # (the x + qS part is handled separately; only the penalty remains)
        terminal = -0.5 * params.phi * self.q_grid ** 2
        self.h[0, :, -1] = terminal
        self.h[1, :, -1] = terminal

    def _jump_terms(
        self, h_k: np.ndarray, k: int, tau: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the optimal jump contributions and store δ*.

        tau = T − t (time remaining)

        Returns:
            jump_a  — ask jump value improvement (n_q,)
            jump_b  — bid jump value improvement (n_q,)
            da      — optimal ask spreads (n_q,)
            db      — optimal bid spreads (n_q,)
        """
        p = self.p
        A_k = p.A[k]
        gamma_k = p.gamma * p.sigma[k] ** 2

        # Exact ΔVᵃ from ansatz substitution (see docs/math_derivation.md §8):
        #   ΔVᵃ(q) = δᵃ + h(q−1) − h(q) + γσₖ²·q·τ − ½γσₖ²·τ
        # So C(q) = h(q−1) − h(q) + γσₖ²·q·τ − ½γσₖ²·τ
        # and δᵃ* = 1/κ − C(q)
        #
        # Note sign: +γσₖ²·q·τ (positive), so long inventory (q>0) increases C,
        # which DECREASES δᵃ* (tightens ask) — correct: MM wants to sell.
        # Previous version had −γσₖ²·q·τ which inverted the skew direction.
        dV_a = np.zeros(self.n_q)
        dV_b = np.zeros(self.n_q)

        # Interior points — exact formula from FOC derivation
        dV_a[1:] = (h_k[:-1] - h_k[1:]
                    + gamma_k * self.q_grid[1:] * tau
                    - 0.5 * gamma_k * tau)
        dV_b[:-1] = (h_k[1:] - h_k[:-1]
                     - gamma_k * self.q_grid[:-1] * tau
                     - 0.5 * gamma_k * tau)

        # Boundary: cannot sell if at -q_max, cannot buy if at +q_max
        dV_a[0] = 0.0   # no ask at minimum inventory
        dV_b[-1] = 0.0  # no bid at maximum inventory

        da, lambda_a = optimal_spread_and_rate(dV_a, A_k, p.kappa)
        db, lambda_b = optimal_spread_and_rate(dV_b, A_k, p.kappa)

        # Jump contributions to PDE right-hand side
        jump_a = lambda_a * (dV_a + da)
        jump_b = lambda_b * (dV_b + db)

        # Zero out boundaries
        jump_a[0] = 0.0
        jump_b[-1] = 0.0

        return jump_a, jump_b, da, db

    def solve(self) -> None:
        """
        Backward induction from t=T to t=0.
        At each step we update h¹ and h² simultaneously (coupled via q_kj terms).
        """
        p = self.p
        q12, q21 = p.q_switch

        print(f"Solving HJB: {p.n_t} time steps, {self.n_q} inventory levels, 2 regimes")
        print(f"  σ = ({p.sigma[0]}, {p.sigma[1]}), γ = {p.gamma}, κ = {p.kappa}")
        print(f"  Transition rates: q₁₂ = {q12}, q₂₁ = {q21}")

        for i in range(p.n_t - 1, -1, -1):
            t_idx = i + 1          # current "future" index (already solved)
            t_next = i             # index we're solving for
            tau = (p.n_t - i) * self.dt    # time remaining at step i

            h1_curr = self.h[0, :, t_idx]
            h2_curr = self.h[1, :, t_idx]

            # Jump terms for each regime
            ja1, jb1, da1, db1 = self._jump_terms(h1_curr, 0, tau)
            ja2, jb2, da2, db2 = self._jump_terms(h2_curr, 1, tau)

            # Coupling terms: regime 1 is pulled toward regime 2 at rate q12
            coupling1 = q12 * (h2_curr - h1_curr)
            coupling2 = q21 * (h1_curr - h2_curr)

            # Explicit Euler update (Crank-Nicolson refinement applied below)
            # ∂h/∂t = jump_a + jump_b + coupling
            rhs1 = ja1 + jb1 + coupling1
            rhs2 = ja2 + jb2 + coupling2

            h1_new = h1_curr + self.dt * rhs1
            h2_new = h2_curr + self.dt * rhs2

            # ── Crank-Nicolson correction (one iteration) ──────────────
            # Recompute jump terms at new estimates, average with old
            ja1_new, jb1_new, da1, db1 = self._jump_terms(h1_new, 0, tau - self.dt)
            ja2_new, jb2_new, da2, db2 = self._jump_terms(h2_new, 1, tau - self.dt)

            coupling1_new = q12 * (h2_new - h1_new)
            coupling2_new = q21 * (h1_new - h2_new)

            rhs1_new = ja1_new + jb1_new + coupling1_new
            rhs2_new = ja2_new + jb2_new + coupling2_new

            # Averaged (CN) update
            self.h[0, :, t_next] = h1_curr + 0.5 * self.dt * (rhs1 + rhs1_new)
            self.h[1, :, t_next] = h2_curr + 0.5 * self.dt * (rhs2 + rhs2_new)

            # Store optimal spreads
            self.delta_a[0, :, t_next] = da1
            self.delta_a[1, :, t_next] = da2
            self.delta_b[0, :, t_next] = db1
            self.delta_b[1, :, t_next] = db2

            if i % (p.n_t // 10) == 0:
                print(f"  t-step {p.n_t - i}/{p.n_t}  |  h¹[q=0] = {self.h[0, p.q_max, t_next]:.4f}"
                      f"  h²[q=0] = {self.h[1, p.q_max, t_next]:.4f}")

        print("Solve complete.")

    def get_spread_table(self) -> dict:
        """
        Returns a dict of spread tables for use by the simulator.
        Tables indexed by [regime, q_idx, t_idx].
        """
        return {
            "q_grid": self.q_grid.tolist(),
            "n_t": self.p.n_t,
            "T": self.p.T,
            "delta_a_regime1": self.delta_a[0].tolist(),
            "delta_a_regime2": self.delta_a[1].tolist(),
            "delta_b_regime1": self.delta_b[0].tolist(),
            "delta_b_regime2": self.delta_b[1].tolist(),
        }

    def save(self, out_dir: str = "results") -> None:
        """Save h arrays and spread tables to disk."""
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "h1.npy"), self.h[0])
        np.save(os.path.join(out_dir, "h2.npy"), self.h[1])
        np.save(os.path.join(out_dir, "delta_a_regime1.npy"), self.delta_a[0])
        np.save(os.path.join(out_dir, "delta_a_regime2.npy"), self.delta_a[1])
        np.save(os.path.join(out_dir, "delta_b_regime1.npy"), self.delta_b[0])
        np.save(os.path.join(out_dir, "delta_b_regime2.npy"), self.delta_b[1])
        # Also save as JSON for the C++ simulator to read
        with open(os.path.join(out_dir, "spread_table.json"), "w") as f:
            json.dump(self.get_spread_table(), f)
        print(f"Saved results to {out_dir}/")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Solve coupled HJB PDEs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    params = Params.from_yaml(args.config)
    solver = HJBSolver(params)
    solver.solve()
    solver.save(args.out)


if __name__ == "__main__":
    main()
