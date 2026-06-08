"""
visualize_pde.py
================
Loads solved HJB results and generates publication-quality plots:

  1. Value function h(q, t) for both regimes
  2. Optimal ask/bid spread surfaces δ*(q, t) per regime
  3. Spread skew vs inventory at fixed times
  4. Regime comparison: spread widening in chaotic vs calm

Run after hjb_solver.py has saved results to results/.
    python pde_solver/src/visualize_pde.py
"""

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

RESULTS_DIR = "results"
PLOTS_DIR   = "results/plots"


def load_results(results_dir: str) -> dict:
    """Load all numpy arrays saved by hjb_solver."""
    files = {
        "h1":    "h1.npy",
        "h2":    "h2.npy",
        "da1":   "delta_a_regime1.npy",
        "da2":   "delta_a_regime2.npy",
        "db1":   "delta_b_regime1.npy",
        "db2":   "delta_b_regime2.npy",
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}. Run hjb_solver.py first.")
        data[key] = np.load(path)

    n_q, n_t = data["h1"].shape
    q_max = (n_q - 1) // 2
    data["q_grid"] = np.arange(-q_max, q_max + 1, dtype=float)
    data["t_grid"] = np.linspace(0, 1.0, n_t)
    data["q_max"]  = q_max
    return data


# ── Plot 1: Value function surfaces ───────────────────────────────────

def plot_value_functions(data: dict, save_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), subplot_kw={"projection": "3d"})
    T_mat, Q_mat = np.meshgrid(data["t_grid"], data["q_grid"])

    for ax, h, title, cmap in zip(
        axes,
        [data["h1"], data["h2"]],
        ["h¹(q,t) — Calm regime (σ₁=0.5)", "h²(q,t) — Chaotic regime (σ₂=2.0)"],
        [cm.viridis, cm.plasma],
    ):
        surf = ax.plot_surface(T_mat, Q_mat, h, cmap=cmap, linewidth=0, alpha=0.85)
        ax.set_xlabel("Time t", labelpad=8)
        ax.set_ylabel("Inventory q", labelpad=8)
        ax.set_zlabel("h(q,t)", labelpad=8)
        ax.set_title(title, fontsize=12, pad=10)
        fig.colorbar(surf, ax=ax, shrink=0.5)

    fig.suptitle("Value Function — Regime-Switching Market Making", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "value_functions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Plot 2: Spread surfaces ────────────────────────────────────────────

def plot_spread_surfaces(data: dict, save_dir: str) -> None:
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    T_mat, Q_mat = np.meshgrid(data["t_grid"], data["q_grid"])

    configs = [
        (data["da1"], "Ask spread δᵃ* — Calm",    "viridis",  (0, 0)),
        (data["db1"], "Bid spread δᵇ* — Calm",    "viridis",  (0, 1)),
        (data["da2"], "Ask spread δᵃ* — Chaotic", "plasma",   (1, 0)),
        (data["db2"], "Bid spread δᵇ* — Chaotic", "plasma",   (1, 1)),
    ]

    for arr, title, cmap, (r, c) in configs:
        ax = fig.add_subplot(gs[r, c], projection="3d")
        surf = ax.plot_surface(T_mat, Q_mat, arr, cmap=cmap, linewidth=0, alpha=0.85)
        ax.set_xlabel("t", labelpad=6)
        ax.set_ylabel("q", labelpad=6)
        ax.set_zlabel("δ*", labelpad=6)
        ax.set_title(title, fontsize=11)
        fig.colorbar(surf, ax=ax, shrink=0.45)

    fig.suptitle("Optimal Spread Surfaces — Both Regimes", fontsize=13)
    path = os.path.join(save_dir, "spread_surfaces.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Plot 3: Spread vs inventory at fixed times ─────────────────────────

def plot_spread_skew(data: dict, save_dir: str) -> None:
    t_grid = data["t_grid"]
    n_t = len(t_grid)
    snapshot_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    snap_indices = [int(f * (n_t - 1)) for f in snapshot_fracs]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(snap_indices)))

    panels = [
        (axes[0, 0], data["da1"], "δᵃ* — Calm regime"),
        (axes[0, 1], data["db1"], "δᵇ* — Calm regime"),
        (axes[1, 0], data["da2"], "δᵃ* — Chaotic regime"),
        (axes[1, 1], data["db2"], "δᵇ* — Chaotic regime"),
    ]

    for ax, arr, title in panels:
        for idx, color, frac in zip(snap_indices, colors, snapshot_fracs):
            label = f"t = {frac:.2f}T"
            ax.plot(data["q_grid"], arr[:, idx], color=color, label=label, lw=1.8)
        ax.axvline(0, color="gray", lw=0.5, ls="--")
        ax.set_xlabel("Inventory q")
        ax.set_ylabel("Optimal half-spread δ*")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Spread Skew vs Inventory at Different Times", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "spread_skew.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Plot 4: Regime comparison at q=0 ──────────────────────────────────

def plot_regime_comparison(data: dict, save_dir: str) -> None:
    q_mid = data["q_max"]   # index for q=0
    t_grid = data["t_grid"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Ask spread over time at q=0
    axes[0].plot(t_grid, data["da1"][q_mid, :], label="Calm (σ₁=0.5)", color="#1D9E75", lw=2)
    axes[0].plot(t_grid, data["da2"][q_mid, :], label="Chaotic (σ₂=2.0)", color="#D85A30", lw=2, ls="--")
    axes[0].set_xlabel("Time t")
    axes[0].set_ylabel("Optimal ask half-spread δᵃ*")
    axes[0].set_title("Ask Spread Over Time (q=0)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Total spread = δᵃ* + δᵇ*
    total1 = data["da1"][q_mid, :] + data["db1"][q_mid, :]
    total2 = data["da2"][q_mid, :] + data["db2"][q_mid, :]
    axes[1].plot(t_grid, total1, label="Calm (σ₁=0.5)", color="#1D9E75", lw=2)
    axes[1].plot(t_grid, total2, label="Chaotic (σ₂=2.0)", color="#D85A30", lw=2, ls="--")
    axes[1].fill_between(t_grid, total1, total2, alpha=0.12, color="#D85A30",
                         label="Regime premium")
    axes[1].set_xlabel("Time t")
    axes[1].set_ylabel("Total spread δᵃ* + δᵇ*")
    axes[1].set_title("Total Spread: Regime Premium (q=0)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Regime Comparison — Spread Widening in Chaotic Regime", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "regime_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Plot 5: HMM blended spread as pi varies ───────────────────────────

def plot_hmm_blending(data: dict, save_dir: str) -> None:
    q_mid = data["q_max"]   # q=0
    t_grid = data["t_grid"]
    pi_values = [0.0, 0.2, 0.5, 0.8, 1.0]
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(pi_values)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for pi, color in zip(pi_values, colors):
        da_blend = (1 - pi) * data["da1"][q_mid, :] + pi * data["da2"][q_mid, :]
        db_blend = (1 - pi) * data["db1"][q_mid, :] + pi * data["db2"][q_mid, :]
        total    = da_blend + db_blend
        lbl = f"π={pi:.1f}"
        axes[0].plot(t_grid, da_blend, color=color, label=lbl, lw=1.8)
        axes[1].plot(t_grid, total,    color=color, label=lbl, lw=1.8)

    for ax, ylabel, title in zip(
        axes,
        ["Blended δᵃ*", "Blended total spread"],
        ["Ask Half-Spread vs π (HMM belief)", "Total Spread vs π (HMM belief)"],
    ):
        ax.set_xlabel("Time t")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9, title="P(regime=2)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("HMM-Blended Spreads: Effect of Regime Belief π", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "hmm_blending.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=RESULTS_DIR)
    parser.add_argument("--plots",   default=PLOTS_DIR)
    args = parser.parse_args()

    os.makedirs(args.plots, exist_ok=True)
    data = load_results(args.results)

    print("Generating plots...")
    plot_value_functions(data,   args.plots)
    plot_spread_surfaces(data,   args.plots)
    plot_spread_skew(data,       args.plots)
    plot_regime_comparison(data, args.plots)
    plot_hmm_blending(data,      args.plots)
    print(f"\nAll plots saved to {args.plots}/")


if __name__ == "__main__":
    main()
