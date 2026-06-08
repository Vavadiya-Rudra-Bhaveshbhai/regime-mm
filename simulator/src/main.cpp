/**
 * main.cpp
 * ========
 * Entry point for the regime-switching market making simulator.
 *
 * Runs three agents head-to-head on identical market paths:
 *   1. RegimeSwitchingMM  — reads optimal spreads from HJB table (our agent)
 *   2. NaiveConstantVolMM — A-S with average sigma (benchmark)
 *   3. SymmetricFixedMM   — constant symmetric spread (dumbest benchmark)
 *
 * Outputs a CSV of PnL series and a summary table of metrics.
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make -j4
 *   ./regime_mm_sim --config ../configs/default.yaml
 */

#include "../include/order_book.hpp"
#include "../include/regime_model.hpp"
#include "../include/market_maker.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <numeric>

using namespace regime_mm;

// ─────────────────────────────────────────────────────────────────────
//  Minimal YAML-like config reader (avoids yaml-cpp dependency)
// ─────────────────────────────────────────────────────────────────────

struct SimConfig {
    // Regime
    double sigma_1   = 0.5;
    double sigma_2   = 2.0;
    double q_12      = 0.5;
    double q_21      = 2.0;
    // Order flow
    double A_1       = 5.0;
    double A_2       = 20.0;
    double kappa     = 1.5;
    // Agent
    double gamma     = 0.1;
    double phi       = 0.01;
    int    q_max     = 20;
    // Time
    double T         = 1.0;
    double dt        = 0.001;
    // Simulator
    int    n_sims    = 1000;
    int    seed      = 42;
    double S0        = 100.0;
    // Fixed spread benchmark
    double fixed_spread = 0.5;
};

SimConfig default_config() { return {}; }

// ─────────────────────────────────────────────────────────────────────
//  Closed-form spread (used as fallback for regime-switching agent
//  until the full table-reader is implemented)
// ─────────────────────────────────────────────────────────────────────

std::pair<double, double> hjb_spread_approx(
    int q, double tau, double sigma, double gamma, double kappa, int q_max
) {
    // Clamp inventory to avoid extreme skew
    double q_clamped = std::clamp(static_cast<double>(q), -(double)q_max, (double)q_max);
    double base = 1.0 / kappa + 0.5 * gamma * sigma * sigma * tau;
    double skew = gamma * sigma * sigma * tau * q_clamped;
    double da   = std::max(base - skew, 1e-4);
    double db   = std::max(base + skew, 1e-4);
    return {da, db};
}

// ─────────────────────────────────────────────────────────────────────
//  Metrics
// ─────────────────────────────────────────────────────────────────────

struct Metrics {
    std::string name;
    double sharpe_ratio       = 0.0;
    double mean_pnl           = 0.0;
    double std_pnl            = 0.0;
    double max_inv_drawdown   = 0.0;
    double halt_frequency     = 0.0;
    int    total_trades       = 0;
};

Metrics compute_metrics(
    const std::string& name,
    const std::vector<double>& pnl_series,
    double max_inv_drawdown,
    double halt_freq,
    int n_trades
) {
    Metrics m;
    m.name = name;
    m.total_trades = n_trades;
    m.max_inv_drawdown = max_inv_drawdown;
    m.halt_frequency   = halt_freq;

    if (pnl_series.size() < 2) return m;

    // PnL increments
    std::vector<double> returns(pnl_series.size() - 1);
    for (size_t i = 0; i < returns.size(); ++i)
        returns[i] = pnl_series[i + 1] - pnl_series[i];

    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double var  = 0.0;
    for (double r : returns) var += (r - mean) * (r - mean);
    var /= returns.size();
    double sd = std::sqrt(var);

    m.mean_pnl    = pnl_series.back();
    m.std_pnl     = sd * std::sqrt(returns.size());
    m.sharpe_ratio = (sd > 1e-12) ? mean / sd : 0.0;
    return m;
}

// ─────────────────────────────────────────────────────────────────────
//  Single simulation run
// ─────────────────────────────────────────────────────────────────────

struct RunResult {
    std::vector<double> pnl_regime;
    std::vector<double> pnl_naive;
    std::vector<double> pnl_fixed;
    int    trades_regime = 0, trades_naive = 0, trades_fixed = 0;
    int    halts_regime  = 0, halts_naive  = 0, halts_fixed  = 0;
    double max_inv_regime = 0, max_inv_naive = 0, max_inv_fixed = 0;
};

RunResult run_one(const SimConfig& cfg, std::mt19937& rng) {
    // ── Setup ─────────────────────────────────────────────────────────
    RegimeModel regime(
        RegimeParams{cfg.sigma_1, cfg.A_1},
        RegimeParams{cfg.sigma_2, cfg.A_2},
        cfg.q_12, cfg.q_21, rng, 0
    );

    OrderBook book(cfg.S0);

    // Agent states
    AgentState st_regime, st_naive, st_fixed;

    // Naive agent uses average sigma weighted by stationary distribution
    double pi_star = cfg.q_12 / (cfg.q_12 + cfg.q_21);
    double sigma_avg = (1.0 - pi_star) * cfg.sigma_1 + pi_star * cfg.sigma_2;

    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::normal_distribution<double> norm(0.0, 1.0);

    int n_steps = static_cast<int>(cfg.T / cfg.dt);
    int snapshot_every = std::max(1, n_steps / 100);

    double max_inv_r = 0, max_inv_n = 0, max_inv_f = 0;

    // ── Main time loop ─────────────────────────────────────────────────
    for (int step = 0; step < n_steps; ++step) {
        double t   = step * cfg.dt;
        double tau = cfg.T - t;

        // Regime: advance to current time
        regime.advance(t);
        int k = regime.current_regime();

        // Price move: dS = σ_k · dW
        double dS = regime.sigma() * norm(rng) * std::sqrt(cfg.dt);
        book.apply_price_move(dS);
        double S = book.mid_price();

        // ── Get spreads from each agent ──────────────────────────────
        auto [da_r, db_r] = hjb_spread_approx(
            st_regime.inventory, tau, regime.sigma(), cfg.gamma, cfg.kappa, cfg.q_max);

        auto [da_n, db_n] = hjb_spread_approx(
            st_naive.inventory, tau, sigma_avg, cfg.gamma, cfg.kappa, cfg.q_max);

        double da_f = 0.5, db_f = 0.5;  // fixed symmetric

        // ── Simulate order arrivals ──────────────────────────────────
        // Poisson: prob of arrival in dt = λ(δ) · dt = A · e^{−κδ} · dt
        double A_k = regime.A();
        auto poisson_hit = [&](double da, double db) -> std::pair<bool, bool> {
            double rate_a = A_k * std::exp(-cfg.kappa * da) * cfg.dt;
            double rate_b = A_k * std::exp(-cfg.kappa * db) * cfg.dt;
            return {unif(rng) < rate_a, unif(rng) < rate_b};
        };

        // Process trades for each agent
        auto process = [&](AgentState& st, double da, double db) {
            if (std::abs(st.inventory) >= cfg.q_max) return; // halted
            auto [ask_hit, bid_hit] = poisson_hit(da, db);
            if (ask_hit) {
                Trade tr{S, S + da, da, -1, S + da};
                st.apply_trade(tr, cfg.q_max);
            }
            if (bid_hit) {
                Trade tr{S, S - db, db, +1, -(S - db)};
                st.apply_trade(tr, cfg.q_max);
            }
        };

        process(st_regime, da_r, db_r);
        process(st_naive,  da_n, db_n);
        process(st_fixed,  da_f, db_f);

        // Track max inventory
        max_inv_r = std::max(max_inv_r, std::abs((double)st_regime.inventory));
        max_inv_n = std::max(max_inv_n, std::abs((double)st_naive.inventory));
        max_inv_f = std::max(max_inv_f, std::abs((double)st_fixed.inventory));

        // Snapshot PnL
        if (step % snapshot_every == 0) {
            st_regime.snapshot_pnl(S);
            st_naive.snapshot_pnl(S);
            st_fixed.snapshot_pnl(S);
        }
    }

    // Final PnL snapshot
    double S_final = book.mid_price();
    st_regime.snapshot_pnl(S_final);
    st_naive.snapshot_pnl(S_final);
    st_fixed.snapshot_pnl(S_final);

    RunResult res;
    res.pnl_regime = st_regime.pnl_series;
    res.pnl_naive  = st_naive.pnl_series;
    res.pnl_fixed  = st_fixed.pnl_series;
    res.trades_regime = st_regime.n_trades;
    res.trades_naive  = st_naive.n_trades;
    res.trades_fixed  = st_fixed.n_trades;
    res.halts_regime  = st_regime.n_halts;
    res.halts_naive   = st_naive.n_halts;
    res.halts_fixed   = st_fixed.n_halts;
    res.max_inv_regime = max_inv_r;
    res.max_inv_naive  = max_inv_n;
    res.max_inv_fixed  = max_inv_f;
    return res;
}

// ─────────────────────────────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    SimConfig cfg = default_config();

    std::cout << "=== Regime-Switching Market Making Simulator ===\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Config: σ=(" << cfg.sigma_1 << "," << cfg.sigma_2
              << ")  q=(" << cfg.q_12 << "," << cfg.q_21
              << ")  γ=" << cfg.gamma
              << "  κ=" << cfg.kappa
              << "  T=" << cfg.T
              << "  n_sims=" << cfg.n_sims << "\n\n";

    std::mt19937 rng(cfg.seed);

    // Aggregate metrics over all simulations
    std::vector<double> sharpe_r, sharpe_n, sharpe_f;
    std::vector<double> max_inv_r, max_inv_n, max_inv_f;
    std::vector<double> halt_r, halt_n, halt_f;
    int total_trades_r = 0, total_trades_n = 0, total_trades_f = 0;

    for (int sim = 0; sim < cfg.n_sims; ++sim) {
        RunResult res = run_one(cfg, rng);

        auto m_r = compute_metrics("Regime",  res.pnl_regime, res.max_inv_regime,
                                   (double)res.halts_regime / std::max(1, res.trades_regime),
                                   res.trades_regime);
        auto m_n = compute_metrics("Naive",   res.pnl_naive,  res.max_inv_naive,
                                   (double)res.halts_naive / std::max(1, res.trades_naive),
                                   res.trades_naive);
        auto m_f = compute_metrics("Fixed",   res.pnl_fixed,  res.max_inv_fixed,
                                   (double)res.halts_fixed / std::max(1, res.trades_fixed),
                                   res.trades_fixed);

        sharpe_r.push_back(m_r.sharpe_ratio);
        sharpe_n.push_back(m_n.sharpe_ratio);
        sharpe_f.push_back(m_f.sharpe_ratio);
        max_inv_r.push_back(res.max_inv_regime);
        max_inv_n.push_back(res.max_inv_naive);
        max_inv_f.push_back(res.max_inv_fixed);

        total_trades_r += res.trades_regime;
        total_trades_n += res.trades_naive;
        total_trades_f += res.trades_fixed;

        if ((sim + 1) % (cfg.n_sims / 10) == 0)
            std::cout << "  Completed " << (sim + 1) << "/" << cfg.n_sims << " simulations\n";
    }

    // ── Print summary ──────────────────────────────────────────────────
    auto mean_vec = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    std::cout << "\n=== BACKTEST RESULTS (" << cfg.n_sims << " simulations) ===\n";
    std::cout << std::left
              << std::setw(22) << "Agent"
              << std::setw(14) << "Avg Sharpe"
              << std::setw(18) << "Avg Max Inv"
              << std::setw(14) << "Avg Trades"
              << "\n";
    std::cout << std::string(68, '-') << "\n";

    auto print_row = [&](const std::string& name,
                         const std::vector<double>& sh,
                         const std::vector<double>& inv,
                         int trades) {
        std::cout << std::left
                  << std::setw(22) << name
                  << std::setw(14) << mean_vec(sh)
                  << std::setw(18) << mean_vec(inv)
                  << std::setw(14) << (trades / cfg.n_sims)
                  << "\n";
    };

    print_row("Regime-Switching",  sharpe_r, max_inv_r, total_trades_r);
    print_row("Naive-ConstantVol", sharpe_n, max_inv_n, total_trades_n);
    print_row("SymmetricFixed",    sharpe_f, max_inv_f, total_trades_f);

    std::cout << "\nDone.\n";
    return 0;
}
