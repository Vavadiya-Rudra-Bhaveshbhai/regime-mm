#pragma once
/**
 * market_maker.hpp
 * ================
 * Abstract base class for all market making agents.
 * Concrete strategies implement get_spreads().
 *
 * Agents implemented:
 *   - RegimeSwitchingMM  : reads optimal spreads from the HJB table
 *   - NaiveConstantVolMM : uses A-S closed form with average σ
 *   - SymmetricFixedMM   : posts a fixed constant spread always
 */

#include "order_book.hpp"
#include "regime_model.hpp"
#include <string>
#include <vector>
#include <numeric>
#include <cmath>

namespace regime_mm {

// ─────────────────────────────────────────────────────────────────────
//  Agent state (shared across all agents)
// ─────────────────────────────────────────────────────────────────────

struct AgentState {
    double cash       = 0.0;  ///< Cumulative cash (from trades)
    int    inventory  = 0;    ///< Shares held
    int    n_trades   = 0;    ///< Total trades executed
    int    n_halts    = 0;    ///< Times inventory limit was hit (toxic flow)
    std::vector<double> pnl_series;  ///< Mark-to-market PnL snapshots

    double mark_to_market(double mid) const noexcept {
        return cash + static_cast<double>(inventory) * mid;
    }

    void apply_trade(const Trade& t, int q_max) {
        cash      += t.cash_delta;
        inventory += t.qty;
        ++n_trades;
        if (std::abs(inventory) >= q_max) ++n_halts;
    }

    void snapshot_pnl(double mid) {
        pnl_series.push_back(mark_to_market(mid));
    }
};

// ─────────────────────────────────────────────────────────────────────
//  Abstract base
// ─────────────────────────────────────────────────────────────────────

class MarketMakerBase {
public:
    virtual ~MarketMakerBase() = default;

    /// Name of the strategy (for reports)
    virtual std::string name() const = 0;

    /**
     * Compute optimal ask/bid half-spreads for the current state.
     *
     * @param q       current inventory
     * @param t       current time [0, T]
     * @param regime  current (or inferred) regime (0 or 1)
     * @param pi      regime 2 posterior (only used by HMM agent)
     * @param tau     time remaining T - t
     * @return {delta_ask, delta_bid}
     */
    virtual std::pair<double, double> get_spreads(
        int    q,
        double t,
        int    regime,
        double pi,
        double tau
    ) const = 0;

    AgentState& state() { return state_; }
    const AgentState& state() const { return state_; }

    void reset() { state_ = AgentState{}; }

protected:
    AgentState state_;
};

// ─────────────────────────────────────────────────────────────────────
//  NaiveConstantVolMM — uses fixed average sigma (benchmark)
// ─────────────────────────────────────────────────────────────────────

class NaiveConstantVolMM : public MarketMakerBase {
public:
    NaiveConstantVolMM(
        double sigma_avg, ///< Average volatility (ignores regimes)
        double gamma,
        double kappa,
        int    q_max
    )
        : sigma_(sigma_avg), gamma_(gamma), kappa_(kappa), q_max_(q_max) {}

    std::string name() const override { return "Naive-ConstantVol"; }

    std::pair<double, double> get_spreads(
        int q, double /*t*/, int /*regime*/, double /*pi*/, double tau
    ) const override {
        // A-S closed form with fixed sigma
        double base  = 1.0 / kappa_ + 0.5 * gamma_ * sigma_ * sigma_ * tau;
        double skew  = gamma_ * sigma_ * sigma_ * tau * static_cast<double>(q);
        double da    = std::max(base - skew, 1e-4);
        double db    = std::max(base + skew, 1e-4);
        return {da, db};
    }

private:
    double sigma_, gamma_, kappa_;
    int    q_max_;
};

// ─────────────────────────────────────────────────────────────────────
//  SymmetricFixedMM — dumbest possible benchmark
// ─────────────────────────────────────────────────────────────────────

class SymmetricFixedMM : public MarketMakerBase {
public:
    explicit SymmetricFixedMM(double fixed_half_spread)
        : spread_(fixed_half_spread) {}

    std::string name() const override { return "SymmetricFixed"; }

    std::pair<double, double> get_spreads(
        int /*q*/, double /*t*/, int /*regime*/, double /*pi*/, double /*tau*/
    ) const override {
        return {spread_, spread_};
    }

private:
    double spread_;
};

} // namespace regime_mm
