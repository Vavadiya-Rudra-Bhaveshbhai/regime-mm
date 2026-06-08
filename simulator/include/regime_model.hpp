#pragma once
/**
 * regime_model.hpp
 * ================
 * 2-state continuous-time Markov chain for volatility regimes.
 *
 * Regime 0 = calm   (σ₁ small, λ₁ low)
 * Regime 1 = chaotic (σ₂ large, λ₂ high)
 *
 * State transitions are simulated using exponential inter-arrival times.
 */

#include <random>
#include <cassert>
#include <stdexcept>
#include <ostream>
#include <array>

namespace regime_mm {

// ─────────────────────────────────────────────────────────────────────
//  RegimeParams — parameters for one regime
// ─────────────────────────────────────────────────────────────────────

struct RegimeParams {
    double sigma;    ///< Volatility
    double A;        ///< Baseline order arrival rate (orders/sec)
};

// ─────────────────────────────────────────────────────────────────────
//  RegimeModel
// ─────────────────────────────────────────────────────────────────────

class RegimeModel {
public:
    /**
     * @param params_0   Parameters for regime 0 (calm)
     * @param params_1   Parameters for regime 1 (chaotic)
     * @param q_01       Transition rate 0→1  (calm → chaotic)
     * @param q_10       Transition rate 1→0  (chaotic → calm)
     * @param rng        Shared random number generator
     * @param init_regime Starting regime (0 or 1)
     */
    RegimeModel(
        RegimeParams params_0,
        RegimeParams params_1,
        double       q_01,
        double       q_10,
        std::mt19937& rng,
        int          init_regime = 0
    )
        : params_{ params_0, params_1 }
        , q_{ q_01, q_10 }
        , rng_(rng)
        , current_regime_(init_regime)
    {
        if (init_regime != 0 && init_regime != 1)
            throw std::invalid_argument("Regime must be 0 or 1");
        sample_next_switch_time();
    }

    // ── Accessors ────────────────────────────────────────────────────

    int    current_regime() const noexcept { return current_regime_; }
    double sigma()          const noexcept { return params_[current_regime_].sigma; }
    double A()              const noexcept { return params_[current_regime_].A; }
    double next_switch_time() const noexcept { return next_switch_t_; }

    const RegimeParams& params(int regime) const { return params_.at(regime); }

    /**
     * Stationary probability of regime 1:  q_01 / (q_01 + q_10)
     */
    double stationary_prob_chaotic() const noexcept {
        return q_[0] / (q_[0] + q_[1]);
    }

    // ── Simulation ───────────────────────────────────────────────────

    /**
     * Advance to time `t_now`.
     * If `t_now >= next_switch_t_`, the regime switches and a new
     * switch time is sampled.
     *
     * Call this once per simulation step.
     *
     * @return true if a regime switch occurred this call.
     */
    bool advance(double t_now) noexcept {
        bool switched = false;
        while (t_now >= next_switch_t_) {
            current_regime_ = 1 - current_regime_;   // flip 0↔1
            sample_next_switch_time_from(next_switch_t_);
            switched = true;
        }
        return switched;
    }

    // ── Brownian motion helper ────────────────────────────────────────

    /**
     * Sample a price increment: dS = σ_k · dW = σ_k · N(0, dt)
     */
    double sample_price_move(double dt) {
        std::normal_distribution<double> N(0.0, std::sqrt(dt));
        return sigma() * N(rng_);
    }

    // ── Diagnostics ──────────────────────────────────────────────────

    friend std::ostream& operator<<(std::ostream& os, const RegimeModel& m) {
        os << "RegimeModel{ regime=" << m.current_regime_
           << " (" << (m.current_regime_ == 0 ? "calm" : "chaotic") << ")"
           << "  σ=" << m.sigma()
           << "  A=" << m.A()
           << "  next_switch_t=" << m.next_switch_t_
           << " }";
        return os;
    }

private:
    std::array<RegimeParams, 2> params_;
    std::array<double, 2>       q_;      ///< q_[k] = rate of leaving regime k
    std::mt19937&                rng_;
    int                          current_regime_;
    double                       next_switch_t_ = 0.0;

    void sample_next_switch_time() {
        sample_next_switch_time_from(0.0);
    }

    void sample_next_switch_time_from(double t_from) {
        // Time until next switch ~ Exp(q_k)
        std::exponential_distribution<double> exp_dist(q_[current_regime_]);
        next_switch_t_ = t_from + exp_dist(rng_);
    }
};

} // namespace regime_mm
