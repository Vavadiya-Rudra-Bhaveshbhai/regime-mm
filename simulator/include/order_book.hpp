#pragma once
/**
 * order_book.hpp
 * ==============
 * Minimal limit order book for the regime-switching market making simulator.
 *
 * Tracks the current mid-price and the MM's posted bid/ask quotes.
 * In a full LOB simulator the book would store all resting limit orders;
 * here we keep only the MM's two quotes (sufficient for the HJB agent).
 */

#include <cstdint>
#include <optional>
#include <ostream>
#include <stdexcept>

namespace regime_mm {

// ─────────────────────────────────────────────────────────────────────
//  Side enum
// ─────────────────────────────────────────────────────────────────────

enum class Side { BID, ASK };

inline const char* side_str(Side s) { return s == Side::BID ? "BID" : "ASK"; }

// ─────────────────────────────────────────────────────────────────────
//  Quote — the MM's resting limit order on one side
// ─────────────────────────────────────────────────────────────────────

struct Quote {
    double price;       ///< Quoted price
    double half_spread; ///< Distance from mid-price (δᵃ or δᵇ)
    Side   side;

    double distance_from_mid(double mid) const noexcept {
        return (side == Side::ASK) ? (price - mid) : (mid - price);
    }
};

// ─────────────────────────────────────────────────────────────────────
//  Trade — result of a market order matching against the MM's quote
// ─────────────────────────────────────────────────────────────────────

struct Trade {
    double mid_price;    ///< Mid-price at time of trade
    double exec_price;   ///< Price at which the trade executed
    double half_spread;  ///< The half-spread earned
    int    qty;          ///< +1 if MM bought (bid hit), -1 if MM sold (ask hit)
    double cash_delta;   ///< Change in MM cash (positive = inflow)
};

// ─────────────────────────────────────────────────────────────────────
//  OrderBook
// ─────────────────────────────────────────────────────────────────────

class OrderBook {
public:
    explicit OrderBook(double initial_mid = 100.0)
        : mid_price_(initial_mid) {}

    // ── Mid-price ────────────────────────────────────────────────────

    double mid_price() const noexcept { return mid_price_; }

    void set_mid_price(double s) {
        if (s <= 0.0) throw std::domain_error("Mid-price must be positive");
        mid_price_ = s;
    }

    void apply_price_move(double dS) noexcept { mid_price_ += dS; }

    // ── MM quote management ──────────────────────────────────────────

    /**
     * Post or refresh the MM's bid and ask quotes.
     *
     * @param delta_a  ask half-spread (ask = mid + delta_a)
     * @param delta_b  bid half-spread (bid = mid − delta_b)
     */
    void post_quotes(double delta_a, double delta_b) noexcept {
        ask_ = Quote{ mid_price_ + delta_a, delta_a, Side::ASK };
        bid_ = Quote{ mid_price_ - delta_b, delta_b, Side::BID };
    }

    std::optional<Quote> ask() const noexcept { return ask_; }
    std::optional<Quote> bid() const noexcept { return bid_; }

    // ── Matching ─────────────────────────────────────────────────────

    /**
     * A market BUY arrives — hits the MM's ask.
     * MM sells 1 share: cash += (mid + δᵃ), inventory -= 1.
     *
     * @return Trade record, or nullopt if no ask is posted.
     */
    std::optional<Trade> hit_ask() noexcept {
        if (!ask_) return std::nullopt;
        Trade t{
            .mid_price   = mid_price_,
            .exec_price  = ask_->price,
            .half_spread = ask_->half_spread,
            .qty         = -1,                       // MM sold
            .cash_delta  = ask_->price               // MM received ask price
        };
        ask_ = std::nullopt;   // quote consumed
        return t;
    }

    /**
     * A market SELL arrives — hits the MM's bid.
     * MM buys 1 share: cash -= (mid − δᵇ), inventory += 1.
     *
     * @return Trade record, or nullopt if no bid is posted.
     */
    std::optional<Trade> hit_bid() noexcept {
        if (!bid_) return std::nullopt;
        Trade t{
            .mid_price   = mid_price_,
            .exec_price  = bid_->price,
            .half_spread = bid_->half_spread,
            .qty         = +1,                       // MM bought
            .cash_delta  = -bid_->price              // MM paid bid price
        };
        bid_ = std::nullopt;
        return t;
    }

    // ── Diagnostics ──────────────────────────────────────────────────

    friend std::ostream& operator<<(std::ostream& os, const OrderBook& ob) {
        os << "OrderBook{ mid=" << ob.mid_price_;
        if (ob.ask_) os << "  ask=" << ob.ask_->price << "(δ=" << ob.ask_->half_spread << ")";
        if (ob.bid_) os << "  bid=" << ob.bid_->price << "(δ=" << ob.bid_->half_spread << ")";
        os << " }";
        return os;
    }

private:
    double             mid_price_;
    std::optional<Quote> ask_;
    std::optional<Quote> bid_;
};

} // namespace regime_mm
