//! Track position marks, exit thresholds, and running performance.

use crate::Position;

use chrono::NaiveDate;

/// A position tracked until an exit threshold or any leg expiry is reached.
pub struct OpenPosition {
    /// Underlying ticker for this position.
    pub ticker: String,
    /// Option legs and signed entry premium.
    pub position: Position,
    pt_mark: f64,
    sl_mark: f64,
    wins: f64,
    losses: f64,
    pnl: f64,
    drawdown: f64,
    log_start_price: f64,
    log_last_price: f64,
}

/// Performance from observed marks, without a contract multiplier.
pub struct Metrics {
    /// Latest position mark minus its entry premium.
    pub pnl: f64,
    /// Positive mark changes divided by total absolute mark changes; zero when unchanged.
    pub profit_factor: f64,
    /// Lowest observed P&L relative to entry, capped at zero; always nonpositive.
    pub drawdown: f64,
}

impl OpenPosition {
    /// Start tracking at the entry premium with explicit profit and stop mark thresholds.
    pub fn new(ticker: String, position: Position, pt_mark: f64, sl_mark: f64) -> Self {
        let log_price = position.premium;

        Self {
            ticker,
            position,
            pt_mark,
            sl_mark,
            wins: 0.0,
            losses: 0.0,
            pnl: 0.0,
            drawdown: 0.0,
            log_start_price: log_price,
            log_last_price: log_price,
        }
    }

    /// Return whether `date` is on or after any leg's expiry.
    pub fn is_expired(&self, date: NaiveDate) -> bool {
        self.position.legs.iter().any(|(leg, _)| date >= leg.expire)
    }

    /// Record a position mark and return metrics when an exit condition is met.
    ///
    /// An exit occurs at any leg expiry, `price <= sl_mark`, or `price >= pt_mark`.
    pub fn update(&mut self, date: NaiveDate, price: f64) -> Option<Metrics> {
        let ret = price - self.log_last_price;
        if ret > 0.0 {
            self.wins += ret;
        } else {
            self.losses -= ret;
        }
        self.log_last_price = price;
        self.pnl = price - self.log_start_price;
        self.drawdown = self.drawdown.min(self.pnl);
        if self.is_expired(date) {
            return Some(self.metrics());
        }

        if price <= self.sl_mark || price >= self.pt_mark {
            return Some(self.metrics());
        }

        None
    }

    /// Return performance through the latest observed mark.
    pub fn metrics(&self) -> Metrics {
        let total_ret = self.wins + self.losses;
        let profit_factor = if total_ret != 0.0 {
            self.wins / total_ret
        } else {
            0.0
        };

        Metrics {
            pnl: self.pnl,
            profit_factor,
            drawdown: self.drawdown,
        }
    }
}
