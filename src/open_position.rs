use crate::Position;

use chrono::NaiveDate;

pub struct OpenPosition {
    pub ticker: String,
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

pub struct Metrics {
    pub pnl: f64,
    pub profit_factor: f64,
    pub drawdown: f64,
}

impl OpenPosition {
    pub fn new(ticker: String, position: Position, pt_mark: f64, sl_mark: f64) -> Self {
        let log_price = f64::ln(position.premium);

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

    pub fn is_expired(&self, date: NaiveDate) -> bool {
        self.position.legs.iter().any(|(leg, _)| leg.expire >= date)
    }

    pub fn update(&mut self, date: NaiveDate, price: f64) -> Option<Metrics> {
        let log_price = f64::ln(price);
        let side = self.position.side();
        let ret = side * (log_price - self.log_last_price);
        if ret > 0.0 {
            self.wins += ret;
        } else {
            self.losses -= ret;
        }
        self.pnl = side * (self.log_start_price - self.log_last_price);
        self.drawdown = self.drawdown.min(self.pnl);
        if self.is_expired(date) {
            return Some(self.metrics());
        }

        if side * (price - self.sl_mark) <= 0.0 || side * (price - self.pt_mark) >= 0.0 {
            return Some(self.metrics());
        }

        None
    }

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
