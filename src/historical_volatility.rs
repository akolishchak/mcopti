//! Realized volatility estimates derived from market data.

use chrono::{Duration, NaiveDate};
use std::{error::Error, fmt};

use crate::market_data::DbMode;
use crate::{Column, MarketData};


pub struct HistoricalVolatility {
    returns: Vec<f64>,
}

#[derive(Debug)]
pub enum HistoricalVolatilityError {
    MarketData(rusqlite::Error),
}

impl fmt::Display for HistoricalVolatilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MarketData(e) => write!(f, "market data error: {e}"),
        }
    }
}

impl Error for HistoricalVolatilityError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MarketData(e) => Some(e),
        }
    }
}

impl From<rusqlite::Error> for HistoricalVolatilityError {
    fn from(value: rusqlite::Error) -> Self {
        Self::MarketData(value)
    }
}

impl HistoricalVolatility {
    pub fn new(
        ticker: &str,
        as_of: NaiveDate,
        max_lookback_days: i64,
    ) -> Result<Self, HistoricalVolatilityError> {
        Self::with_data_dir(ticker, as_of, max_lookback_days, "data")
    }

    pub fn with_data_dir(
        ticker: &str,
        as_of: NaiveDate,
        max_lookback_days: i64,
        data_dir: &str,
    ) -> Result<Self, HistoricalVolatilityError> {
        let md = MarketData::new(data_dir, "1d", DbMode::Read)?;
        Self::from_market_data(md, ticker, as_of, max_lookback_days)
    }

    fn from_market_data(
        md: MarketData,
        ticker: &str,
        as_of: NaiveDate,
        max_lookback_days: i64,
    ) -> Result<Self, HistoricalVolatilityError> {
        let start_dt = as_of - Duration::days(max_lookback_days + 10);
        let data = md
            .columns(&[Column::CoLogAdj, Column::OcLogAdj])
            .fetch(ticker, start_dt, as_of)
            ?;
        // full daily log returns (close-to-close) = co_log_adj + oc_log_adj
        let returns: Vec<_> = data
            .into_iter()
            .map(|(_, values)| values.into_iter().sum::<f64>())
            .collect();

        Ok(Self {
            returns,
        })
    }

    pub fn rv(&self, lookback_days: i64) -> f64 {
        // calendar -> trading days (approximation)
        let n = (lookback_days as f64 * 252.0 / 365.0).round().max(1.0) as usize;
        let returns_len = self.returns.len();
        assert!(returns_len >= n);

        // last n returns
        let returns = &self.returns[returns_len.saturating_sub(n)..];
        let mean = returns.iter().sum::<f64>() / n as f64;
        let sumsq_dev = returns.iter()
            .map(|r| {
                let dev = r - mean;
                dev*dev
            })
            .sum::<f64>();
        let divisor = if n > 1 { n - 1} else { n };
        let var_per_trading_day = sumsq_dev / divisor as f64;
        // sigma_252
        (252.0 * var_per_trading_day).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_DIR: &str = "tests/fixtures";

    #[test]
    fn rv_matches_values() {
        let as_of = NaiveDate::from_ymd_opt(2023, 6, 30).unwrap();
        let hv = HistoricalVolatility::with_data_dir("MSFT", as_of, 160, FIXTURE_DIR)
            .expect("failed to load market data fixture");

        let approx = |a: f64, b: f64| (a - b).abs() < 1e-12;
        assert!(approx(hv.rv(30), 0.244_401_957_887_463));
        assert!(approx(hv.rv(60), 0.224_599_865_371_256));
        assert!(approx(hv.rv(120), 0.260_623_691_712_029));
    }

    #[test]
    #[should_panic]
    fn rv_panics_when_insufficient_history() {
        let as_of = NaiveDate::from_ymd_opt(2023, 6, 30).unwrap();
        let hv = HistoricalVolatility::with_data_dir("MSFT", as_of, 10, FIXTURE_DIR)
            .expect("failed to load market data fixture");
        let _ = hv.rv(120);
    }
}
