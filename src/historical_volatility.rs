//! Realized volatility estimates derived from market data.

use chrono::{Duration, NaiveDate};
use std::{error::Error, fmt};

use crate::market_data::DbMode;
use crate::{Column, MarketData};

/// Adjusted daily log returns for annualized realized volatility estimates.
pub struct HistoricalVolatility {
    returns: Vec<f64>,
}

/// Failure to load history or estimate volatility for a requested lookback.
#[derive(Debug)]
pub enum HistoricalVolatilityError {
    /// The historical price database could not be opened or queried.
    MarketData(rusqlite::Error),
    /// The lookback is negative or its date range cannot be represented.
    InvalidInput(&'static str),
    /// The requested estimate needs more daily returns than were loaded.
    InsufficientHistory {
        /// Number of daily returns needed for the requested estimate.
        required: usize,
        /// Number of daily returns loaded from the database.
        available: usize,
    },
}

impl fmt::Display for HistoricalVolatilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MarketData(e) => write!(f, "market data error: {e}"),
            Self::InvalidInput(msg) => write!(f, "invalid volatility lookback: {msg}"),
            Self::InsufficientHistory {
                required,
                available,
            } => write!(
                f,
                "insufficient market history: need {required} returns, found {available}; ingest more historical data for this ticker"
            ),
        }
    }
}

impl Error for HistoricalVolatilityError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MarketData(e) => Some(e),
            _ => None,
        }
    }
}

impl From<rusqlite::Error> for HistoricalVolatilityError {
    fn from(value: rusqlite::Error) -> Self {
        Self::MarketData(value)
    }
}

impl HistoricalVolatility {
    /// Load daily returns through `as_of` from `data/market_data_1d.db`.
    ///
    /// `max_lookback_days` is a calendar-day horizon; loading adds a ten-day buffer.
    ///
    /// # Errors
    ///
    /// Returns an error for a negative or unrepresentable lookback, or a database failure.
    pub fn new(
        ticker: &str,
        as_of: NaiveDate,
        max_lookback_days: i64,
    ) -> Result<Self, HistoricalVolatilityError> {
        Self::with_data_dir(ticker, as_of, max_lookback_days, "data")
    }

    /// Load daily returns from `<data_dir>/market_data_1d.db`.
    ///
    /// Lookback units and buffering match [`Self::new`].
    ///
    /// # Errors
    ///
    /// Returns an error for a negative or unrepresentable lookback, or a database failure.
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
        if max_lookback_days < 0 {
            return Err(HistoricalVolatilityError::InvalidInput(
                "days must be nonnegative",
            ));
        }
        let start_dt = max_lookback_days
            .checked_add(10)
            .and_then(Duration::try_days)
            .and_then(|lookback| as_of.checked_sub_signed(lookback))
            .ok_or(HistoricalVolatilityError::InvalidInput(
                "date range is too large",
            ))?;
        let data = md
            .columns(&[Column::CoLogAdj, Column::OcLogAdj])
            .fetch(ticker, start_dt, as_of)?;
        // full daily log returns (close-to-close) = co_log_adj + oc_log_adj
        let returns: Vec<_> = data
            .into_iter()
            .map(|(_, values)| values.into_iter().sum::<f64>())
            .collect();

        Ok(Self { returns })
    }

    /// Annualized realized volatility, or an error if the requested history is unavailable.
    ///
    /// Converts calendar days to `round(days * 252 / 365)` observations, with a
    /// minimum of one. Uses sample variance and 252 trading days per year.
    ///
    /// # Errors
    ///
    /// Returns an error for a negative lookback or insufficient loaded returns.
    pub fn rv(&self, lookback_days: i64) -> Result<f64, HistoricalVolatilityError> {
        if lookback_days < 0 {
            return Err(HistoricalVolatilityError::InvalidInput(
                "days must be nonnegative",
            ));
        }
        // calendar -> trading days (approximation)
        let n = (lookback_days as f64 * 252.0 / 365.0).round().max(1.0) as usize;
        let returns_len = self.returns.len();
        if returns_len < n {
            return Err(HistoricalVolatilityError::InsufficientHistory {
                required: n,
                available: returns_len,
            });
        }

        // last n returns
        let returns = &self.returns[returns_len.saturating_sub(n)..];
        let mean = returns.iter().sum::<f64>() / n as f64;
        let sumsq_dev = returns
            .iter()
            .map(|r| {
                let dev = r - mean;
                dev * dev
            })
            .sum::<f64>();
        let divisor = if n > 1 { n - 1 } else { n };
        let var_per_trading_day = sumsq_dev / divisor as f64;
        // sigma_252
        Ok((252.0 * var_per_trading_day).sqrt())
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
        assert!(approx(hv.rv(30).unwrap(), 0.244_401_957_887_463));
        assert!(approx(hv.rv(60).unwrap(), 0.224_599_865_371_256));
        assert!(approx(hv.rv(120).unwrap(), 0.260_623_691_712_029));
    }

    #[test]
    fn rv_returns_error_when_insufficient_history() {
        let as_of = NaiveDate::from_ymd_opt(2023, 6, 30).unwrap();
        let hv = HistoricalVolatility::with_data_dir("MSFT", as_of, 10, FIXTURE_DIR)
            .expect("failed to load market data fixture");
        assert!(matches!(
            hv.rv(120),
            Err(HistoricalVolatilityError::InsufficientHistory { .. })
        ));
    }
}
