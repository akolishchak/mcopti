//! Bundle option chain data, vol surface, calendar, and config for simulation.

use crate::{
    Config, DEFAULT_CONFIG, MarketCalendar, OptionChain, RawOptionChain, VolSurface,
    VolSurfaceError,
};
use chrono::Datelike;

/// Option quotes, volatility surface, calendar, and settings for one ticker.
pub struct Context {
    /// Underlying ticker used to look up historical prices.
    pub ticker: String,
    /// Quotes grouped by option side and expiry.
    pub option_chain: OptionChain,
    /// Implied volatility surface built from the option chain.
    pub vol_surface: VolSurface,
    /// US market sessions covering the chain date and its expiries.
    pub calendar: MarketCalendar,
    /// Simulation settings, initialized from [`DEFAULT_CONFIG`].
    pub config: Config,
    /// Annual continuously compounded risk-free rate, as a decimal.
    pub risk_free_rate: f64,
}

impl Context {
    /// Build a simulation context, returning invalid quote/interpolation errors.
    ///
    /// Quotes must be grouped by expiry and sorted by strike within each side.
    ///
    /// # Errors
    ///
    /// Returns an error if the quotes cannot form a valid [`VolSurface`].
    pub fn from_raw_option_chain(
        ticker: &str,
        raw_option_chain: &RawOptionChain,
    ) -> Result<Self, VolSurfaceError> {
        let option_chain = OptionChain::from_raw(raw_option_chain);
        let vol_surface = VolSurface::new(&option_chain)?;
        let year = raw_option_chain.date.year();
        let last_year = raw_option_chain
            .data
            .iter()
            .map(|c| c.expiration.year())
            .max()
            .unwrap_or(year);
        let calendar = MarketCalendar::new(year - 1, last_year.max(year + 1));

        Ok(Self {
            ticker: ticker.to_string(),
            option_chain,
            vol_surface,
            calendar,
            config: DEFAULT_CONFIG,
            risk_free_rate: raw_option_chain.risk_free_rate,
        })
    }
}
