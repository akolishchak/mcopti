//! Bundle option chain data, vol surface, calendar, and config for simulation.

use crate::{Config, DEFAULT_CONFIG, MarketCalendar, OptionChain, RawOptionChain, VolSurface};
use chrono::Datelike;

pub struct Context {
    pub ticker: String,
    pub option_chain: OptionChain,
    pub vol_surface: VolSurface,
    pub calendar: MarketCalendar,
    pub config: Config,
}

impl Context {
    pub fn from_raw_option_chain(ticker: &str, raw_option_chain: &RawOptionChain) -> Self {
        let option_chain = OptionChain::from_raw(raw_option_chain);
        let vol_surface = VolSurface::new(&option_chain);
        let year = raw_option_chain.date.year();
        let calendar = MarketCalendar::new(year, year + 1);

        Self {
            ticker: ticker.to_string(),
            option_chain,
            vol_surface,
            calendar,
            config: DEFAULT_CONFIG,
        }
    }
}
