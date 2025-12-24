use chrono::Datelike;
use crate::{VolSurface, USMarketCalendar, OptionChain};


pub struct Context {
    ticker: String,
    vol_surface: VolSurface,
    calendar: USMarketCalendar,
}

impl Context {
    pub fn from_option_chain(ticker: &str, option_chain: &OptionChain) -> Self {
        let year = option_chain.date.year();
        let vol_surface = VolSurface::new(option_chain);
        let calendar = USMarketCalendar::new(year, year+1);

        Self {
            ticker: ticker.to_string(),
            vol_surface,
            calendar,
        }
    }

    
}