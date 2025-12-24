use chrono::Datelike;
use crate::{VolSurface, USMarketCalendar, OptionChain, RawOptionChain};


pub struct Context {
    ticker: String,
    option_chain: OptionChain,
    vol_surface: VolSurface,
    calendar: USMarketCalendar,
}

impl Context {
    pub fn from_raw_option_chain(ticker: &str, raw_option_chain: &RawOptionChain) -> Self {
        let option_chain = OptionChain::from_raw(raw_option_chain);
        let vol_surface = VolSurface::new(&option_chain);
        let year = raw_option_chain.date.year();
        let calendar = USMarketCalendar::new(year, year+1);

        Self {
            ticker: ticker.to_string(),
            option_chain,
            vol_surface,
            calendar,
        }
    }

    
}