//! Crate entry points and public re-exports for option simulation.

pub mod backtest;
pub mod config;
pub mod context;
pub mod entry_barriers;
pub mod historical_volatility;
pub mod leg;
pub mod leg_universe;
pub mod market_calendar;
pub mod market_data;
pub mod open_position;
pub mod option_chain;
pub mod option_chain_db;
pub mod option_math;
pub mod pchip;
pub mod position;
pub mod raw_option_chain;
pub mod scenario;
pub mod simulator;
pub mod spread_screener;
pub mod vol_dynamics;
pub mod volsurface;

pub use {
    backtest::{Backtest, BacktestParameters, BacktestSummary, ChainScreener, ScreenerCandidate},
    config::{Config, DEFAULT_CONFIG},
    context::Context,
    entry_barriers::EntryBarriers,
    historical_volatility::{HistoricalVolatility, HistoricalVolatilityError},
    leg::{Leg, LegBuilder, LegBuilderError},
    leg_universe::LegUniverse,
    market_calendar::MarketCalendar,
    market_data::{Column, IngestError, MarketData},
    open_position::{Metrics, OpenPosition},
    option_chain::{OptionChain, OptionChainSide},
    option_chain_db::{OptionChainDb, OptionChainDbError, OptionsDbMode},
    option_math::{bs_price, ncdf, simulate_paths},
    pchip::Pchip,
    position::Position,
    raw_option_chain::{OptionType, RawOptionChain},
    scenario::{Scenario, ScenarioError},
    simulator::{Simulator, SimulatorError},
    vol_dynamics::{VolDynamicsError, mu_table, vol_factor_table},
    volsurface::{VolSurface, interp_linear_kgrid, linspace_vec},
};
