//! Crate entry points and public re-exports for option simulation.

pub mod pchip;
pub mod raw_option_chain;
pub mod option_chain;
pub mod volsurface;
pub mod market_calendar;
pub mod market_data;
pub mod historical_volatility;
pub mod option_math;
pub mod vol_dynamics;
pub mod context;
pub mod leg;
pub mod position;
pub mod leg_universe;
pub mod scenario;
pub mod simulator;
pub mod entry_barriers;
pub mod config;

pub use {
    raw_option_chain::{RawOptionChain, OptionType},
    option_chain::{OptionChain, OptionChainSide},
    volsurface::{VolSurface, interp_linear_kgrid, linspace_vec},
    pchip::Pchip,
    market_calendar::MarketCalendar,
    market_data::{MarketData, Column, IngestError},
    historical_volatility::{HistoricalVolatility, HistoricalVolatilityError},
    option_math::{ncdf, bs_price, simulate_paths},
    vol_dynamics::{vol_factor_table, mu_table, VolDynamicsError},
    context::Context,
    leg::Leg,
    position::Position,
    leg_universe::LegUniverse,
    scenario::{Scenario, ScenarioError},
    simulator::{Simulator, SimulatorError},
    entry_barriers::EntryBarriers,
    config::{Config, DEFAULT_CONFIG},
};
