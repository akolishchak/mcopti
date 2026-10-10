//! Fast, simple Monte Carlo evaluation of option strategies.
//!
//! # Overview
//!
//! `mcopti` is a fast option simulation engine with minimal dependencies.
//! It supports any strategy you can express as a combination of option legs
//! on the same underlying. You choose the contracts and the quantities to
//! buy or sell.
//!
//! The inputs are an option chain, historical prices for the underlying, and
//! the position you want to evaluate. Using that market data, the engine
//! simulates possible movements of the underlying and tracks the position's
//! value as prices change and time passes, from entry through expiration.
//!
//! The result for each position gives its expected profit or loss at expiration
//! and its simulated risk: the largest loss relative to entry observed along
//! any simulated path. It also reports expected profit or loss divided by that
//! risk, expressing the expected return relative to the losses encountered
//! while holding the position.
//!
//! The example below loads market data, builds a position from its legs, and
//! runs an evaluation. A fixed random seed makes the simulated market movements
//! reproducible. The pricing and simulation components can also be used
//! independently in your own analysis.
//!
//! # Example: a call credit spread
//!
//! Sell one ARM call at strike 150 and buy one at strike 155, both expiring
//! September 12, 2025. The example uses the repository's JSON fixture and
//! 10,000 simulation paths.
//!
//! First, prepare price history from the repository root. [`Scenario::new`]
//! reads `data/market_data_1d.db` relative to the working directory:
//!
//! ```sh
//! cargo run --bin ingest_market_data -- ARM 2024-01-01 2025-09-08
//! ```
//!
//! Then load the quotes, construct the spread, and run the simulation:
//!
//! ```no_run
//! use chrono::NaiveDate;
//! use mcopti::{
//!     Context, LegBuilder, LegUniverse, OptionType, Position, Scenario, Simulator,
//!     raw_option_chain::parse_option_chain_file,
//! };
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let raw = parse_option_chain_file(
//!         "tests/fixtures/ARM_option_chain_20250908_160038.json",
//!     )?;
//!     let mut context = Context::from_raw_option_chain("ARM", &raw)?;
//!     context.config.paths = 10_000;
//!
//!     let expiry = NaiveDate::from_ymd_opt(2025, 9, 12).unwrap();
//!     let builder = LegBuilder::new(&context);
//!     let position = Position::default()
//!         .add(builder.create(OptionType::Call, 150.0, expiry)?, -1)
//!         .add(builder.create(OptionType::Call, 155.0, expiry)?, 1);
//!
//!     let universe = LegUniverse::from_positions(vec![position]);
//!     let scenario = Scenario::new(&context, &universe)?;
//!     let results = Simulator::default().run(&context, universe, &scenario)?;
//!     let stats = &results[0];
//!     println!("Expected P&L: {:.4}", stats.expected_value);
//!     println!("Simulated risk: {:.4}", stats.risk);
//!     println!("Return/risk: {:.4}", stats.ror);
//!     Ok(())
//! }
//! ```
//!
//! The result gives the average P&L at expiration, the largest loss relative
//! to entry observed during simulation, and expected P&L divided by that risk.
//!
//! After preparing history, you can also run the repository's example with
//! `cargo run --example getting_started`. Simulation requires existing price
//! history; missing or insufficient data returns an error.
//!
//! # Simulation model
//!
//! ## Volatility surface
//!
//! [`VolSurface`] builds separate call and put surfaces from the option chain's
//! implied volatilities. It uses shape-preserving PCHIP interpolation in
//! log-moneyness (`ln(strike / spot)`) and time to expiry, working in total
//! variance (`IV² × time to expiry`). A tapered adjustment blends toward
//! nondecreasing total variance across maturities while preserving the fitted
//! rows at quoted expiries. Outside the quoted maturity range, lookups use
//! the nearest expiry's total variance.
//!
//! ## Price paths
//!
//! [`Scenario`] generates Monte Carlo paths using **geometric Brownian motion
//! (GBM) with time-varying volatility**, a lognormal model for the underlying
//! price. Drift is fitted to historical prices, and current implied volatility
//! is scaled using historical realized volatility. Volatility varies over time
//! according to the same calibrated schedule for every path.
//!
//! The time grid follows US equity trading sessions, including holidays and
//! early closes, with separate intraday steps and overnight gaps.
//!
//! ## Option valuation
//!
//! At each step, [`Simulator`] calculates Black–Scholes option prices from the
//! simulated spot, remaining time to expiry, and implied volatility read from
//! the surface at that spot. This repricing captures the effects of changes in
//! price, time, and modeled volatility on each leg's value. Pricing uses the
//! snapshot's risk-free rate and zero dividend yield. Expired legs retain their
//! European intrinsic payoff through the rest of the simulation.
//!
//! # API guide
//!
//! Common types and functions are re-exported at the crate root. Use these
//! building blocks in order; the modules below provide additional helpers.
//!
//! | Task | API |
//! | --- | --- |
//! | Load an option snapshot and settings | [`RawOptionChain`] → [`Context`] |
//! | Build and group positions | [`LegBuilder`] → [`Position`] → [`LegUniverse`] |
//! | Generate spot paths and price positions | [`Scenario`] → [`Simulator`] |
//! | Screen and backtest historical snapshots | [`SpreadScreener`] → [`Backtest`] |
//!
//! # Units
//!
//! - Volatility and annual rates are decimals: `0.20` means 20%.
//! - Time to expiry uses calendar years of 365 days.
//! - Leg quantities are positive for long positions and negative for short positions.
//! - Premiums, marks, and P&L have no contract multiplier.
//!
//! # Backtesting
//!
//! [`Backtest`] uses `YYYY-MM-DD` snapshot directories containing `options.db`.
//! Create each database from its directory's JSON files with the existing binary:
//!
//! ```sh
//! cargo run --bin ingest_option_chain_db -- path/to/2025-09-08
//! ```
//!
//! See [`MarketData`] and [`OptionChainDb`] for direct database access.

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
mod serde_helpers;
pub mod simulator;
pub mod spread_screener;
pub mod vol_dynamics;
pub mod volsurface;

#[doc(inline)]
pub use {
    backtest::{
        Backtest, BacktestExitParameters, BacktestParameters, BacktestSummary, ChainScreener,
        ScreenerCandidate,
    },
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
    option_math::{bs_price, bs_price_with_rate, ncdf, simulate_paths},
    pchip::Pchip,
    position::Position,
    raw_option_chain::{OptionType, RawOptionChain},
    scenario::{Scenario, ScenarioError},
    simulator::{Simulator, SimulatorError},
    spread_screener::SpreadScreener,
    vol_dynamics::{VolDynamicsError, mu_table, vol_factor_table},
    volsurface::{VolSurface, VolSurfaceError, interp_linear_kgrid, linspace_vec},
};
