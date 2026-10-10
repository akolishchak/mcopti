# mcopti

`mcopti` is a fast option simulation engine with minimal dependencies.
It supports any strategy you can express as a combination of option legs
on the same underlying. You choose the contracts and the quantities to
buy or sell.

**Inputs**

- **Option chain:** current option quotes for the underlying.
- **Historical prices:** the underlying's price history.
- **Position:** the strategy's option legs and quantities.

Using these inputs, the engine simulates possible movements of the underlying
and tracks the position's value as prices change and time passes, from entry
through expiration.

**Outputs for each position**

- **Expected P&L:** average profit or loss at expiration.
- **Simulated risk:** the largest loss relative to entry observed along any
  simulated path.
- **Return/risk:** expected P&L divided by simulated risk, expressing the expected
  return relative to the losses encountered while holding the position.

A fixed random seed makes the simulated market movements reproducible.
The pricing and simulation components can also be used independently in your
own analysis. The example below loads market data, builds a position from its
legs, and runs a complete evaluation.

## Example: a call credit spread

Sell one ARM call at strike 150 and buy one at strike 155, both expiring
September 12, 2025. The example uses the repository's JSON fixture and
10,000 simulation paths.

First, prepare price history from the repository root. [Scenario::new]
reads `data/market_data_1d.db` relative to the working directory:

```sh
cargo run --bin ingest_market_data -- ARM 2024-01-01 2025-09-08
```

Then load the quotes, construct the spread, and run the simulation:

```rust
use chrono::NaiveDate;
use mcopti::{
    Context, LegBuilder, LegUniverse, OptionType, Position, Scenario, Simulator,
    raw_option_chain::parse_option_chain_file,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let raw = parse_option_chain_file(
        "tests/fixtures/ARM_option_chain_20250908_160038.json",
    )?;
    let mut context = Context::from_raw_option_chain("ARM", &raw)?;

    // Optional tuning
    context.config.paths = 10_000;
    context.config.step_minutes = 30;

    let expiry = NaiveDate::from_ymd_opt(2025, 9, 12).unwrap();
    let builder = LegBuilder::new(&context);
    let position = Position::default()
        .add(builder.create(OptionType::Call, 150.0, expiry)?, -1)
        .add(builder.create(OptionType::Call, 155.0, expiry)?, 1);

    let universe = LegUniverse::from_positions(vec![position]);
    let scenario = Scenario::new(&context, &universe)?;
    let results = Simulator::default().run(&context, universe, &scenario)?;
    let stats = &results[0];
    println!("Expected P&L: {:.4}", stats.expected_value);
    println!("Simulated risk: {:.4}", stats.risk);
    println!("Return/risk: {:.4}", stats.ror);
    Ok(())
}
```

After preparing history, you can also run the repository's example:

```sh
cargo run --example getting_started
```

## Simulation model

### Volatility surface

[VolSurface] builds separate call and put surfaces from the option chain's
implied volatilities. It uses shape-preserving [PCHIP] interpolation in
log-moneyness (`ln(strike / spot)`) and time to expiry, working in total
variance (`IV² × time to expiry`). A tapered adjustment blends toward
nondecreasing total variance across maturities while preserving the fitted
rows at quoted expiries. Outside the quoted maturity range, lookups use
the nearest expiry's total variance. A precomputed maturity lookup table
speeds repeated volatility queries.

### Price paths

[Scenario] generates Monte Carlo paths using **geometric Brownian motion
(GBM) with time-varying volatility**, a lognormal model for the underlying
price. Drift is fitted to historical prices, and current implied volatility
is scaled using historical realized volatility. Volatility varies over time
according to the same calibrated schedule for every path.

The time grid follows US equity trading sessions in New York time, including
holidays and early closes, with variance split between intraday steps and
overnight gaps.

### Option valuation

At each step, [Simulator] calculates Black–Scholes option prices from the
simulated spot, remaining time to expiry, and implied volatility read from
the surface at that spot. This repricing captures the effects of changes in
price, time, and modeled volatility on each leg's value. Pricing uses the
snapshot's risk-free rate and zero dividend yield. Expired legs retain their
European intrinsic payoff through the rest of the simulation. Paths are
processed in parallel with Rayon.

## API guide

Common types and functions are re-exported at the crate root.

| Task | API |
| --- | --- |
| Load an option snapshot and settings | [RawOptionChain] → [Context] |
| Build and group positions | [LegBuilder] → [Position] → [LegUniverse] |
| Generate spot paths and price positions | [Scenario] → [Simulator] |
| Screen and backtest historical snapshots | [SpreadScreener] → [Backtest] |

- [VolSurface::row] returns total variance across a fixed log-moneyness grid;
  [VolSurface::iv] returns implied volatility.
- [Simulator::run] returns `Result<Vec<SimulationStats>, SimulatorError>`,
  one result per position in input order. Each [SimulationStats] contains the original
  `position`, expected P&L (`expected_value`), the largest simulated loss from
  entry (`risk`), and expected P&L divided by risk (`ror`).
- [EntryBarriers::new] computes entry barrier z-scores (`z_win`, `z_loss`) from
  target mark inversion and scenario volatility scaling.

## Units

- Volatility and annual rates are decimals: `0.20` means 20%.
- Time to expiry uses calendar years of 365 days.
- Leg quantities are positive for long positions and negative for short positions.
- Premiums, marks, and P&L have no contract multiplier.

## Data and assumptions

- Option chain input should be grouped by expiration and sorted by strike within each
  expiry; the bucketing and binary search assume ordered data.
- [Scenario::new], [MarketData::default_read], and [HistoricalVolatility] read a SQLite DB at
  `data/market_data_1d.db` with a `candles` table containing the columns listed in
  [src/market_data.rs](src/market_data.rs). The `data/` directory is gitignored.

For library use, create and populate the database using the existing methods before
constructing a scenario:

```rust,no_run
use chrono::NaiveDate;
use mcopti::MarketData;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut history = MarketData::default_write("1d")?;
    history.ingest(
        "ARM",
        "1d",
        NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        NaiveDate::from_ymd_opt(2025, 9, 8).unwrap(),
    )?;
    Ok(())
}
```

Write mode creates the directory, database, and schema. Ingestion downloads history
from Yahoo Finance and can be repeated to update it. Simulation only reads that
history; it returns an error if the database or required history is unavailable.

## Market data ingest CLI

Use the built-in binary to create/update `data/market_data_1d.db` from Yahoo Finance data.
To use it without cloning the repository after publication:

```sh
cargo install mcopti --bin ingest_market_data
ingest_market_data ARM 2024-01-01 2025-09-08
```

Run:

```sh
cargo run --bin ingest_market_data -- <ticker|file_name> [start YYYY-MM-DD] [end YYYY-MM-DD]
```

Examples:

```sh
cargo run --bin ingest_market_data -- AAPL
cargo run --bin ingest_market_data -- AAPL 2024-01-01
cargo run --bin ingest_market_data -- tickers.txt 2024-01-01 2025-01-01
```

## Option chain ingest CLI

Use the built-in binary to ingest all option chain JSON files from a folder into
`options.db` in that same folder. Backtesting uses `YYYY-MM-DD` snapshot
directories, each containing one of these databases.

Run:

```sh
cargo run --bin ingest_option_chain_db -- <folder_path>
```

Example:

```sh
cargo run --bin ingest_option_chain_db -- tests/fixtures
```

## Backtest spreads binary

Run:

```sh
cargo run --release --bin backtest_spreads -- <data_dir> [profit_take] [stop_loss]
```

## Project goals

- Fast option simulation engine with minimal dependencies.
- Reproducible Monte Carlo analytics with seeded runs.
- Composable building blocks for surfaces, legs, and positions.

## Module map

- [src/raw_option_chain.rs](src/raw_option_chain.rs): parse option chain JSON into typed structs.
- [src/option_chain.rs](src/option_chain.rs): bucket quotes by expiry and side, expose per-expiry slices.
- [src/volsurface.rs](src/volsurface.rs): build a total-variance surface with a maturity adjustment
  toward nondecreasing variance, and query IV or slopes.
- [src/market_calendar.rs](src/market_calendar.rs): US trading calendar and session times.
- [src/market_data.rs](src/market_data.rs): SQLite market data access.
- [src/historical_volatility.rs](src/historical_volatility.rs): realized vol estimates from market data.
- [src/scenario.rs](src/scenario.rs): build time grids, vol multipliers, and Monte Carlo price paths.
- [src/leg.rs](src/leg.rs), [src/position.rs](src/position.rs),
  [src/leg_universe.rs](src/leg_universe.rs): multi-leg position structures.
- [src/simulator.rs](src/simulator.rs): price legs and positions across scenario paths.
- [src/entry_barriers.rs](src/entry_barriers.rs): invert target marks to implied spot barriers and z-scores.

## Tests

Run:

```sh
cargo test
```

Tests use fixtures in `tests/fixtures` and will copy the market data DB into `data/`
when needed.

## Benchmarks

Run all benches:

```sh
cargo bench
```

Run individual benches:

```sh
cargo bench --bench scenario_bench
cargo bench --bench simulator_bench
cargo bench --bench barriers_bench
```

[RawOptionChain]: https://docs.rs/mcopti/latest/mcopti/struct.RawOptionChain.html
[Context]: https://docs.rs/mcopti/latest/mcopti/struct.Context.html
[LegBuilder]: https://docs.rs/mcopti/latest/mcopti/struct.LegBuilder.html
[Position]: https://docs.rs/mcopti/latest/mcopti/struct.Position.html
[LegUniverse]: https://docs.rs/mcopti/latest/mcopti/struct.LegUniverse.html
[Scenario]: https://docs.rs/mcopti/latest/mcopti/struct.Scenario.html
[Scenario::new]: https://docs.rs/mcopti/latest/mcopti/struct.Scenario.html#method.new
[Simulator]: https://docs.rs/mcopti/latest/mcopti/struct.Simulator.html
[Simulator::run]: https://docs.rs/mcopti/latest/mcopti/struct.Simulator.html#method.run
[SimulationStats]: https://docs.rs/mcopti/latest/mcopti/simulator/struct.SimulationStats.html
[SpreadScreener]: https://docs.rs/mcopti/latest/mcopti/struct.SpreadScreener.html
[Backtest]: https://docs.rs/mcopti/latest/mcopti/struct.Backtest.html
[VolSurface]: https://docs.rs/mcopti/latest/mcopti/struct.VolSurface.html
[VolSurface::row]: https://docs.rs/mcopti/latest/mcopti/struct.VolSurface.html#method.row
[VolSurface::iv]: https://docs.rs/mcopti/latest/mcopti/struct.VolSurface.html#method.iv
[PCHIP]: https://docs.rs/mcopti/latest/mcopti/struct.Pchip.html
[EntryBarriers::new]: https://docs.rs/mcopti/latest/mcopti/struct.EntryBarriers.html#method.new
[MarketData::default_read]: https://docs.rs/mcopti/latest/mcopti/struct.MarketData.html#method.default_read
[HistoricalVolatility]: https://docs.rs/mcopti/latest/mcopti/struct.HistoricalVolatility.html
