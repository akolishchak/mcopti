# mcopti

Monte Carlo option strategy simulation and volatility surface tooling in Rust.

mcopti parses option chain JSON, builds a smooth implied vol surface, and simulates
multi-leg positions along Monte Carlo price paths driven by historical and implied vol.

## Project goals
- Fast option simulation engine with minimal dependencies.
- Reproducible Monte Carlo analytics with seeded runs.
- Composable building blocks for surfaces, legs, and positions.

## Highlights
- Shape-preserving PCHIP interpolation and a log-moneyness volatility surface with a tau lookup table.
- US equities market calendar (New York time) with holidays and early closes.
- Historical volatility and drift from a SQLite market data store.
- Scenario builder with intraday steps and overnight variance split.
- Parallel Monte Carlo leg and position valuation using Rayon.
- Entry barrier analytics (`z_win`, `z_loss`) via mark-to-spot inversion.

## Quickstart
```rust
use chrono::NaiveDate;
use mcopti::{
    Context, LegBuilder, LegUniverse, Position, Scenario, Simulator, OptionType,
    raw_option_chain::parse_option_chain_file,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let raw = parse_option_chain_file("tests/fixtures/ARM_option_chain_20250908_160038.json")?;
    let mut ctx = Context::from_raw_option_chain("ARM", &raw);

    // Optional tuning
    ctx.config.paths = 10_000;
    ctx.config.step_minutes = 30;

    let expiry = NaiveDate::from_ymd_opt(2025, 9, 12).unwrap();
    let builder = LegBuilder::new(&ctx);
    let short = builder
        .create(OptionType::Call, 150.0, expiry)
        .expect("short leg");
    let long = builder
        .create(OptionType::Call, 155.0, expiry)
        .expect("long leg");

    let mut pos = Position::default();
    pos.push(short, -1);
    pos.push(long, 1);

    let universe = LegUniverse::from_positions(vec![pos]);
    let scenario = Scenario::new(&ctx, &universe)?;
    let sim = Simulator::default();
    let metrics = sim
        .run(&ctx, &universe, &scenario)
        .expect("simulation produced no metrics");

    println!("positions: {}", metrics.len());
    if let Some(first) = metrics.first() {
        println!(
            "first position -> expected_value: {:.4}, risk: {:.4}",
            first.expected_value, first.risk
        );
    }
    Ok(())
}
```
Run the same example from the repository root:
```
cargo run --example getting_started
```

## Data and assumptions
- Option chain input should be grouped by expiration and sorted by strike within each
  expiry; the bucketing and binary search assume ordered data.
- `MarketData::default_read` and `HistoricalVolatility` expect a SQLite DB at
  `data/market_data_1d.db` with a `candles` table containing the columns listed in
  `src/market_data.rs`. The `data/` directory is gitignored.
- The market calendar is US equities (New York time) with standard holidays and
  early closes.

## Market data ingest CLI
Use the built-in binary to create/update `data/market_data_1d.db` from Yahoo Finance data.

Run:
```
cargo run --bin ingest_market_data -- <ticker|file_name> [start YYYY-MM-DD] [end YYYY-MM-DD]
```

Examples:
```
cargo run --bin ingest_market_data -- AAPL
cargo run --bin ingest_market_data -- AAPL 2024-01-01
cargo run --bin ingest_market_data -- tickers.txt 2024-01-01 2025-01-01
```

## Shapes and outputs
- `VolSurface::row` returns total variance across a fixed log-moneyness grid; `VolSurface::iv`
  returns implied vol.
- `Simulator::run` returns `Result<Vec<Metrics>, SimulatorError>`, one `Metrics` per position:
  `expected_value` and drawdown-based `risk`.
- `EntryBarriers::new` computes per-position entry barrier z-scores (`z_win`, `z_loss`) from
  target mark inversion and scenario volatility scaling.

## Module map
- `src/raw_option_chain.rs`: parse option chain JSON into typed structs.
- `src/option_chain.rs`: bucket quotes by expiry and side, expose per-expiry slices.
- `src/volsurface.rs`: build a monotone total-variance surface and query IV or slopes.
- `src/market_calendar.rs`: US trading calendar and session times.
- `src/market_data.rs`: SQLite market data access.
- `src/historical_volatility.rs`: realized vol estimates from market data.
- `src/scenario.rs`: build time grids, vol multipliers, and Monte Carlo price paths.
- `src/leg.rs`, `src/position.rs`, `src/leg_universe.rs`: multi-leg position structures.
- `src/simulator.rs`: price legs and positions across scenario paths.
- `src/entry_barriers.rs`: invert target marks to implied spot barriers and z-scores.

## Tests
Run:
```
cargo test
```
Tests use fixtures in `tests/fixtures` and will copy the market data DB into `data/`
when needed.

## Benchmarks
Run all benches:
```
cargo bench
```

Run individual benches:
```
cargo bench --bench scenario_bench
cargo bench --bench simulator_bench
cargo bench --bench barriers_bench
```
