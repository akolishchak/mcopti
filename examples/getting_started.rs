use chrono::NaiveDate;
use mcopti::{
    Context, LegBuilder, LegUniverse, OptionType, Position, Scenario, Simulator,
    raw_option_chain::parse_option_chain_file,
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let chain_path = manifest_dir.join("tests/fixtures/ARM_option_chain_20250908_160038.json");

    let raw = parse_option_chain_file(chain_path)?;
    let mut ctx = Context::from_raw_option_chain("ARM", &raw);

    // Optional tuning
    ctx.config.paths = 10_000;
    ctx.config.step_minutes = 30;

    let expiry = NaiveDate::from_ymd_opt(2025, 9, 12).expect("valid date");
    let builder = LegBuilder::new(&ctx);
    let short = builder
        .create(OptionType::Call, 150.0, expiry)
        .expect("short leg");
    let long = builder
        .create(OptionType::Call, 155.0, expiry)
        .expect("long leg");

    let mut pos = Position::new();
    pos.push(short, -1);
    pos.push(long, 1);

    let universe = LegUniverse::from_positions(vec![pos]);
    let scenario = Scenario::new(&ctx, &universe)?;
    let sim = Simulator::new();
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
