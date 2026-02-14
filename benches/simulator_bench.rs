use chrono::NaiveDate;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use mcopti::{
    Config, Context, DEFAULT_CONFIG, LegUniverse, OptionType, Position, Scenario, leg::LegBuilder,
    Simulator, raw_option_chain::parse_option_chain_file,
};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

fn bench_simulator_run(c: &mut Criterion) {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    ensure_market_data_db(&manifest_dir);
    let (context, universe) = build_case(100_000, &manifest_dir);
    let scenario = Scenario::new(&context, &universe)
        .expect("failed to build scenario");

    c.bench_function("simulator_run_paths_100k", |b| {
        b.iter(|| {
            let metrics = Simulator::new()
                .run(black_box(&context), black_box(&universe), black_box(&scenario))
                .expect("simulation returned no metrics");
            black_box(metrics);
        });
    });
}


fn build_case(paths: usize, manifest_dir: &Path) -> (Context, LegUniverse) {
    let chain_path = manifest_dir.join("tests/fixtures/ARM_option_chain_20250908_160038.json");
    let raw_chain =
        parse_option_chain_file(&chain_path).expect("failed to load option chain fixture");
    let mut context = Context::from_raw_option_chain("ARM", &raw_chain);
    context.config = Config {
        paths,
        seed: 7,
        iv_floor: 1.0,
        ..DEFAULT_CONFIG
    };

    let expiry = NaiveDate::from_ymd_opt(2025, 9, 12).expect("invalid fixture expiry");
    let builder = LegBuilder::new(&context);
    let short = builder
        .create(OptionType::Call, 150.0, expiry)
        .expect("missing short leg");
    let long = builder
        .create(OptionType::Call, 155.0, expiry)
        .expect("missing long leg");

    let mut position = Position::new();
    position.push(short, -1);
    position.push(long, 1);
    let leg_universe = LegUniverse::from_positions(vec![position]);
    (context, leg_universe)
}

fn ensure_market_data_db(manifest_dir: &Path) {
    let fixture_db = manifest_dir.join("tests/fixtures/market_data_1d.db");
    let local_db = manifest_dir.join("data/market_data_1d.db");
    let needs_copy = match (fs::metadata(&fixture_db), fs::metadata(&local_db)) {
        (Ok(f_meta), Ok(l_meta)) => f_meta.len() != l_meta.len(),
        (Ok(_), Err(_)) => true,
        _ => true,
    };

    if needs_copy {
        if let Some(parent) = local_db.parent() {
            fs::create_dir_all(parent).expect("failed to create data dir");
        }
        fs::copy(&fixture_db, &local_db).expect("failed to copy fixture market data db");
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(20));
    targets = bench_simulator_run
}
criterion_main!(benches);
