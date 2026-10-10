use chrono::NaiveDate;
use mcopti::{
    Context, HistoricalVolatility, HistoricalVolatilityError, LegBuilder, LegUniverse, OptionType,
    Position, RawOptionChain, Scenario, ScenarioError, Simulator, SimulatorError, VolSurfaceError,
    raw_option_chain::parse_option_chain_file,
};
use std::path::Path;

fn raw_chain() -> RawOptionChain {
    parse_option_chain_file(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/ARM_option_chain_20250908_160038.json"),
    )
    .unwrap()
}

fn context() -> Context {
    Context::from_raw_option_chain("ARM", &raw_chain()).unwrap()
}

fn expiry() -> NaiveDate {
    NaiveDate::from_ymd_opt(2025, 9, 12).unwrap()
}

fn universe(context: &Context) -> LegUniverse {
    let leg = LegBuilder::new(context)
        .create(OptionType::Call, 150.0, expiry())
        .unwrap();
    LegUniverse::from_positions(vec![Position::default().add(leg, 1)])
}

fn manual_scenario() -> Scenario {
    Scenario {
        dt_years: vec![0.1, 0.1],
        tau_driver: vec![0.2, 0.1],
        iv_mult_path: vec![1.0, 1.0],
        s_path: vec![145.0, 146.0],
        max_expire: expiry(),
        var_cum: vec![0.0, 0.01, 0.02],
    }
}

#[test]
fn single_expiry_and_single_side_surfaces_are_supported() {
    for side in [None, Some(OptionType::Call), Some(OptionType::Put)] {
        let mut raw = raw_chain();
        raw.data
            .retain(|c| c.expiration == expiry() && side.is_none_or(|side| c.option_type == side));
        let ctx = Context::from_raw_option_chain("ARM", &raw).unwrap();
        let tau = (expiry() - raw.date).num_days() as f64 / 365.0;
        for quote_side in [OptionType::Call, OptionType::Put] {
            let row = ctx.vol_surface.row(quote_side, tau);
            if side.is_none_or(|side| side == quote_side) {
                let row = row.unwrap();
                assert_eq!(row.len(), ctx.vol_surface.row_len());
                assert!(row.iter().all(|v| v.is_finite() && *v > 0.0));
                assert!(
                    ctx.vol_surface
                        .iv(quote_side, tau, 150.0)
                        .unwrap()
                        .is_finite()
                );
                // A single maturity uses its quoted variance row for extrapolation.
                assert_eq!(&*ctx.vol_surface.row(quote_side, tau * 2.0).unwrap(), &*row);
            } else {
                assert!(matches!(row, Err(VolSurfaceError::MissingSide(_))));
            }
        }
    }
}

#[test]
fn empty_and_unsorted_chains_return_errors() {
    let mut raw = raw_chain();
    raw.data.clear();
    assert!(Context::from_raw_option_chain("ARM", &raw).is_err());

    let mut raw = raw_chain();
    raw.data.swap(0, 1);
    assert!(Context::from_raw_option_chain("ARM", &raw).is_err());

    let mut raw = raw_chain();
    raw.data.truncate(1);
    assert!(Context::from_raw_option_chain("ARM", &raw).is_err());

    let mut raw = raw_chain();
    raw.last_price = 0.0;
    assert!(Context::from_raw_option_chain("ARM", &raw).is_err());
}

#[test]
fn surface_queries_reject_invalid_numbers() {
    let ctx = context();
    assert!(ctx.vol_surface.row(OptionType::Call, f64::NAN).is_err());
    assert!(ctx.vol_surface.row(OptionType::Call, -0.1).is_err());
    assert!(ctx.vol_surface.iv(OptionType::Call, 0.1, 0.0).is_err());
    assert!(
        ctx.vol_surface
            .iv_slope(OptionType::Call, 0.0, 0.0)
            .is_err()
    );
}

#[test]
fn absent_ticker_returns_insufficient_history_error() {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let hv =
        HistoricalVolatility::with_data_dir("__ABSENT__", expiry(), 30, data_dir.to_str().unwrap())
            .unwrap();
    assert!(matches!(
        hv.rv(30),
        Err(HistoricalVolatilityError::InsufficientHistory { available: 0, .. })
    ));
    assert!(matches!(
        hv.rv(-1),
        Err(HistoricalVolatilityError::InvalidInput(_))
    ));
}

#[test]
fn scenario_rejects_invalid_configuration_before_loading_history() {
    for step in [0, -1, i64::MAX] {
        let mut ctx = context();
        ctx.config.step_minutes = step;
        assert!(matches!(
            Scenario::new(&ctx, &universe(&ctx)),
            Err(ScenarioError::InvalidInput(_))
        ));
    }
    let mut ctx = context();
    ctx.config.paths = 0;
    assert!(matches!(
        Scenario::new(&ctx, &universe(&ctx)),
        Err(ScenarioError::InvalidInput(_))
    ));

    for clamp in [(3.0, 0.5), (f64::NAN, 3.0), (0.0, 3.0)] {
        let mut ctx = context();
        ctx.config.factor_clamp = clamp;
        assert!(matches!(
            Scenario::new(&ctx, &universe(&ctx)),
            Err(ScenarioError::InvalidInput(_))
        ));
    }
}

#[test]
fn scenario_rejects_empty_universe_and_invalid_expiries() {
    let ctx = context();
    let empty = LegUniverse::from_positions(vec![]);
    assert!(matches!(
        Scenario::new(&ctx, &empty),
        Err(ScenarioError::InvalidInput(_))
    ));
    for date in [
        ctx.option_chain.date.pred_opt().unwrap(),
        NaiveDate::from_ymd_opt(2025, 9, 13).unwrap(),
    ] {
        let mut legs = universe(&ctx);
        legs.max_expire = date;
        assert!(matches!(
            Scenario::new(&ctx, &legs),
            Err(ScenarioError::InvalidInput(_))
        ));
    }
}

#[test]
fn simulator_rejects_inconsistent_scenario_arrays() {
    let ctx = context();
    let scenario = manual_scenario();
    assert!(
        Simulator::default()
            .run(&ctx, universe(&ctx), &scenario)
            .is_ok()
    );
    for field in ["paths", "durations", "multipliers", "variance", "expiry"] {
        let mut scenario = manual_scenario();
        match field {
            "paths" => scenario.s_path.push(999.0),
            "durations" => {
                scenario.dt_years.pop();
            }
            "multipliers" => {
                scenario.iv_mult_path.pop();
            }
            "variance" => {
                scenario.var_cum.pop();
            }
            "expiry" => scenario.max_expire = NaiveDate::from_ymd_opt(2025, 9, 19).unwrap(),
            _ => unreachable!(),
        }
        assert!(
            matches!(
                Simulator::default().run(&ctx, universe(&ctx), &scenario),
                Err(SimulatorError::InvalidInput(_))
            ),
            "field {field}"
        );
    }
}

#[test]
fn simulator_rejects_invalid_path_prices() {
    let ctx = context();
    for price in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut scenario = manual_scenario();
        scenario.s_path[0] = price;
        assert!(matches!(
            Simulator::default().run(&ctx, universe(&ctx), &scenario),
            Err(SimulatorError::InvalidInput(_))
        ));
    }
}
