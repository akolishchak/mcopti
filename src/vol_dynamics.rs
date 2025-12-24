
use chrono::{NaiveDate, Duration};

use crate::{Column, HistoricalVolatility, MarketData, OptionType, USMarketCalendar, VolSurface};

pub fn vol_factor_table(ticker: &str, as_of: NaiveDate, volsurface: &VolSurface, calendar: &USMarketCalendar, side: OptionType, ncal_max: i64, clamp: (f64, f64)) -> Vec<f64> {

    let hv = HistoricalVolatility::new(ticker, as_of, ncal_max);
    let mut td = 0;
    let td_cum: Vec<_> = (1..=ncal_max)
        .map(|d| {
            let day = as_of - Duration::days(d);
            if calendar.is_trading_day(day) {
                td += 1;
            }
            td
        })
        .collect();

    let mut f_by_day = Vec::with_capacity(ncal_max as usize +1);
    f_by_day.push(f64::NAN);
    for (d, &tdays) in (1..=ncal_max).zip(td_cum.iter()) {
        // realized vol (annualized, 365) over TRADING DAYS window
        let rv = hv.rv(tdays);
        //  IV is from the current surface, tenor by CALENDAR d
        let iv_now = volsurface.iv(side, d as f64 / 365.0, volsurface.spot);
        f_by_day.push((rv / iv_now).clamp(clamp.0, clamp.1));
    }

    f_by_day
}

pub fn mu_table(ticker: &str, as_of: NaiveDate, clamp: (f64, f64)) -> f64 {
    // long-horizon trend: fit a single slope on log prices over a 120–250 trading day window
    const MIN_LOOKBACK: usize = 120;
    const MAX_LOOKBACK: usize = 250;

    let md = MarketData::default_read("1d")
        .unwrap()
        .columns(&[Column::AdjClose]);

    // fetch enough calendar days to cover the lookback band.
    let lookback_calendar = MAX_LOOKBACK as i64 * 2;
    let start_date = as_of - Duration::days(lookback_calendar);
    let rows = md.fetch(ticker, start_date, as_of).unwrap();

    if rows.len() < MIN_LOOKBACK {
        return 0.0;
    }

    let window = rows.len().min(MAX_LOOKBACK);
    let slice = &rows[rows.len() - window..];

    let mut sum_t = 0.0;
    let mut sum_log = 0.0;
    let mut sum_tt = 0.0;
    let mut sum_tlog = 0.0;

    for (i, (_, values)) in slice.iter().enumerate() {
        // trading-day clock: 252 trading days ~ 1 year
        let t = i as f64 / 252.0;
        let lp = values[0].ln();
        sum_t += t;
        sum_log += lp;
        sum_tt += t * t;
        sum_tlog += t * lp;
    }

    let n = window as f64;
    let cov = sum_tlog - (sum_t * sum_log) / n;
    let var_t = sum_tt - (sum_t * sum_t) / n;
    let slope = if var_t > 0.0 { cov / var_t } else { 0.0 };
    slope.clamp(clamp.0, clamp.1)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OptionChain, raw_option_chain::parse_option_chain_file};
    use std::{fs, path::PathBuf};

    const ARM_CHAIN_PATH: &str = "tests/fixtures/ARM_option_chain_20250908_160038.json";

    #[test]
    fn vol_factor_table_matches_reference() {
        ensure_market_data_db();

        let chain_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(ARM_CHAIN_PATH);
        let raw_chain = parse_option_chain_file(chain_path).expect("failed to load option chain fixture");
        let chain = OptionChain::from_raw(raw_chain);
        let surface = VolSurface::new(&chain);
        let calendar = USMarketCalendar::new(2024, 2026);
        let as_of = NaiveDate::from_ymd_opt(2025, 9, 5).unwrap();

        let factors = vol_factor_table("ARM", as_of, &surface, &calendar, OptionType::Call, 30, (0.5, 3.0));

        // Reference data generated via src/probability_vol_110.py::build_factor_table_from_hv
        // with the ARM chain fixture and atm_side='call'.
        let expected = [
            f64::NAN,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.7105790750101338,
            0.7096720400065878,
            1.222790919020243,
            1.1674783679207656,
            1.084510702604946,
            1.0842909166125334,
            1.084202652687481,
            1.0842085828677583,
            0.9934164631752868,
            0.9564403808742541,
            0.9571767450675535,
            0.8960498854630609,
            0.8962834815222475,
            0.8960584055929454,
            0.9325119936408497,
            0.9306660016267816,
            0.8911665716992232,
            0.891189860175403,
            0.888661415801199,
            0.886687949160405,
            0.8855322047514117,
            1.0062517819995178,
            0.9821716149492258,
            0.9846842489695313,
        ];

        assert_eq!(factors.len(), expected.len());
        assert!(factors[0].is_nan());

        for (idx, (&got, &want)) in factors.iter().zip(expected.iter()).enumerate().skip(1) {
            let diff = (got - want).abs();
            assert!(
                diff < 1e-9,
                "calendar day {idx}: expected {want}, got {got}, diff {diff}"
            );
        }
    }

    #[test]
    fn mu_table_matches_reference() {
        ensure_market_data_db();
        let as_of = NaiveDate::from_ymd_opt(2025, 9, 5).unwrap();
        let mu = mu_table("ARM", as_of, (-0.3, 0.3));

        // Expected drift from long-horizon log-slope (see src/probability_vol_110.py replacement).
        let expected = -0.040_959_103_675_939_16;
        let diff = (mu - expected).abs();
        assert!(
            diff < 1e-12,
            "expected {expected}, got {mu}, diff {diff}"
        );
    }

    fn ensure_market_data_db() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let fixture_db = manifest_dir.join("tests/fixtures/market_data_1d.db");
        assert!(
            fixture_db.exists(),
            "fixture market_data_1d.db is missing at {}",
            fixture_db.display()
        );

        let local_data_dir = manifest_dir.join("data");
        let local_db = local_data_dir.join("market_data_1d.db");
        let needs_copy = match (fixture_db.metadata(), local_db.metadata()) {
            (Ok(f_meta), Ok(l_meta)) => f_meta.len() != l_meta.len(),
            (Ok(_), Err(_)) => true,
            _ => true,
        };

        if needs_copy {
            fs::create_dir_all(&local_data_dir).expect("failed to create crate data directory");
            fs::copy(&fixture_db, &local_db).expect("failed to copy fixture market data db");
        }
    }
}
