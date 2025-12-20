
use chrono::{NaiveDate, Duration};

use crate::{HistoricalVolatility, VolSurface, OptionType, USMarketCalendar};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::option_chain::parse_option_chain_file;
    use std::{fs, path::PathBuf};

    const ARM_CHAIN_PATH: &str = "tests/fixtures/ARM_option_chain_20250908_160038.json";

    #[test]
    fn vol_factor_table_matches_reference() {
        ensure_market_data_db();

        let chain_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(ARM_CHAIN_PATH);
        let chain = parse_option_chain_file(chain_path).expect("failed to load option chain fixture");
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

    fn ensure_market_data_db() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let local_data_dir = manifest_dir.join("data");
        let local_db = local_data_dir.join("market_data_1d.db");
        if let Ok(meta) = local_db.metadata() {
            if meta.len() > 1024 * 1024 {
                return;
            }
            let _ = fs::remove_file(&local_db);
        }

        fs::create_dir_all(&local_data_dir).expect("failed to create crate data directory");

        let workspace_db = manifest_dir
            .parent()
            .expect("workspace root")
            .join("data/market_data_1d.db");
        fs::copy(&workspace_db, &local_db).expect("failed to copy market data db for tests");
    }
}
