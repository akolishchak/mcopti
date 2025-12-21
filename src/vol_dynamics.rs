
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

pub fn mu_table(ticker: &str, as_of: NaiveDate, ncal_max: i64, clamp: (f64, f64)) -> Vec<f64> {
    let md = MarketData::default_read("1d")
        .unwrap()
        .columns(&[Column::AdjClose]);

    let start_date = as_of - Duration::days(ncal_max + 5);
    let rows = md.fetch(ticker, start_date, as_of).unwrap();

    if rows.is_empty() {
        return vec![f64::NAN; ncal_max as usize + 1];
    }

    let (anchor_dt, anchor_vals) = rows.last().unwrap();
    let anchor_date = anchor_dt.date_naive();
    let anchor_close = anchor_vals[0];

    // rows are sorted, unique, with sufficient history
    let mut mu_by_day = vec![f64::NAN; ncal_max as usize + 1]; // idx 0 unused
    let (lo, hi) = clamp;

    let mut last_mu = f64::NAN;
    let mut fill_start = 1;

    // walk from most recent backward, filling calendar-day slots up to each span.
    for (dt, values) in rows.iter().rev() {
        let span_days = (anchor_date - dt.date_naive()).num_days() as usize;
        if span_days == 0 {
            continue;
        }
        let close = values[0];
        last_mu = (anchor_close / close).ln() * 365.0 / span_days as f64;
        last_mu = last_mu.clamp(lo, hi);

        let end = span_days.min(ncal_max as usize);
        if end >= fill_start {
            mu_by_day[fill_start..=end].iter_mut().for_each(|slot| *slot = last_mu);
            fill_start = end + 1;
        }
        if fill_start > ncal_max as usize {
            break;
        }
    }

    // ackfill beyond oldest row if needed.
    mu_by_day[fill_start..=ncal_max as usize]
        .iter_mut()
        .for_each(|slot| *slot = last_mu);

    mu_by_day
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

    #[test]
    fn mu_table_matches_reference() {
        ensure_market_data_db();
        let as_of = NaiveDate::from_ymd_opt(2025, 9, 5).unwrap();
        let mu = mu_table("ARM", as_of, 30, (-1.0, 0.5));

        // Generated via src/probability_vol_110.py::build_mu_table on the same data slice.
        let expected = [
            f64::NAN,
            0.5,
            0.5,
            0.5,
            -0.05280648807325038,
            -0.05280648807325038,
            -0.05280648807325038,
            -0.05280648807325038,
            -1.0,
            -0.7243570634459671,
            -0.5479747236800254,
            0.09379161150602244,
            0.047215466048862384,
            0.047215466048862384,
            0.047215466048862384,
            0.5,
            0.5,
            0.5,
            -0.41976047663695426,
            -0.09283983495103236,
            -0.09283983495103236,
            -0.09283983495103236,
            -0.28334819345128853,
            -0.3891443778134891,
            -0.45754224736927523,
            -0.3011930545071657,
            -0.03109705896484995,
            -0.03109705896484995,
            -0.03109705896484995,
            0.23909553691038646,
            0.18186713933666795,
        ];

        assert_eq!(mu.len(), expected.len());
        assert!(mu[0].is_nan());

        for (idx, (&got, &want)) in mu.iter().zip(expected.iter()).enumerate().skip(1) {
            let diff = (got - want).abs();
            assert!(
                diff < 1e-12,
                "calendar day {idx}: expected {want}, got {got}, diff {diff}"
            );
        }
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
