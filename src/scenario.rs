use chrono::Duration;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::{Context, LegUniverse, OptionType, vol_dynamics::mu_table, vol_factor_table};

const SECONDS_PER_YEAR: f64 = 365.0 * 24.0 * 3600.0;
const MINUTES_PER_DAY: f64 = 24.0 * 60.0;
const OVERNIGHT_VAR_FRACTION: f64 = 0.30;

pub struct Scenario {
    pub dt_years: Vec<f64>, // aka dt_arr: per-step durations in years
    pub tau_driver: Vec<f64>,
    pub iv_mult_path: Vec<f64>,
    pub s_path: Vec<f64>,
}

impl Scenario {
    pub fn new(context: &Context, leg_universe: &LegUniverse) -> Self {
        let start_date = context.option_chain.date;
        let expiry_date = leg_universe.max_expire;
        let calendar = &context.calendar;
        let (_, expiry_close) = calendar.session(expiry_date);
        let step_minutes = context.config.step_minutes;
        let days_to_expiry = (expiry_date - start_date).num_days();
        let capacity = (
            days_to_expiry * context.calendar.max_session_mins()
            / step_minutes + days_to_expiry
        ) as usize;
        let paths = context.config.paths;
        let seed = Some(context.config.seed);

        let mut dt_years = Vec::with_capacity(capacity);
        let mut tau_driver = Vec::with_capacity(capacity);
        let mut iv_mult_path = Vec::with_capacity(capacity);
        let mut sigma_eff = Vec::with_capacity(capacity);
        // Reused buffer for per-day median; keeps allocations off the hot path.
        let mut median_buf = Vec::with_capacity(64);

        // TODO: consider storing shocks once and deriving S paths deterministically for both sides
        let side = if leg_universe.put_present { OptionType::Put } else { OptionType::Call };
        let vol_surface = &context.vol_surface;
        let ncal_max = days_to_expiry.min(365).max(1);
        let factor_clamp = context.config.factor_clamp;
        // dynamic factor curve f_day
        let f_by_day = vol_factor_table(
            &context.ticker,
            start_date,
            vol_surface,
            calendar,
            side,
            ncal_max,
            factor_clamp,
        );
        let iv_level_clamp = context.config.iv_level_clamp;
        // Real-world drift: long-horizon log-slope; fallback to 0 if unavailable
        let mu_trend = mu_table(&context.ticker, start_date, (-0.3, 0.3));
        let s0 = context.option_chain.spot;
        // TODO: per-strategy strikes (e.g., short strike for spreads) instead of s0-only lookup
        let s_iv_floor = s0 * context.config.iv_floor;

        let mut day = start_date;
        let step = Duration::minutes(step_minutes);
        loop {
            let day_start = dt_years.len();
            let (open_dt, close_dt) = calendar.session(day);
            let mut t = open_dt;
            let mut intraday_minutes = 0.0;
            median_buf.clear();

            while t < close_dt {
                let next = (t + step).min(close_dt);
                let dt_year = (next - t).as_seconds_f64() / SECONDS_PER_YEAR;
                let tau = ((expiry_close - next).as_seconds_f64() as f64 / SECONDS_PER_YEAR).max(1e-8);

                dt_years.push(dt_year);
                tau_driver.push(tau);

                let iv = vol_surface
                    .iv(side, tau, s0)
                    .max(vol_surface.iv(side, tau, s_iv_floor));
                let d = ((tau * 365.0).round().max(1.0)) as i32; // >=1 day
                let idx = d.clamp(1, (f_by_day.len() - 1) as i32) as usize;
                let sigma = (f_by_day[idx] * iv).clamp(1e-6, 5.0);
                let iv_mult = (sigma / iv.max(1e-8)).clamp(iv_level_clamp.0, iv_level_clamp.1);

                iv_mult_path.push(iv_mult);
                sigma_eff.push(0.0); // placeholder, filled per-day below
                intraday_minutes += (next - t).as_seconds_f64() / 60.0;
                // Keep a separate buffer because median_inplace permutes the data.
                median_buf.push(sigma);

                t = next;
            }
            let intra_end = dt_years.len();
            let mut day_end = intra_end;

            if day == expiry_date {
                fill_sigma_eff(
                    day_start,
                    intra_end,
                    day_end,
                    intraday_minutes,
                    &dt_years,
                    &mut sigma_eff,
                    &mut median_buf,
                );
                break;
            }

            let next_day = calendar.next_trading_day(day);
            let (next_open, _) = calendar.session(next_day);
            let gap = next_open - close_dt;
            // Use full-resolution gap (seconds) to handle partial-day closures accurately.
            let end = if gap.num_seconds() > 0 {
                let dt_year = gap.as_seconds_f64() / SECONDS_PER_YEAR;
                let tau = ((expiry_close - next_open).as_seconds_f64() / SECONDS_PER_YEAR).max(1e-8);

                dt_years.push(dt_year);
                tau_driver.push(tau);

                let iv = vol_surface
                    .iv(side, tau, s0)
                    .max(vol_surface.iv(side, tau, s_iv_floor));
                let d = ((tau * 365.0).round().max(1.0)) as i32;
                let idx = d.clamp(1, (f_by_day.len() - 1) as i32) as usize;
                let sigma = (f_by_day[idx] * iv).clamp(1e-6, 5.0);
                let iv_mult = (sigma / iv.max(1e-8)).clamp(iv_level_clamp.0, iv_level_clamp.1);

                iv_mult_path.push(iv_mult);
                sigma_eff.push(0.0); // overnight gap step already ends at next open
                if intraday_minutes <= 0.0 {
                    median_buf.push(sigma);
                }
                day_end = sigma_eff.len();

                day_end
            } else {
                intra_end
            };

            fill_sigma_eff(
                day_start,
                intra_end,
                end,
                intraday_minutes,
                &dt_years,
                &mut sigma_eff,
                &mut median_buf,
            );
            day = next_day;
        }

        // Stream simulation to avoid extra drift/vol buffers.
        let steps = sigma_eff.len();
        let mut rng = match seed {
            Some(s) => Xoshiro256PlusPlus::seed_from_u64(s),
            None => Xoshiro256PlusPlus::from_os_rng(),
        };
        let mut s_path = Vec::with_capacity(paths * steps);
        for _ in 0..paths {
            let mut log_price = s0.ln();
            for (&sigma, &dt) in sigma_eff.iter().zip(dt_years.iter()) {
                let vol = sigma * dt.sqrt();
                let drift = (mu_trend - 0.5 * sigma * sigma) * dt;
                let rand: f64 = rng.sample(StandardNormal);
                log_price += rand * vol + drift;
                s_path.push(log_price.exp());
            }
        }

        Self { dt_years, tau_driver, iv_mult_path, s_path }
    }
}

fn fill_sigma_eff(
    start: usize,
    intra_end: usize,
    end: usize,
    intraday_minutes: f64,
    dt_years: &[f64],
    sigma_eff: &mut [f64],
    median_buf: &mut Vec<f64>,
) {
    // median_buf holds the sigma samples for this day; it is mutated in place.
    let sigma_day = median_inplace(median_buf).unwrap_or(0.2);

    if intraday_minutes <= 0.0 {
        for slot in &mut sigma_eff[start..end] {
            *slot = sigma_day;
        }
        return;
    }

    let intra_scale = ((1.0 - OVERNIGHT_VAR_FRACTION) * (MINUTES_PER_DAY / intraday_minutes)).max(1e-12);
    let sigma_intra = sigma_day * intra_scale.sqrt();
    for slot in sigma_eff[start..intra_end].iter_mut() {
        *slot = sigma_intra;
    }
    for (offset, slot) in sigma_eff[intra_end..end].iter_mut().enumerate() {
        let dt_days = dt_years[intra_end + offset] * 365.0;
        let ov_scale = (OVERNIGHT_VAR_FRACTION / dt_days.max(1e-12)).max(1e-12);
        *slot = sigma_day * ov_scale.sqrt();
    }
}

fn median_inplace(buf: &mut Vec<f64>) -> Option<f64> {
    if buf.is_empty() {
        return None;
    }
    let mid = buf.len() / 2;
    buf.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    if buf.len() % 2 == 1 {
        Some(buf[mid])
    } else {
        let lower_max = buf[..mid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some((lower_max + buf[mid]) * 0.5)
    }
}
