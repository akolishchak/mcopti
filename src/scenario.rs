use chrono::Duration;

use crate::{Context, LegUniverse, OptionType, vol_dynamics::mu_table, vol_factor_table};

pub struct Scenario {

}

impl Scenario {
    pub fn new(context: &Context, leg_universe: &LegUniverse) -> Self {
        let start_date = context.option_chain.date;
        let expiry_date = leg_universe.max_expire;
        let calendar = &context.calendar;

        let step_minutes = context.config.step_minutes;
        let days_to_expiry = (expiry_date - start_date).num_days();
        let capacity = days_to_expiry * context.calendar.max_session_mins() / step_minutes;

        let mut step_ends = Vec::with_capacity(capacity as usize);
        let mut day = start_date;
        let mut skip_open = false;
        let step = Duration::minutes(step_minutes);
        loop {
            let (open_dt, close_dt) = calendar.session(day);
            let mut t = if skip_open {
                open_dt + step
            } else {
                open_dt
            };

            while t < close_dt {
                let next = (t + step).min(close_dt);
                step_ends.push(next);
                t = next;
            }

            if day == expiry_date {
                break;
            }

            day = calendar.next_trading_day(day);
            let (next_open, _) = calendar.session(day);
            let gap_days = (next_open - close_dt).num_days();
            if gap_days > 0 {
                step_ends.push(next_open);
                skip_open = true; // already added the open as the overnight step end
            } else {
                skip_open = false;
            }
        }

        let (_, expiry_close) = calendar.session(expiry_date);
        let seconds_per_year = 365.0 * 24.0 * 3600.0;
        let tau_driver: Vec<_> = step_ends.iter()
            .map(|t| ((expiry_close - t).as_seconds_f64() / seconds_per_year).max(1e-8))
            .collect();

        // TODO: consider to store shocks once and derive S_path_put or/and
        // S_path_call deterministically from the same shocks
        let side = if leg_universe.put_present { OptionType::Put } else { OptionType::Call };
        let vol_surface = &context.vol_surface;
        // dynamic factor curve f_day
        let tau_days = days_to_expiry;
        let ncal_max = days_to_expiry.min(365).max(1);
        let factor_clamp = &context.config.factor_clamp;
        let f_by_day = vol_factor_table(&context.ticker, start_date, vol_surface, calendar, side, ncal_max, *factor_clamp);

        let iv_level_clamp = &context.config.iv_level_clamp;
        let iv_mult_path: Vec<_> = f_by_day.iter()
            .map(|f| f.clamp(iv_level_clamp.0, iv_level_clamp.1))
            .collect();

        // Real-world drift: long-horizon log-slope; fallback to 0 if unavailable
        let mu_tred = mu_table(&context.ticker, start_date, (-0.3, 0.3));

        let s0 = context.option_chain.spot;
        // TODO: per-strategy strikes, e.g. short strike fro spreads
        let s_iv_floor = s0 * context.config.iv_floor;
        let sigma_cal_path: Vec<_> = tau_driver
            .iter()
            .map(|&tau| {
                let iv = vol_surface
                    .iv(side, tau, s0)
                    .max(vol_surface.iv(side, tau, s_iv_floor));
                let d = ((tau * 365.0).round().max(1.0)) as i32;
                let idx = d.clamp(1, (f_by_day.len() - 1) as i32) as usize;
                (f_by_day[idx] * iv).clamp(1e-6, 5.0)
            })
            .collect();
        



        Self {

        }
    }
}
