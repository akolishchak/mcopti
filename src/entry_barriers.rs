//! Entry-time mark inversion for deriving win/loss spot barriers.

use crate::{Context, OptionType, Position, Scenario, bs_price, interp_linear_kgrid};
use std::borrow::Cow;

const EPSILON: f64 = 1e-8;
const SECONDS_PER_YEAR: f64 = 365.0 * 24.0 * 3600.0;

#[derive(Debug)]
pub struct EntryBarriers {
    pub z_win: f64,
    pub z_loss: f64,
}

struct LegEvalInput<'a> {
    option_type: OptionType,
    strike: f64,
    ln_strike: f64,
    qty: f64,
    tau: f64,
    inv_tau: f64,
    w_row: Cow<'a, [f64]>,
}

impl EntryBarriers {
    pub fn new(
        context: &Context,
        position: &Position,
        scenario: &Scenario,
        profit_take: f64,
        stop_loss: f64,
    ) -> Self {
        if position.legs.is_empty() {
            return Self {
                z_win: f64::NAN,
                z_loss: f64::NAN,
            };
        }

        if scenario.tau_driver.len() != scenario.dt_years.len()
            || scenario.var_cum.len() != scenario.dt_years.len() + 1
        {
            return Self {
                z_win: f64::NAN,
                z_loss: f64::NAN,
            };
        }

        let s0 = context.option_chain.spot.max(EPSILON);
        let leg_inputs = Self::build_leg_inputs(context, position);
        let premium = position.premium;

        // Side is +1 for debit, -1 for credit, keeping PT/SL math sign-safe.
        let side = if premium < 0.0 { -1.0 } else { 1.0 };
        let mark_win = premium * (1.0 + profit_take.max(0.0) * side);
        let mark_loss = premium * (1.0 - stop_loss.max(0.0) * side);

        let s_win = Self::find_s_for_mark(|s| Self::mark_at_spot(&leg_inputs, s), mark_win, s0);
        let s_loss = Self::find_s_for_mark(|s| Self::mark_at_spot(&leg_inputs, s), mark_loss, s0);

        // Compute sigT to this position's expiry inside the global (universe-max) scenario.
        // Same alignment idea as simulator: tau_to_pos = tau_max - tau_offset, where
        // tau_offset = (max_expiry_close - pos_expiry_close).
        let mut pos_expire = position.legs[0].0.expire;
        for (leg, _) in position.legs.iter().skip(1) {
            if leg.expire > pos_expire {
                pos_expire = leg.expire;
            }
        }
        if scenario.max_expire < pos_expire {
            return Self {
                z_win: f64::NAN,
                z_loss: f64::NAN,
            };
        }
        let (_, max_close) = context.calendar.session(scenario.max_expire);
        let (_, pos_close) = context.calendar.session(pos_expire);
        let tau_offset = ((max_close - pos_close).as_seconds_f64() / SECONDS_PER_YEAR).max(0.0);

        // first step where tau_max <= tau_offset is at/after pos expiry close; include that step
        let mut step_end = scenario.dt_years.len();
        for (i, &tau_max) in scenario.tau_driver.iter().enumerate() {
            if tau_max <= tau_offset + EPSILON {
                step_end = i + 1;
                break;
            }
        }
        let sig_t = scenario
            .var_cum
            .get(step_end)
            .copied()
            .unwrap_or(f64::NAN)
            .sqrt();

        let mut z_win = f64::NAN;
        let mut z_loss = f64::NAN;
        if sig_t.is_finite() && sig_t > 0.0 {
            if let Some(s) = s_win.filter(|v| v.is_finite() && *v > 0.0) {
                z_win = (s / s0).ln().abs() / sig_t;
            }
            if let Some(s) = s_loss.filter(|v| v.is_finite() && *v > 0.0) {
                z_loss = (s / s0).ln().abs() / sig_t;
            }
        }

        Self { z_win, z_loss }
    }

    fn build_leg_inputs<'a>(context: &'a Context, position: &Position) -> Vec<LegEvalInput<'a>> {
        let (_, as_of_close) = context.calendar.session(context.option_chain.date);
        position
            .legs
            .iter()
            .map(|(leg, qty)| {
                let (_, leg_close) = context.calendar.session(leg.expire);
                let tau =
                    ((leg_close - as_of_close).as_seconds_f64() / SECONDS_PER_YEAR).max(EPSILON);
                LegEvalInput {
                    option_type: leg.option_type,
                    strike: leg.strike,
                    ln_strike: leg.strike.ln(),
                    qty: *qty as f64,
                    tau,
                    inv_tau: 1.0 / tau,
                    w_row: context.vol_surface.row(leg.option_type, tau),
                }
            })
            .collect()
    }

    fn mark_at_spot(inputs: &[LegEvalInput<'_>], s: f64) -> f64 {
        let s = s.max(EPSILON);
        let ln_s = s.ln();
        let mut mark = 0.0;

        for leg in inputs.iter() {
            let k = leg.ln_strike - ln_s;
            let w = interp_linear_kgrid(k, leg.w_row.as_ref());
            let iv = (w.max(1e-12) * leg.inv_tau).sqrt();
            let leg_mark = bs_price(leg.option_type, s, leg.strike, leg.tau, iv);
            mark += leg.qty * leg_mark;
        }

        mark
    }

    /// Non-monotone root find for f(S)=mark(S)-target.
    /// - Finds all zero-crossings on a log grid and returns the root closest to S0.
    /// - If no bracket is found, returns the nearest |f| point.
    pub fn find_s_for_mark<F>(mut mark_at_s: F, target_mark: f64, s0: f64) -> Option<f64>
    where
        F: FnMut(f64) -> f64,
    {
        const GRID_N: usize = 192;
        const BISECT_ITERS: usize = 32;
        const LO_MULT: f64 = 0.25;
        const HI_MULT: f64 = 4.0;
        const F_EPS: f64 = 1e-10;

        let s0 = s0.max(1e-12);
        let ln_s0 = s0.ln();
        let lo = (s0 * LO_MULT).max(1e-12);
        let hi = (s0 * HI_MULT).max(lo * 1.0001);
        let r = (hi / lo).powf(1.0 / ((GRID_N - 1) as f64));

        let mut s_left = lo;
        let mut f_left = mark_at_s(s_left) - target_mark;
        if !f_left.is_finite() {
            return None;
        }
        if f_left.abs() <= F_EPS {
            return Some(s_left);
        }

        let mut best_near_s = s_left;
        let mut best_near_abs = f_left.abs();
        let mut best_root: Option<(f64, f64)> = None;

        for _ in 1..GRID_N {
            let s_right = (s_left * r).min(hi);
            let f_right = mark_at_s(s_right) - target_mark;
            if !f_right.is_finite() {
                return None;
            }

            let abs_fr = f_right.abs();
            if abs_fr < best_near_abs {
                best_near_abs = abs_fr;
                best_near_s = s_right;
                if abs_fr <= F_EPS {
                    return Some(s_right);
                }
            }

            if f_left.is_sign_negative() != f_right.is_sign_negative() {
                let mut a = s_left;
                let mut b = s_right;
                let mut fa = f_left;

                for _ in 0..BISECT_ITERS {
                    let m = 0.5 * (a + b);
                    let fm = mark_at_s(m) - target_mark;
                    if !fm.is_finite() {
                        break;
                    }
                    if fm.abs() <= F_EPS {
                        a = m;
                        b = m;
                        break;
                    }
                    if fa.is_sign_negative() != fm.is_sign_negative() {
                        b = m;
                    } else {
                        a = m;
                        fa = fm;
                    }
                }

                let root = 0.5 * (a + b);
                let dist = (root.ln() - ln_s0).abs();
                match best_root {
                    None => best_root = Some((root, dist)),
                    Some((_, best_dist)) if dist < best_dist => best_root = Some((root, dist)),
                    _ => {}
                }
            }

            s_left = s_right;
            f_left = f_right;
        }

        if let Some((root, _)) = best_root {
            Some(root)
        } else {
            Some(best_near_s)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LegUniverse, leg::LegBuilder, raw_option_chain::parse_option_chain_file};
    use chrono::NaiveDate;
    use std::fs;
    use std::path::PathBuf;

    fn load_context() -> Context {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        ensure_market_data_db(&manifest_dir);
        let chain_path = manifest_dir.join("tests/fixtures/ARM_option_chain_20250908_160038.json");
        let raw_chain =
            parse_option_chain_file(&chain_path).expect("failed to load option chain fixture");
        Context::from_raw_option_chain("ARM", &raw_chain)
    }

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn make_position(context: &Context) -> Position {
        let expiry = NaiveDate::from_ymd_opt(2025, 9, 12).expect("invalid expiry");
        let builder = LegBuilder::new(context);
        let short_call = builder
            .create(OptionType::Call, 150.0, expiry)
            .expect("missing short call");
        let long_call = builder
            .create(OptionType::Call, 155.0, expiry)
            .expect("missing long call");

        let mut position = Position::new();
        position.push(short_call, -1);
        position.push(long_call, 1);
        position
    }

    fn valid_scenario(max_expire: NaiveDate) -> Scenario {
        Scenario {
            dt_years: vec![0.1, 0.1, 0.1],
            tau_driver: vec![0.5, 0.2, 0.0],
            iv_mult_path: vec![1.0, 1.0, 1.0],
            s_path: vec![100.0, 100.0, 100.0],
            max_expire,
            var_cum: vec![0.0, 0.01, 0.02, 0.03],
        }
    }

    #[test]
    fn find_s_for_mark_picks_closest_root_for_non_monotone_function() {
        let root = EntryBarriers::find_s_for_mark(|s| (s - 80.0) * (s - 120.0), 0.0, 95.0)
            .expect("expected a root");

        assert!(
            approx_eq(root, 80.0, 1e-3),
            "expected root near 80, got {root}"
        );
    }

    #[test]
    fn find_s_for_mark_returns_none_for_non_finite_function_values() {
        let root = EntryBarriers::find_s_for_mark(|_| f64::NAN, 0.0, 100.0);
        assert!(root.is_none(), "expected None for non-finite evaluations");
    }

    #[test]
    fn new_returns_nan_for_invalid_scenario_shapes() {
        let context = load_context();
        let position = make_position(&context);

        // tau_driver len != dt_years len
        let bad = Scenario {
            dt_years: vec![0.1, 0.1],
            tau_driver: vec![0.2],
            iv_mult_path: vec![],
            s_path: vec![],
            max_expire: NaiveDate::from_ymd_opt(2025, 9, 12).expect("invalid date"),
            var_cum: vec![0.0, 0.01, 0.02],
        };

        let barriers = EntryBarriers::new(&context, &position, &bad, 0.5, 0.5);
        assert!(barriers.z_win.is_nan(), "expected z_win NaN");
        assert!(barriers.z_loss.is_nan(), "expected z_loss NaN");
    }

    #[test]
    fn new_returns_nan_when_scenario_max_expire_is_before_position_expire() {
        let context = load_context();
        let position = make_position(&context);
        let scenario = valid_scenario(NaiveDate::from_ymd_opt(2025, 9, 5).expect("invalid date"));

        let barriers = EntryBarriers::new(&context, &position, &scenario, 0.5, 0.5);
        assert!(barriers.z_win.is_nan(), "expected z_win NaN");
        assert!(barriers.z_loss.is_nan(), "expected z_loss NaN");
    }

    #[test]
    fn new_produces_finite_barriers_for_valid_inputs() {
        let context = load_context();
        let position = make_position(&context);
        let expiry = NaiveDate::from_ymd_opt(2025, 9, 12).expect("invalid date");
        let scenario = valid_scenario(expiry);

        let barriers = EntryBarriers::new(&context, &position, &scenario, 0.5, 0.5);
        assert!(barriers.z_win.is_finite(), "expected finite z_win");
        assert!(barriers.z_loss.is_finite(), "expected finite z_loss");
        assert!(barriers.z_win >= 0.0, "expected non-negative z_win");
        assert!(barriers.z_loss >= 0.0, "expected non-negative z_loss");
    }

    #[test]
    fn barriers_regression_for_arm_call_credit_spread() {
        let mut context = load_context();
        context.config.paths = 2;
        context.config.seed = 7;

        let position = make_position(&context);
        let universe = LegUniverse::from_positions(vec![position]);
        let scenario = Scenario::new(&context, &universe).expect("failed to build scenario");
        let barriers = EntryBarriers::new(&context, &universe.positions[0], &scenario, 1.0, 0.3);

        assert!(
            approx_eq(barriers.z_win, 16.898_228_026_517_117, 1e-12),
            "z_win changed: got {}, want {}",
            barriers.z_win,
            16.898_228_026_517_117_f64
        );
        assert!(
            approx_eq(barriers.z_loss, 0.004_223_387_519_968_438, 1e-12),
            "z_loss changed: got {}, want {}",
            barriers.z_loss,
            0.004_223_387_519_968_438_f64
        );
    }

    fn ensure_market_data_db(manifest_dir: &PathBuf) {
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
}
