//! Price legs and positions along simulated scenario paths.

use crate::{
    Context, LegUniverse, OptionType, Scenario, bs_price, interp_linear_kgrid, linspace_vec,
};
use rayon::prelude::*;
use std::borrow::Cow;
use std::{error::Error, fmt};

const SECONDS_PER_YEAR: f64 = 365.0 * 24.0 * 3600.0;
const EPSILON: f64 = 1e-8;

pub struct Simulator {
    exit_grid: Vec<f64>,
    commission_per_trade: f64,
    close_slippage_frac: f64,
}

#[derive(Debug)]
pub struct Metrics {
    pub expected_value: f64,
    pub risk: f64,
}

#[derive(Debug)]
pub enum SimulatorError {
    InvalidInput(&'static str),
    EmptyExpirySlice { expire_id: usize },
}

impl fmt::Display for SimulatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid simulation input: {msg}"),
            Self::EmptyExpirySlice { expire_id } => {
                write!(f, "empty expiry slice for expire_id={expire_id}")
            }
        }
    }
}

impl Error for SimulatorError {}

struct ExpiryData {
    taus: Vec<f64>,
    rows: Vec<f64>,
    step_stride: usize,
}

struct PathAcc {
    // sum of final marks across processed paths, per position
    sum_last: Vec<f64>,
    // worst (lowest) running mark seen across all processed paths, per position
    worst_min: Vec<f64>,
    // number of full paths aggregated into this accumulator
    paths_done: usize,
}

impl PathAcc {
    fn new(pos_count: usize) -> Self {
        Self {
            sum_last: vec![0.0; pos_count],
            worst_min: vec![f64::INFINITY; pos_count],
            paths_done: 0,
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.paths_done += other.paths_done;
        for ((sum_last, worst_min), (&other_sum_last, &other_worst_min)) in self
            .sum_last
            .iter_mut()
            .zip(self.worst_min.iter_mut())
            .zip(other.sum_last.iter().zip(other.worst_min.iter()))
        {
            *sum_last += other_sum_last;
            *worst_min = (*worst_min).min(other_worst_min);
        }
        self
    }
}

impl Simulator {
    pub fn new() -> Self {
        let exit_grid = linspace_vec(0.1, 1.0, 10);
        Self {
            exit_grid,
            commission_per_trade: 0.0,
            close_slippage_frac: 0.0,
        }
    }

    pub fn exit_grid(mut self, start: f64, end: f64, points: usize) -> Self {
        self.exit_grid = linspace_vec(start, end, points);
        self
    }

    pub fn commission_per_trade(mut self, value: f64) -> Self {
        self.commission_per_trade = value;
        self
    }

    pub fn close_slippage_frac(mut self, value: f64) -> Self {
        self.close_slippage_frac = value;
        self
    }

    // returns per-position metrics (expected value and drawdown-based risk) across paths.
    pub fn run(
        &self,
        context: &Context,
        universe: &LegUniverse,
        scenario: &Scenario,
    ) -> Result<Vec<Metrics>, SimulatorError> {
        let calendar = &context.calendar;
        let vol_surface = &context.vol_surface;
        let s_path = &scenario.s_path;
        let iv_mult_path = &scenario.iv_mult_path;
        let tau_driver = &scenario.tau_driver;
        let steps = scenario.tau_driver.len();
        let leg_count = universe.legs.len();
        let max_expire = universe.max_expire;
        let positions_idx = &universe.positions_idx;
        let positions = &universe.positions;
        let pos_count = positions_idx.len();
        let ln_strike: Vec<f64> = universe.legs.iter().map(|leg| leg.strike.ln()).collect();
        let (_, max_expiry_close) = calendar.session(max_expire);
        if steps == 0 || leg_count == 0 {
            return Err(SimulatorError::InvalidInput(
                "scenario has zero steps or universe has zero legs",
            ));
        }
        // Scenario paths are stored as contiguous [path][step] blocks.
        let paths = s_path.len() / steps;
        if paths == 0 {
            return Err(SimulatorError::InvalidInput("scenario has zero paths"));
        }
        let row_stride = vol_surface.row_len();
        if row_stride == 0 {
            return Err(SimulatorError::InvalidInput(
                "vol surface row length is zero",
            ));
        }

        // walk expiries in order so we can align each slice to the common tau grid and reuse precomputed rows.
        let mut expiry_data: Vec<ExpiryData> = Vec::with_capacity(universe.expiries());
        for expire_slice in universe.expiry_slices() {
            let first_leg = expire_slice
                .legs
                .first()
                .ok_or(SimulatorError::EmptyExpirySlice {
                    expire_id: expire_slice.expire_id,
                })?;
            let (_, leg_close) = calendar.session(first_leg.expire);
            // Align this expiry's timeline with the global driver so tau hits zero when the slice expires.
            let tau_offset =
                ((max_expiry_close - leg_close).as_seconds_f64() / SECONDS_PER_YEAR).max(0.0);
            let legs_in_expiry = expire_slice.legs.len();
            let step_stride = legs_in_expiry * row_stride;

            // Vec over steps -> (tau, leg-major flat rows for that step)
            let mut taus = Vec::with_capacity(steps);
            let mut rows = Vec::with_capacity(steps * step_stride);
            for (tau, iv_mult) in tau_driver.iter().zip(iv_mult_path.iter()) {
                let tau_to_expire = tau - tau_offset;
                // Once tau is exhausted, remaining steps are post-expiry.
                if tau_to_expire <= EPSILON {
                    break;
                }

                // Precompute a contiguous chunk of vol rows scaled by the path's variance multiplier.
                // Rows store total variance, so the scale applies as iv_mult^2.
                let scale = iv_mult * iv_mult;
                // Cache per-step rows to avoid re-reading the surface when multiple legs share type.
                let mut row_call: Option<Cow<'_, [f64]>> = None;
                let mut row_put: Option<Cow<'_, [f64]>> = None;

                // Build leg-major row blocks to keep indexing cheap in inner loop.
                for leg in expire_slice.legs.iter() {
                    let src: &[f64] = match leg.option_type {
                        OptionType::Call => row_call.get_or_insert_with(|| {
                            vol_surface.row(OptionType::Call, tau_to_expire)
                        }),
                        OptionType::Put => row_put
                            .get_or_insert_with(|| vol_surface.row(OptionType::Put, tau_to_expire)),
                    };
                    rows.extend(src.iter().map(|&w| w * scale));
                }
                taus.push(tau_to_expire);
            }
            expiry_data.push(ExpiryData {
                taus,
                rows,
                step_stride,
            });
        }

        // parallelize across path chunks and reduce into per-position aggregates
        let threads = rayon::current_num_threads().max(1);
        // chunk by paths to keep per-thread work sizable
        let chunk_paths = (paths / (threads * 4).max(1)).max(1);
        let chunk_len = chunk_paths * steps;

        let acc = s_path
            .par_chunks(chunk_len)
            .fold(
                // each worker builds a local accumulator for its assigned path chunks
                || PathAcc::new(pos_count),
                |mut acc, path_chunk| {
                    let mut leg_marks: Vec<f64> = vec![0.0; leg_count];
                    let mut min_mark: Vec<f64> = vec![0.0; pos_count];
                    let mut last_mark: Vec<f64> = vec![0.0; pos_count];

                    for path_local in path_chunk.chunks_exact(steps) {
                        let mut first_step = true;
                        for (step, &s) in path_local.iter().enumerate() {
                            let ln_s = s.ln();
                            //
                            // mark legs
                            //
                            // track the absolute leg offset; expiry_slices() yields legs in the same stable order as universe.legs.
                            let mut leg_idx = 0;
                            for (expire_slice, expiry_data) in
                                universe.expiry_slices().zip(expiry_data.iter())
                            {
                                let steps_to_expiry = expiry_data.taus.len();
                                if step < steps_to_expiry {
                                    let tau = expiry_data.taus[step];
                                    let step_base = step * expiry_data.step_stride;
                                    let rows = &expiry_data.rows
                                        [step_base..step_base + expiry_data.step_stride];
                                    for (slice_leg_idx, leg) in expire_slice.legs.iter().enumerate()
                                    {
                                        let offset = slice_leg_idx * row_stride;
                                        let w_row = &rows[offset..offset + row_stride];
                                        // log-moneyness drives the vol-surface lookup.
                                        let k = ln_strike[leg_idx] - ln_s;
                                        // w is total variance at (k, tau); convert to IV before pricing.
                                        let w = interp_linear_kgrid(k, w_row);
                                        let iv = (w / tau).sqrt();
                                        // mark-to-market the leg using slice-specific rows and path price.
                                        leg_marks[leg_idx] =
                                            bs_price(leg.option_type, s, leg.strike, tau, iv);

                                        // safe to advance: each slice covers a disjoint, ordered range of legs.
                                        leg_idx += 1;
                                    }
                                } else {
                                    // After expiry, keep the terminal payoff flat through the remaining steps.
                                    // This models European intrinsic value after expiration.
                                    let s = path_local[steps_to_expiry.saturating_sub(1)];
                                    for leg in expire_slice.legs.iter() {
                                        let value_at_expire = match leg.option_type {
                                            OptionType::Call => (s - leg.strike).max(0.0),
                                            OptionType::Put => (leg.strike - s).max(0.0),
                                        };
                                        leg_marks[leg_idx] = value_at_expire;
                                        leg_idx += 1;
                                    }
                                }
                            }

                            //
                            // mark positions
                            //
                            for ((pos_idx, last_mark), min_mark) in positions_idx
                                .iter()
                                .zip(last_mark.iter_mut())
                                .zip(min_mark.iter_mut())
                            {
                                let mut mark = 0.0;
                                for &(leg_idx, qty) in pos_idx.iter() {
                                    mark += (qty as f64) * leg_marks[leg_idx];
                                }
                                *last_mark = mark;
                                if first_step {
                                    *min_mark = mark;
                                } else if mark < *min_mark {
                                    *min_mark = mark;
                                }
                            }
                            first_step = false;
                        }

                        acc.paths_done += 1;
                        // merge one completed path into this worker-local accumulator
                        for ((sum_last, worst_min), (&path_last, &path_min)) in acc
                            .sum_last
                            .iter_mut()
                            .zip(acc.worst_min.iter_mut())
                            .zip(last_mark.iter().zip(min_mark.iter()))
                        {
                            *sum_last += path_last;
                            *worst_min = (*worst_min).min(path_min);
                        }
                    }

                    acc
                },
            )
            // merge all worker accumulators into a final aggregate
            .reduce(|| PathAcc::new(pos_count), |a, b| a.merge(b));

        let mut metrics = Vec::with_capacity(pos_count);
        let inv_paths = 1.0 / (acc.paths_done as f64);
        for (pos, (&sum_last, &worst_min)) in positions
            .iter()
            .zip(acc.sum_last.iter().zip(acc.worst_min.iter()))
        {
            let expected_value = sum_last * inv_paths - pos.premium;
            let risk = (pos.premium - worst_min).max(0.0);
            metrics.push(Metrics {
                expected_value,
                risk,
            });
        }
        Ok(metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{Config, DEFAULT_CONFIG},
        leg::LegBuilder,
        position::Position,
        raw_option_chain::parse_option_chain_file,
    };
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

    fn assert_close(got: f64, want: f64, tol: f64, label: &str) {
        let diff = (got - want).abs();
        assert!(
            diff < tol,
            "{label} mismatch: expected {want}, got {got}, diff {diff}"
        );
    }

    #[test]
    fn expected_values_match_credit_spreads() {
        let mut context = load_context();
        context.config = Config {
            paths: 2,
            seed: 7,
            iv_floor: 1.0,
            ..DEFAULT_CONFIG
        };

        let expiry = NaiveDate::from_ymd_opt(2025, 9, 12).unwrap();
        let builder = LegBuilder::new(&context);
        let short_call = builder
            .create(OptionType::Call, 150.0, expiry)
            .expect("missing short call");
        let long_call = builder
            .create(OptionType::Call, 155.0, expiry)
            .expect("missing long call");
        let short_put = builder
            .create(OptionType::Put, 125.0, expiry)
            .expect("missing short put");
        let long_put = builder
            .create(OptionType::Put, 120.0, expiry)
            .expect("missing long put");

        let mut call_spread = Position::new();
        call_spread.push(short_call, -1);
        call_spread.push(long_call, 1);

        let mut put_spread = Position::new();
        put_spread.push(short_put, -1);
        put_spread.push(long_put, 1);

        let universe = LegUniverse::from_positions(vec![call_spread, put_spread]);
        let scenario = Scenario::new(&context, &universe).expect("failed to build scenario");
        let metrics = Simulator::new()
            .run(&context, &universe, &scenario)
            .expect("simulation returned no metrics");
        assert_eq!(metrics.len(), 2);

        //   (Ks, Kl) = (150, 155) call -> best_ev = 0.21
        //   (Ks, Kl) = (125, 120) put  -> best_ev = 0.125
        assert_close(
            metrics[0].expected_value,
            0.210_000_000_000_000,
            1e-12,
            "call spread expected_value",
        );
        assert_close(
            metrics[1].expected_value,
            0.125_000_000_000_000,
            1e-12,
            "put spread expected_value",
        );

        assert_close(
            metrics[0].risk,
            0.573_660_576_019_878,
            1e-12,
            "call spread risk",
        );
        assert_close(
            metrics[1].risk,
            0.573_851_912_888_447,
            1e-12,
            "put spread risk",
        );
    }

    #[test]
    fn run_returns_error_when_scenario_has_no_steps() {
        let context = load_context();
        let expiry = NaiveDate::from_ymd_opt(2025, 9, 12).unwrap();
        let builder = LegBuilder::new(&context);
        let call = builder
            .create(OptionType::Call, 150.0, expiry)
            .expect("missing call leg");

        let mut position = Position::new();
        position.push(call, 1);
        let universe = LegUniverse::from_positions(vec![position]);

        let scenario = Scenario {
            dt_years: vec![],
            tau_driver: vec![],
            iv_mult_path: vec![],
            s_path: vec![],
            max_expire: expiry,
            var_cum: vec![0.0],
        };

        let err = Simulator::new()
            .run(&context, &universe, &scenario)
            .expect_err("expected error for zero-step scenario");
        assert!(
            matches!(err, SimulatorError::InvalidInput(_)),
            "expected invalid input error"
        );
    }

    #[test]
    fn run_pre_expiry_matches_manual_surface_pricing() {
        let context = load_context();
        let expiry = NaiveDate::from_ymd_opt(2025, 9, 12).unwrap();
        let builder = LegBuilder::new(&context);
        let call = builder
            .create(OptionType::Call, 150.0, expiry)
            .expect("missing call leg");
        let put = builder
            .create(OptionType::Put, 150.0, expiry)
            .expect("missing put leg");

        let mut position = Position::new();
        position.push(call, 1);
        position.push(put, 1);
        let premium = position.premium;
        let universe = LegUniverse::from_positions(vec![position]);

        let s = 145.0;
        let tau = 0.25;
        let iv_mult = 1.3;
        let scenario = Scenario {
            dt_years: vec![0.0],
            tau_driver: vec![tau],
            iv_mult_path: vec![iv_mult],
            s_path: vec![s],
            max_expire: expiry,
            var_cum: vec![0.0, 0.0],
        };

        let metrics = Simulator::new()
            .run(&context, &universe, &scenario)
            .expect("simulation returned no metrics");
        assert_eq!(metrics.len(), 1);

        let scale = iv_mult * iv_mult;

        let k_call = call.strike.ln() - s.ln();
        let w_call =
            interp_linear_kgrid(k_call, &context.vol_surface.row(OptionType::Call, tau)) * scale;
        let iv_call = (w_call / tau).sqrt();
        let call_mark = bs_price(OptionType::Call, s, call.strike, tau, iv_call);

        let k_put = put.strike.ln() - s.ln();
        let w_put =
            interp_linear_kgrid(k_put, &context.vol_surface.row(OptionType::Put, tau)) * scale;
        let iv_put = (w_put / tau).sqrt();
        let put_mark = bs_price(OptionType::Put, s, put.strike, tau, iv_put);

        let total_mark = call_mark + put_mark;
        let expected_value = total_mark - premium;
        let risk = (premium - total_mark).max(0.0);

        assert_close(
            metrics[0].expected_value,
            expected_value,
            1e-10,
            "expected_value",
        );
        assert_close(metrics[0].risk, risk, 1e-10, "risk");
    }

    #[test]
    fn run_post_expiry_uses_intrinsic_and_tracks_worst_min() {
        let context = load_context();
        let expiry = NaiveDate::from_ymd_opt(2025, 9, 12).unwrap();
        let builder = LegBuilder::new(&context);
        let call = builder
            .create(OptionType::Call, 150.0, expiry)
            .expect("missing call leg");

        let mut long_call = Position::new();
        long_call.push(call, 1);
        let long_premium = long_call.premium;

        let mut short_call = Position::new();
        short_call.push(call, -1);
        let short_premium = short_call.premium;

        let universe = LegUniverse::from_positions(vec![long_call, short_call]);

        let scenario = Scenario {
            dt_years: vec![0.0, 0.0, 0.0],
            tau_driver: vec![0.0, 0.0, 0.0],
            iv_mult_path: vec![1.0, 1.0, 1.0],
            // Two paths, three steps each. With tau exhausted at step zero, intrinsic
            // is locked from the first step of each path and stays flat.
            s_path: vec![140.0, 180.0, 200.0, 170.0, 160.0, 150.0],
            max_expire: expiry,
            var_cum: vec![0.0, 0.0, 0.0, 0.0],
        };

        let metrics = Simulator::new()
            .run(&context, &universe, &scenario)
            .expect("simulation returned no metrics");
        assert_eq!(metrics.len(), 2);

        let intrinsic_path_1 = (140.0_f64 - call.strike).max(0.0);
        let intrinsic_path_2 = (170.0_f64 - call.strike).max(0.0);
        let long_avg_last = (intrinsic_path_1 + intrinsic_path_2) * 0.5;
        let long_worst_min = intrinsic_path_1.min(intrinsic_path_2);

        let expected_long_ev = long_avg_last - long_premium;
        let expected_long_risk = (long_premium - long_worst_min).max(0.0);
        assert_close(
            metrics[0].expected_value,
            expected_long_ev,
            1e-10,
            "long expected_value",
        );
        assert_close(metrics[0].risk, expected_long_risk, 1e-10, "long risk");

        let short_avg_last = -long_avg_last;
        let short_worst_min = (-intrinsic_path_1).min(-intrinsic_path_2);
        let expected_short_ev = short_avg_last - short_premium;
        let expected_short_risk = (short_premium - short_worst_min).max(0.0);
        assert_close(
            metrics[1].expected_value,
            expected_short_ev,
            1e-10,
            "short expected_value",
        );
        assert_close(metrics[1].risk, expected_short_risk, 1e-10, "short risk");
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
