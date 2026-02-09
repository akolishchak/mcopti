//! Price legs and positions along simulated scenario paths.

use crate::{Context, LegUniverse, OptionType, Scenario, bs_price, interp_linear_kgrid, linspace_vec};
use rayon::prelude::*;
use std::borrow::Cow;

const SECONDS_PER_YEAR: f64 = 365.0 * 24.0 * 3600.0;
const EPSILON: f64 = 1e-8;

pub struct Simulator {
    exit_grid: Vec<f64>,
    commission_per_trade: f64,
    close_slippage_frac: f64,
}

pub struct Metrics {
    pub expected_value: f64,
    pub risk: f64,
}

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
    pub fn run(&self, context: &Context, universe: &LegUniverse, scenario: &Scenario) -> Option<Vec<Metrics>> {
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
            return None;
        }
        // Scenario paths are stored as contiguous [path][step] blocks.
        let paths = s_path.len() / steps;
        if paths == 0 {
            return None;
        }
        let row_stride = vol_surface.row_len();
        if row_stride == 0 {
            return None;
        }

        // walk expiries in order so we can align each slice to the common tau grid and reuse precomputed rows.
        let mut expiry_data: Vec<ExpiryData> = Vec::with_capacity(universe.expiries());
        for expire_slice in universe.expiry_slices() {
            let (_, leg_close) = calendar.session(expire_slice.legs.first().unwrap().expire);
            // Align this expiry's timeline with the global driver so tau hits zero when the slice expires.
            let tau_offset = ((max_expiry_close - leg_close).as_seconds_f64() / SECONDS_PER_YEAR).max(0.0);
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
                let mut row_put:  Option<Cow<'_, [f64]>> = None;

                // Build leg-major row blocks to keep indexing cheap in inner loop.
                for leg in expire_slice.legs.iter() {
                    let src: &[f64] = match leg.option_type {
                        OptionType::Call => row_call.get_or_insert_with(|| vol_surface.row(OptionType::Call, tau_to_expire)),
                        OptionType::Put  => row_put .get_or_insert_with(|| vol_surface.row(OptionType::Put,  tau_to_expire)),
                    };                    
                    rows.extend(
                        src
                        .iter()
                        .map(|&w| w * scale),
                    );
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
                            for (expire_slice, expiry_data) in universe
                                .expiry_slices()
                                .zip(expiry_data.iter())
                            {
                                let steps_to_expiry = expiry_data.taus.len();
                                if step < steps_to_expiry {
                                    let tau = expiry_data.taus[step];
                                    let step_base = step * expiry_data.step_stride;
                                    let rows = &expiry_data.rows[step_base..step_base + expiry_data.step_stride];
                                    for (slice_leg_idx, leg) in expire_slice.legs.iter().enumerate() {
                                        let offset = slice_leg_idx * row_stride;
                                        let w_row = &rows[offset..offset + row_stride];
                                        // log-moneyness drives the vol-surface lookup.
                                        let k = ln_strike[leg_idx] - ln_s;
                                        // w is total variance at (k, tau); convert to IV before pricing.
                                        let w = interp_linear_kgrid(k, w_row);
                                        let iv = (w / tau).sqrt();
                                        // mark-to-market the leg using slice-specific rows and path price.
                                        leg_marks[leg_idx] = bs_price(leg.option_type, s, leg.strike, tau, iv);

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
        Some(metrics)
    }

}
