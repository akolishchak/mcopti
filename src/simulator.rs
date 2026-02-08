//! Price legs and positions along simulated scenario paths.

use crate::{Context, LegUniverse, OptionType, Scenario, bs_price, interp_linear_kgrid, linspace_vec, position};
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

    // Returns leg-major values laid out as [leg][path][step] (step is the innermost stride).
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

        // Walk expiries in order so we can align each slice to the common tau grid and reuse precomputed rows.
        let mut expiry_data = Vec::with_capacity(universe.expiries());
        for expire_slice in universe.expiry_slices() {
            let (_, leg_close) = calendar.session(expire_slice.legs.first().unwrap().expire);
            // Align this expiry's timeline with the global driver so tau hits zero when the slice expires.
            let tau_offset = ((max_expiry_close - leg_close).as_seconds_f64() / SECONDS_PER_YEAR).max(0.0);

            // Vec over steps -> (tau, leg-major flat rows for that step)
            let mut step_data = Vec::with_capacity(steps);
            for (tau, iv_mult) in tau_driver.iter().zip(iv_mult_path.iter()) {
                let tau_to_expire = tau - tau_offset;
                // Once tau is exhausted, remaining steps are post-expiry.
                if tau_to_expire <= EPSILON {
                    break;
                }

                // Precompute a contiguous chunk of vol rows scaled by the path's variance multiplier.
                // Rows store total variance, so the scale applies as iv_mult^2.
                let scale = iv_mult * iv_mult;
                let mut rows: Vec<f64> = Vec::with_capacity(expire_slice.legs.len() * row_stride);
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
                step_data.push((tau_to_expire, rows));
            }
            expiry_data.push(step_data);
        }

        // [path][pos] = (mark, drawdown)
        let mut path_pos_values: Vec<(f64, f64)> = vec![(0.0, f64::INFINITY); pos_count * paths];

        // Parallelize across path chunks per leg to keep work balanced without nested pools.
        let threads = rayon::current_num_threads().max(1);
        // Chunk by paths to keep per-thread work sizable.
        let chunk_paths = (paths / (threads * 4).max(1)).max(1);

        path_pos_values
            .par_chunks_mut(chunk_paths * pos_count)
            .enumerate()
            .for_each(|(chunk_idx, path_values_chunk)| {
                // work on a handful of paths at a time to balance parallel work.
                let path0 = chunk_idx * chunk_paths;
                let path1 = (path0 + chunk_paths).min(paths);
                let path_chunk = &s_path[path0 * steps .. path1 * steps];

                let mut leg_marks: Vec<f64> = vec![0.0; leg_count];
                for (path_local, pos_values) in path_chunk
                    .chunks(steps)
                    .zip(path_values_chunk.chunks_mut(pos_count))
                {
                    //
                    // process a path
                    //
                    for (step, &s) in path_local.iter().enumerate() {
                        let ln_s = s.ln();
                        //
                        // mark legs
                        //
                        // track the absolute leg offset; expiry_slices() yields legs in the same stable order as universe.legs.
                        let mut leg_idx = 0;
                        for (expire_slice, step_data) in universe
                            .expiry_slices()
                            .zip(expiry_data.iter())
                        {
                            let steps_to_expiry = step_data.len();
                            if step < steps_to_expiry {
                                let (tau, rows) = &step_data[step];
                                for (slice_leg_idx, leg) in expire_slice.legs.iter().enumerate() {
                                    
                                    let offset = slice_leg_idx * row_stride;
                                    let w_row = &rows[offset..offset + row_stride];
                                    // log-moneyness drives the vol-surface lookup.
                                    let k = ln_strike[leg_idx] - ln_s;
                                    // w is total variance at (k, tau); convert to IV before pricing.
                                    let w = interp_linear_kgrid(k, w_row);
                                    let iv = (w / tau).sqrt();
                                    // mark-to-market the leg using slice-specific rows and path price.
                                    leg_marks[leg_idx] = bs_price(leg.option_type, s, leg.strike, *tau, iv);

                                    // safe to advance: each slice covers a disjoint, ordered range of legs.
                                    leg_idx += 1;
                                }
                            } else {
                                // After expiry, keep the terminal payoff flat through the remaining steps.
                                // This models European intrinsic value after expiration.
                                let s = path_local[steps_to_expiry.saturating_sub(1)];
                                for (_, leg) in expire_slice.legs.iter().enumerate() {
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
                        for ((mark, drawdown), pos_idx) in pos_values
                            .iter_mut()
                            .zip(positions_idx.iter())
                        {
                            *mark = pos_idx
                                        .iter()
                                        .map(|&(leg_idx, qty)| {
                                            (qty as f64) * leg_marks[leg_idx]
                                        })
                                        .sum::<f64>();
                            *drawdown = drawdown.min(*mark);
                        }
                    }
                }
            });

            let mut metrcis = Vec::with_capacity(pos_count);
            for (i, pos) in positions.iter().enumerate() {
                let mut sum_mark = 0.0;
                let mut drawdown = f64::INFINITY;
                for (mark, min_mark) in path_pos_values.iter().skip(i).step_by(pos_count) {
                    sum_mark += *mark;
                    drawdown = drawdown.min(*min_mark);
                }
                let expected_value = pos.premium + sum_mark / paths as f64;
                
                let risk = (pos.premium - drawdown).max(0.0);
                metrcis.push(Metrics {
                    expected_value,
                    risk,
                });
            }
            Some(metrcis)
    }

}
