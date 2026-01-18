//! Price legs and positions along simulated scenario paths.

use crate::{Context, LegUniverse, Scenario, OptionType, bs_price, interp_linear_kgrid, linspace_vec};
use rayon::prelude::*;
use std::borrow::Cow;

const SECONDS_PER_YEAR: f64 = 365.0 * 24.0 * 3600.0;
const EPSILON: f64 = 1e-8;

pub struct Simulator {
    exit_grid: Vec<f64>,
    commission_per_trade: f64,
    close_slippage_frac: f64,
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
    pub fn run(&self, context: &Context, universe: &LegUniverse, scenario: &Scenario) -> Vec<f64> {
        let calendar = &context.calendar;
        let vol_surface = &context.vol_surface;
        let s_path = &scenario.s_path;
        let iv_mult_path = &scenario.iv_mult_path;
        let tau_driver = &scenario.tau_driver;
        let steps = scenario.tau_driver.len();
        let leg_count = universe.legs.len();
        let max_expire = universe.max_expire;
        let (_, max_expiry_close) = calendar.session(max_expire);
        if steps == 0 || leg_count == 0 {
            return Vec::new();
        }
        // Scenario paths are stored as contiguous [path][step] blocks.
        let paths = s_path.len() / steps;
        if paths == 0 {
            return Vec::new();
        }
        let row_stride = vol_surface.row_len();
        if row_stride == 0 {
            return Vec::new();
        }

        //
        // Evaluate legs
        //

        // [leg][path][step]
        let mut values: Vec<f64> = vec![0.0; leg_count * paths * steps];

        // Parallelize across path chunks per leg to keep work balanced without nested pools.
        let leg_stride = paths * steps;
        let threads = rayon::current_num_threads().max(1);
        // Chunk by paths to keep per-thread work sizable.
        let chunk_paths = (paths / (threads * 4).max(1)).max(1);
        let chunk_len = chunk_paths * steps;
        // Track the absolute leg offset; expiry_slices() yields legs in the same stable order as universe.legs.
        let mut leg_idx = 0;

        // Walk expiries in order so we can align each slice to the common tau grid and reuse precomputed rows.
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
            let n = step_data.len();

            for (slice_leg_idx, leg) in expire_slice.legs.iter().enumerate() {
                let leg_vals = &mut values[leg_idx * leg_stride..(leg_idx + 1) * leg_stride];

                leg_vals
                    .par_chunks_mut(chunk_len)
                    .enumerate()
                    .for_each(|(chunk_idx, v_chunk)| {
                        // Work on a handful of paths at a time to balance parallel work.
                        let path_base = chunk_idx * chunk_paths;
                        let paths_in_chunk = v_chunk.len() / steps;

                        for path_local in 0..paths_in_chunk {
                            let global_path = path_base + path_local;
                            let s_slice = &s_path[global_path * steps..(global_path + 1) * steps];
                            let v_slice = &mut v_chunk[path_local * steps..(path_local + 1) * steps];

                            for ((&s, v), (tau, rows_flat)) in s_slice
                                .iter()
                                .zip(v_slice.iter_mut())
                                .zip(step_data.iter())
                            {
                                let offset = slice_leg_idx * row_stride;
                                let w_row = &rows_flat[offset..offset + row_stride];
                                // Log-moneyness drives the vol-surface lookup.
                                let k = (leg.strike / s).ln();
                                // w is total variance at (k, tau); convert to IV before pricing.
                                let w = interp_linear_kgrid(k, w_row);
                                let iv = (w / tau).sqrt();
                                // Mark-to-market the leg using slice-specific rows and path price.
                                *v = bs_price(leg.option_type, s, leg.strike, *tau, iv);
                            }

                            if n < steps {
                                // After expiry, keep the terminal payoff flat through the remaining steps.
                                // This models European intrinsic value after expiration.
                                let s = s_slice[n];
                                let value_at_expire = match leg.option_type {
                                    OptionType::Call => (s - leg.strike).max(0.0),
                                    OptionType::Put => (leg.strike - s).max(0.0),
                                };
                                for v in &mut v_slice[n..] {
                                    *v = value_at_expire;
                                }
                            }
                        }
                    });
                // Safe to advance: each slice covers a disjoint, ordered range of legs.
                leg_idx += 1;
            }
        }

        //
        // Combine legs into postion values
        //
        let positions_idx = &universe.positions_idx;
        let pos_count = positions_idx.len();
        // [position][path][step]
        let mut pos_values: Vec<f64> = vec!(0.0; pos_count * leg_stride);
        for (pos_idx, legs) in positions_idx.iter().enumerate() {
            let dst = &mut pos_values[pos_idx * leg_stride..(pos_idx + 1) * leg_stride];
            for &(leg_idx, qty) in legs.iter() {
                let src = &values[leg_idx * leg_stride..(leg_idx + 1) * leg_stride];
                // Aggregate leg marks by quantity for each position.
                for (d, s) in dst.iter_mut().zip(src.iter()) {
                    *d += s * qty as f64;
                }
            }
        }
        pos_values
    }

    fn compute_pt_sl_stats(&self, universe: &LegUniverse, pos_values: &[f64]) {
        // let v0: Vec<_> = universe.positions.iter()
        //     .map(|pos| pos.premium)
        //     .collect();



    }
    
}
