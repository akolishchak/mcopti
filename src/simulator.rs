use crate::{Context, LegUniverse, Scenario, interp_linear_kgrid, bs_price};
use rayon::prelude::*;

pub struct Simulator {}

impl Simulator {
    // Returns leg-major values laid out as [leg][path][step] (step is the innermost stride).
    pub fn run(context: &Context, universe: &LegUniverse, scenario: &Scenario) -> Vec<f64> {
        let vol_surface = &context.vol_surface;
        let s_path = &scenario.s_path;
        let iv_mult_path = &scenario.iv_mult_path;
        let tau_driver = &scenario.tau_driver;
        let steps = scenario.tau_driver.len();
        let leg_count = universe.legs.len();
        if steps == 0 || leg_count == 0 {
            return Vec::new();
        }
        let paths = s_path.len() / steps;
        if paths == 0 {
            return Vec::new();
        }
        let first_leg = &universe.legs[0];
        let first_tau = tau_driver[0];
        let row_stride = vol_surface.row(first_leg.option_type, first_tau).len();
        if row_stride == 0 {
            return Vec::new();
        }

        //
        // Evaluate legs
        //

        // [leg][path][step]
        let mut values: Vec<f64> = vec![0.0; leg_count * paths * steps];

        // Vec over steps -> (tau, leg-major flat rows for that step)
        let step_data: Vec<(f64, Vec<f64>)> = tau_driver
            .iter()
            .zip(iv_mult_path.iter())
            .map(|(&tau, &iv_mult)| {
                let scale = iv_mult * iv_mult;
                let mut rows: Vec<f64> = Vec::with_capacity(leg_count * row_stride);
                for leg in universe.legs.iter() {
                    rows.extend(
                        vol_surface
                            .row(leg.option_type, tau)
                            .iter()
                            .map(|&w| w * scale),
                    );
                }
                (tau, rows)
            })
            .collect();

        // Parallelize across path chunks per leg to keep work balanced without nested pools.
        let leg_stride = paths * steps;
        let threads = rayon::current_num_threads().max(1);
        let chunk_paths = (paths / (threads * 4).max(1)).max(1);
        let chunk_len = chunk_paths * steps;

        for leg_idx in 0..leg_count {
            let leg = &universe.legs[leg_idx];
            let leg_vals = &mut values[leg_idx * leg_stride..(leg_idx + 1) * leg_stride];

            leg_vals
                .par_chunks_mut(chunk_len)
                .enumerate()
                .for_each(|(chunk_idx, v_chunk)| {
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
                            let offset = leg_idx * row_stride;
                            let w_row = &rows_flat[offset..offset + row_stride];
                            let k = (leg.strike / s).ln();
                            let w = interp_linear_kgrid(k, w_row);
                            let iv = (w / tau).sqrt();
                            *v = bs_price(leg.option_type, s, k, *tau, iv);
                        }
                    }
                });
        }

        //
        // Combine legs into postion values
        //
        let positions_idx = &universe.positions_idx;
        let pos_count = positions_idx.len();
        // [leg][path][step]
        let mut pos_values: Vec<f64> = vec!(0.0; pos_count * paths * steps);
        for legs in positions_idx.iter() {
            for &(idx, qty) in legs.iter() {
                let start = idx * leg_stride;
                let end = (idx+1) * leg_stride;
                for (v, p) in values[start..end].iter().zip(pos_values[start..end].iter_mut()) {
                    *p += v * qty as f64;
                }
            }
        }
        pos_values
    }
    
}
