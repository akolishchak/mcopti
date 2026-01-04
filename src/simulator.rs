use crate::{Context, LegUniverse, Scenario, interp_linear_kgrid, bs_price};
use rayon::prelude::*;

pub struct Simulator {}

impl Simulator {
    pub fn run(context: &Context, universe: &LegUniverse, scenario: &Scenario) {
        let vol_surface = &context.vol_surface;
        let s_path = &scenario.s_path;
        let iv_mult_path = &scenario.iv_mult_path;
        let tau_driver = &scenario.tau_driver;
        let steps = scenario.tau_driver.len();
        if steps == 0 {
            return;
        }
        let mut value: Vec<f64> = vec![0.0; s_path.len()];
        let paths = s_path.len() / steps;

        // precompute scaled W rows per step/leg
        let step_data: Vec<(f64, Vec<Vec<f64>>)> = tau_driver
            .iter()
            .zip(iv_mult_path.iter())
            .map(|(&tau, &iv_mult)| {
                let scale = iv_mult * iv_mult;
                let rows = universe
                    .legs
                    .iter()
                    .map(|leg| {
                        vol_surface
                            .row(leg.option_type, tau)
                            .iter()
                            .map(|&w| w * scale)
                            .collect::<Vec<f64>>()
                    })
                    .collect::<Vec<Vec<f64>>>();

                (tau, rows)
            })
            .collect();

        // chunk by groups of paths to balance Rayon scheduling overhead and cache locality
        let threads = rayon::current_num_threads().max(1);
        let chunk_paths = (paths / (threads * 4).max(1)).max(1);
        let chunk_len = chunk_paths * steps;

        value
            .par_chunks_mut(chunk_len)
            .zip(s_path.par_chunks(chunk_len))
            .for_each(|(v_chunk, s_chunk)| {
                let paths_in_chunk = v_chunk.len() / steps;
                for path_idx in 0..paths_in_chunk {
                    let v_slice = &mut v_chunk[path_idx * steps..(path_idx + 1) * steps];
                    let s_slice = &s_chunk[path_idx * steps..(path_idx + 1) * steps];

                    for ((&s, v), (tau, leg_rows)) in s_slice
                        .iter()
                        .zip(v_slice.iter_mut())
                        .zip(step_data.iter())
                    {
                        for (leg, w_row) in universe.legs.iter().zip(leg_rows.iter()) {
                            let k = (leg.strike / s).ln();
                            let w = interp_linear_kgrid(k, w_row);
                            let iv = (w / tau).sqrt();
                            *v = bs_price(leg.option_type, s, k, *tau, iv);
                        }
                    }
                }
            });
    }
    
}
