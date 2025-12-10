use crate::{OptionChain, OptionType, Pchip};
use std::borrow::Cow;

const EPSILON: f64 = 1e-12;

// Grid in log-moneyness k = ln(K/spot)
const K_MIN: f64 = -2.5;
const K_MAX: f64 = 2.5;
const K_N: usize = 101;
const TAU_LUT_SIZE: usize = 1024;
const TAU_MONO_MODE: TauMonoMode = TauMonoMode::Taper;
const TAU_NODE_ATOL: f64 = 1e-10;
const TAU_TAPER_WINDOW: f64 = 24.0 / (24.0 * 365.0);  // ~24 hours in year fraction

const K_GRID: [f64; K_N] = linspace_array::<K_N>(K_MIN, K_MAX);

enum TauMonoMode {
    None,
    Taper,
    Soft,
    Hard,
}

pub struct VolSurface {
    spot: f64,
    calls: TauLut,
    puts: TauLut,
}

impl VolSurface {
    pub fn new(option_chain: &OptionChain) -> Self {
        let spot = option_chain.last_price;
        let chain_date = option_chain.date;

        let n_calls = option_chain.data.iter().filter(|c| matches!(c.option_type, OptionType::Call)).count();
        let n_puts = option_chain.data.len() - n_calls;

        let mut calls = TauLut::with_capacity(n_calls);
        let mut puts = TauLut::with_capacity(n_puts);

        for contract in &option_chain.data {
            let tau = (contract.expiration - chain_date).num_days() as f64 / 365.0;
            let k = (contract.strike / spot).ln();
            let iv = contract.implied_volatility;

            match contract.option_type {
                OptionType::Call => calls.push(tau, k, iv),
                OptionType::Put => puts.push(tau, k, iv),
            }
        }

        Self {
            spot,
            calls,
            puts,
        }
    }

    fn row_fast<'a>(&'a self, side: OptionType, tau: f64) -> Cow<'a, [f64]> {
        let lut = match side {
            OptionType::Call => &self.calls,
            OptionType::Put => &self.puts,
        };

        //
        // exact match => return row directly
        //
        let closest_idx = match lut.tau.binary_search_by(|x| x.total_cmp(&tau)) {
            Ok(idx) => idx,
            Err(0) => 0,
            Err(idx) if idx >= lut.tau.len() => lut.tau.len() - 1,
            Err(idx) => {
                let left = lut.tau[idx - 1];
                let right = lut.tau[idx];
                if (tau - left).abs() <= (right - tau).abs() {
                    idx - 1
                } else {
                    idx
                }
            }
        };

        let closest_dist = (lut.tau[closest_idx] - tau).abs();
        if matches!(TAU_MONO_MODE, TauMonoMode::None) || closest_dist < TAU_NODE_ATOL {
            return (&lut.w_raw[closest_idx * K_N..(closest_idx + 1) * K_N]).into();
        }

        //
        // Extrapolation
        //
        if tau < lut.tmin {
            return (&lut.w_raw[0..K_N]).into();
        }
        if tau > lut.tmax {
            return (&lut.w_raw[(lut.tau.len() - 1) * K_N..lut.tau.len() * K_N]).into();
        }

        //
        // Within range: linear interp on LUT between nearest grid points
        //
        let tau_grid_pos = (tau - lut.tmin) * (TAU_LUT_SIZE as f64 - 1.0) / (lut.tmax - lut.tmin).max(EPSILON);
        // position in tau LUT grid
        let i0 = tau_grid_pos.floor() as usize;
        let i1 = (i0 + 1).min(TAU_LUT_SIZE - 1);
        let alpha = tau_grid_pos - (i0 as f64);
        let one_minus_alpha = 1.0 - alpha;

        let raw0 = &lut.w_raw_lut[i0 * K_N..(i0 + 1) * K_N];
        let raw1 = &lut.w_raw_lut[i1 * K_N..(i1 + 1) * K_N];

        if matches!(TAU_MONO_MODE, TauMonoMode::None) {
            let w_raw: Vec<_> = raw0.iter()
                .zip(raw1.iter())
                .map(|(&v0, &v1)| one_minus_alpha * v0 + alpha * v1)
                .collect();
            return w_raw.into();
        }

        //
        // 'taper': blend uplift away from the nearest node
        //
        // 0 at node → 1 beyond taper window
        let phi = (closest_dist / TAU_TAPER_WINDOW).min(1.0);
        let mon0 = &lut.w_mon_lut[i0 * K_N..(i0 + 1) * K_N];
        let mon1 = &lut.w_mon_lut[i1 * K_N..(i1 + 1) * K_N];

        // evaluate Wm at tau by LUT
        let w_tapered: Vec<_> = raw0.iter()
            .zip(raw1.iter())
            .zip(mon0.iter().zip(mon1.iter()))
            .map(|((&r0, &r1), (&m0, &m1))| {
                let w_r = one_minus_alpha * r0 + alpha * r1;
                let w_m = one_minus_alpha * m0 + alpha * m1;
                (w_r + phi * (w_m - w_r)).max(EPSILON)
            })
            .collect();
        
        w_tapered.into()
    }

    fn iv(&self, side: OptionType, tau: f64, strike: f64) -> f64 {
        let k = (strike / self.spot).ln();
        let w_row = self.row_fast(side, tau);

        let w = interp_linear(k, &K_GRID, &w_row);
        w.sqrt().max(EPSILON)
    }

    /// Compute risk-neutral forward F(T), discount factor D(T) ≈ exp(-rT),
    /// short rate r, and GBM drift mu = ln(F/S0)/T from the option chain at this tau.
    ///
    /// Returns: (F, D, r, mu). On failure returns (nan, nan, nan, nan).
    fn implied_forward_and_mu(&self, tau: f64) {


    }


}

struct TauLut {
    pub tau: Vec<f64>,
    pub w_raw_lut: Vec<f64>,
    pub w_mon_lut: Vec<f64>,
    pub w_raw: Vec<f64>,
    pub w_mon: Vec<f64>,
    pub tmin: f64,
    pub tmax: f64,
    tau_range: Vec<(usize, usize)>,
    k: Vec<f64>,
    iv: Vec<f64>,
    index: usize,
    last_tau: Option<f64>,
}

impl TauLut {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            tau: Vec::with_capacity(capacity),
            tau_range: Vec::with_capacity(capacity),
            w_raw_lut: Vec::new(),
            w_mon_lut: Vec::new(),
            w_raw: Vec::new(),
            w_mon: Vec::new(),
            tmin: 0.0,
            tmax: 0.0,
            k: Vec::with_capacity(capacity),
            iv: Vec::with_capacity(capacity),
            index: 0,
            last_tau: None,
        }
    }

    fn push(&mut self, tau: f64, k: f64, iv: f64) {
        let is_new_tau = self.last_tau.map_or(true, |last_tau| (tau - last_tau).abs() > EPSILON);

        if is_new_tau {
            self.tau.push(tau);
            self.tau_range.push((self.index, self.index + 1));
            self.last_tau = Some(tau);
        } else if let Some(last) = self.tau_range.last_mut() {
            last.1 = self.index + 1;
        }
        self.k.push(k);
        self.iv.push(iv);
        self.index += 1;
    }

    fn build(&mut self) {
        let n_tau = self.tau.len();
        let k_n = K_GRID.len();

        self.w_raw.clear();
        // [n_tau * K_N]
        self.w_raw.reserve(n_tau * k_n);
        for (&t, range) in self.tau.iter().zip(self.tau_range.iter()) {
            let k = &self.k[range.0..range.1];
            let iv = &self.iv[range.0..range.1];

            let w: Vec<_> = iv.iter()
                .map(|&iv_i| iv_i * iv_i * t)
                .collect();
            let pchip = Pchip::new(k, w.as_slice())
                .expect("Failed to create PCHIP interpolant");

            self.w_raw.extend(K_GRID.iter().map(|&k| pchip.eval(k).max(EPSILON)));
        }

        // Turn W_raw into column-major (transpose) for per-strike monotonic enforcement.
        // [K_N * n_tau]
        let w_raw_t: Vec<f64> = (0..k_n)
            .flat_map(|col_idx| self.w_raw.iter().skip(col_idx).step_by(k_n))
            .copied()
            .collect();

        // Enforce monotonicity in τ for each k by cumulative max down each column.
        // [K_N * n_tau]
        self.w_mon = w_raw_t
            .chunks_exact(n_tau)
            .flat_map(|col| {
                let mut running_max = f64::NEG_INFINITY;
                col.iter().map(move |&v| {
                    running_max = running_max.max(v);
                    running_max
                })
            })
            .collect();

        self.tmin = self.tau.first().map_or(0.0, |&t| t);
        self.tmax = self.tau.last().map_or(0.0, |&t| t);
        let tau_grid = linspace_vec(self.tmin, self.tmax, TAU_LUT_SIZE);

        // [TAU_LUT_SIZE * K_N] row-major for fast row access
        self.w_raw_lut = self.to_lut(&w_raw_t, (k_n, n_tau), tau_grid.as_slice());
        self.w_mon_lut = self.to_lut(self.w_mon.as_slice(), (k_n, n_tau), tau_grid.as_slice());
        
    }

    /// Create a lookup table by evaluating PCHIP interpolants at given tau_grid.
    /// The input w is in column-major order with shape (k_n, n_tau).
    /// The output LUT is in row-major order with shape (TAU_LUT_SIZE, k_n).
    fn to_lut(&self, w: &[f64], size: (usize, usize), tau_grid: &[f64]) -> Vec<f64> {
        let (n_cols, col_len) = size;
        assert_eq!(w.len(), n_cols * col_len);

        let grid_len = tau_grid.len();
        let mut lut = vec![0.0; grid_len * n_cols];

        for (col_idx, col) in w.chunks_exact(col_len).enumerate() {
            let pchip = Pchip::new(self.tau.as_slice(), col)
                .expect("Failed to create PCHIP interpolant for LUT");
            for (row_idx, &t) in tau_grid.iter().enumerate() {
                let v = pchip.eval(t).max(EPSILON);
                let idx = row_idx * n_cols + col_idx; // row-major (tau_grid x k)
                lut[idx] = v;
            }
        }

        lut
    }
}

fn interp_linear(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    assert!(n >= 2, "xp must have at least two points");
    
    let high = match xp.binary_search_by(|v| v.total_cmp(&x)) {
        Ok(i) => i,
        Err(i) => i.min(n - 1),
    };

    let low = high.saturating_sub(1);
    if low == high {
        return fp[low];
    }
    let t = (x - xp[low]) / (xp[high] - xp[low]).max(EPSILON);
    fp[low] + t * (fp[high] - fp[low])
}

const fn fill_linspace(out: &mut [f64], start: f64, end: f64) {
    let n = out.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        out[0] = start;
        return;
    }

    let step = (end - start) / ((n - 1) as f64);
    let mut i = 0;
    while i < n {
        out[i] = start + step * (i as f64);
        i += 1;
    }
}

// Array version (compile-time size)
pub const fn linspace_array<const N: usize>(start: f64, end: f64) -> [f64; N] {
    let mut out = [0.0; N];
    fill_linspace(&mut out, start, end);
    out
}

// Vec version (runtime size)
pub fn linspace_vec(start: f64, end: f64, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    fill_linspace(&mut out, start, end);
    out
}
