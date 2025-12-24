use crate::{OptionChain, OptionChainSide, OptionType, Pchip};
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
const K_GRID_STEP: f64 = linspace_step(K_N, K_MIN, K_MAX);

enum TauMonoMode {
    None,
    Taper,
    Soft,
    Hard,
}

pub struct VolSurface {
    pub spot: f64,
    calls: TauLut,
    puts: TauLut,
}

impl VolSurface {
    pub fn new(option_chain: &OptionChain) -> Self {
        let spot = option_chain.spot;

        // Precompute LUTs from the already bucketed chain sides.
        let calls = TauLut::from_side(&option_chain.calls);
        let puts = TauLut::from_side(&option_chain.puts);

        Self {
            spot,
            calls,
            puts,
        }
    }

    pub fn row<'a>(&'a self, side: OptionType, tau: f64) -> Cow<'a, [f64]> {
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

    pub fn iv(&self, side: OptionType, tau: f64, strike: f64) -> f64 {
        let k = (strike / self.spot).ln();
        let w_row = self.row(side, tau);

        let w = interp_linear(k, &K_GRID, &w_row);
        (w / tau.max(EPSILON)).sqrt().max(EPSILON)
    }

    pub fn iv_slope(&self, side: OptionType, tau: f64, k_target: f64) -> f64 {
        let w_row = self.row(side, tau);
        let dw_dk = gradient_uniform(&w_row, K_GRID_STEP);
        let k_eval = k_target.clamp(K_MIN, K_MAX);
        let w_val = interp_linear(k_eval, &K_GRID, &w_row);
        let dw_val = interp_linear(k_eval, &K_GRID, &dw_dk);
        let sigma = (w_val.max(EPSILON) / tau).sqrt();

        dw_val / (2.0 * sigma * tau)
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
}

impl TauLut {
    fn from_side(side: &OptionChainSide) -> Self {
        let mut lut = Self {
            tau: side.tau().to_vec(),
            w_raw_lut: Vec::new(),
            w_mon_lut: Vec::new(),
            w_raw: Vec::new(),
            w_mon: Vec::new(),
            tmin: 0.0,
            tmax: 0.0,
        };
        lut.build(side);
        lut
    }

    fn build(&mut self, side: &OptionChainSide) {
        self.tau = side.tau().to_vec();
        let n_tau = self.tau.len();
        let k_n = K_GRID.len();

        self.w_raw.clear();
        // [n_tau * K_N]
        self.w_raw.reserve(n_tau * k_n);
        for bucket in side.slices() {
            let w: Vec<_> = bucket.iv.iter()
                .map(|&iv_i| iv_i * iv_i * bucket.tau)
                .collect();
            let pchip = Pchip::new(bucket.k, w.as_slice())
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

fn gradient_uniform(f: &[f64], h: f64) -> Vec<f64> {
    let n = f.len();
    assert!(n >= 2);

    let mut grad = Vec::with_capacity(n);
    grad.push((f[1] - f[0]) / h);
    if n > 2 {
        let inv_2h = 1.0 / (2.0 * h);
        for w in f.windows(3) {
            grad.push((w[2] - w[0]) * inv_2h);
        }
    }
    grad.push((f[n-1] - f[n-2]) / h);
    grad
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

    let step = linspace_step(n, start, end);
    let mut i = 0;
    while i < n {
        out[i] = start + step * (i as f64);
        i += 1;
    }
}

const fn linspace_step(n: usize, start: f64, end: f64) -> f64 {
    (end - start) / ((n - 1) as f64)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_option_chain::parse_option_chain_file;
    use chrono::NaiveDate;
    use std::path::PathBuf;

    const ARM_CHAIN_PATH: &str = "tests/fixtures/ARM_option_chain_20250908_160038.json";

    fn load_arm_chain() -> OptionChain {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(ARM_CHAIN_PATH);
        let raw = parse_option_chain_file(path).expect("failed to load ARM chain fixture");
        OptionChain::from_raw(&raw)
    }

    fn build_surface(chain: &OptionChain) -> VolSurface {
        VolSurface::new(chain)
    }

    fn tau_from(chain: &OptionChain, year: i32, month: u32, day: u32) -> f64 {
        let exp = NaiveDate::from_ymd_opt(year, month, day).expect("valid expiry date");
        (exp - chain.date).num_days() as f64 / 365.0
    }

    #[test]
    fn iv_slope_matches_reference() {
        let chain = load_arm_chain();
        let surface = build_surface(&chain);
        let spot = chain.spot;

        // Reference slopes pulled from the Python helper `_iv_slope` in
        // src/probability_vol_110.py using the same ARM option chain fixture.
        let cases = [
            (
                OptionType::Call,
                140.0_f64,
                tau_from(&chain, 2025, 10, 17),
                -0.304_397_616_381_687_f64,
            ),
            (
                OptionType::Put,
                120.0_f64,
                tau_from(&chain, 2025, 11, 21),
                -0.205_017_703_325_62_f64,
            ),
            (OptionType::Call, 150.0_f64, 0.25_f64, -0.070_338_261_531_257_f64),
            (OptionType::Put, 110.0_f64, 0.75_f64, -0.020_314_541_565_288_f64),
        ];

        let mut failures = Vec::new();
        for (side, strike, tau, expected) in cases {
            let k_target = (strike / spot).ln();
            let slope = surface.iv_slope(side, tau, k_target);
            let diff = (slope - expected).abs();
            if diff >= 1e-9 {
                failures.push(format!(
                    "side {side:?} strike {strike} tau {tau} expected {expected} got {slope} diff {diff}"
                ));
            }
        }

        if !failures.is_empty() {
            panic!("IV slope mismatches:\\n{}", failures.join("\\n"));
        }
    }

    #[test]
    fn iv_matches_values() {
        let chain = load_arm_chain();
        let surface = build_surface(&chain);

        let cases = [
            (OptionType::Call, 140.0, tau_from(&chain, 2025, 10, 17), 0.439_831_972_631_664_07_f64),
            (OptionType::Put, 120.0, tau_from(&chain, 2025, 11, 21), 0.519_075_134_542_538_2_f64),
            (OptionType::Call, 150.0, 0.25_f64, 0.478_973_206_992_695_74_f64),
        ];

        let mut failures = Vec::new();
        for (side, strike, tau, expected) in cases {
            let iv = surface.iv(side, tau, strike);
            let diff = (iv - expected).abs();
            if diff >= 1e-9 {
                failures.push(format!(
                    "side {side:?} strike {strike} tau {tau} expected {expected} got {iv} (diff {diff})"
                ));
            }
        }

        if !failures.is_empty() {
            panic!("IV mismatches:\\n{}", failures.join("\\n"));
        }
    }

    #[test]
    fn call_row_matches_slice() {
        let chain = load_arm_chain();
        let surface = build_surface(&chain);
        let tau = tau_from(&chain, 2026, 4, 17);

        let row = surface.row(OptionType::Call, tau);

        let samples = [
            (40_usize, -0.5_f64, 0.197_407_263_743_690_89_f64),
            (50, 0.0, 0.135_655_794_423_802_82_f64),
            (60, 0.5, 0.081_273_305_997_084_83_f64),
            (70, 1.0, 1e-12_f64),
        ];

        for (idx, k, expected) in samples {
            let got = row[idx];
            let diff = (got - expected).abs();
            assert!(
                diff < 1e-12,
                "k_grid[{idx}] (k={k}) expected {expected} got {got} diff {diff}"
            );
        }

        let expected_slice = [
            0.142_937_574_337_924_09_f64,
            0.148_229_911_352_458_75_f64,
            0.142_002_465_679_673_9_f64,
            0.137_476_747_958_100_5_f64,
            0.142_719_370_225_230_98_f64,
            0.135_655_794_423_802_82_f64,
            0.139_390_358_882_453_46_f64,
            0.137_884_516_362_119_12_f64,
            0.130_129_085_272_258_92_f64,
            0.129_156_548_286_169_8_f64,
            0.128_218_596_802_933_46_f64,
        ];

        let start = 45;
        for (offset, expected) in expected_slice.iter().enumerate() {
            let idx = start + offset;
            let got = row[idx];
            let diff = (got - expected).abs();
            assert!(
                diff < 1e-8,
                "row[{idx}] expected {expected} got {got} diff {diff}"
            );
        }
    }
}
