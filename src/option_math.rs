//! Option math helpers including NCDF, Black-Scholes, and path simulation.

use crate::OptionType;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;

pub fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26; max error ~1.5e-7
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();

    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;

    let t = 1.0 / (1.0 + p * ax);
    let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    let y = 1.0 - poly * (-ax * ax).exp();

    sign * y
}

pub fn ncdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))
}

pub fn bs_price(option_type: OptionType, s: f64, k: f64, t: f64, sigma: f64) -> f64 {
    let t = t.max(1e-12);
    let sigma = sigma.max(1e-8);
    let vs_t = sigma * t.sqrt();
    let d1 = ((s / k).ln() + 0.5 * sigma * sigma * t) / vs_t;
    let d2 = d1 - vs_t;

    match option_type {
        OptionType::Call => s * ncdf(d1) - k * ncdf(d2),
        OptionType::Put => k * ncdf(-d2) - s * ncdf(-d1),
    }
}

pub fn simulate_paths(
    s0: f64,
    drift: &[f64],
    volatility: &[f64],
    paths: usize,
    seed: Option<u64>,
) -> Vec<f64> {
    assert_eq!(drift.len(), volatility.len());

    let steps = drift.len();
    let n = paths.checked_mul(steps).expect("paths*steps overflow");
    let lof_s0 = s0.ln();

    let mut rng = match seed {
        Some(s) => Xoshiro256PlusPlus::seed_from_u64(s),
        None => Xoshiro256PlusPlus::from_os_rng(),
    };

    let mut shocks = Vec::with_capacity(n);

    for _ in 0..paths {
        let mut log_price = lof_s0;
        for (&vol, &mu) in volatility.iter().zip(drift) {
            let rand: f64 = rng.sample(StandardNormal);
            log_price += rand * vol + mu;
            shocks.push(log_price.exp());
        }
    }

    shocks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn ncdf_scalar_outputs() {
        let cases = [
            (-3.0, 0.0013498980316301035),
            (-1.0, 0.15865525393145707),
            (0.0, 0.5),
            (1.0, 0.8413447460685429),
            (3.0, 0.9986501019683699),
        ];

        for (x, expected) in cases {
            let got = ncdf(x);
            assert!(
                approx_eq(got, expected, 2e-7),
                "ncdf({x}) expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn bs_price_prob_vol_reference() {
        let cases = [
            (
                OptionType::Call,
                100.0,
                105.0,
                0.5,
                0.2,
                3.617_973_846_316_467,
            ),
            (
                OptionType::Put,
                50.0,
                45.0,
                1.2,
                0.35,
                4.988_532_982_757_931,
            ),
            (
                OptionType::Call,
                1.0,
                1.0,
                1e-16,
                1e-9,
                3.996_802_888_650_563_5e-15,
            ),
            (
                OptionType::Put,
                120.0,
                150.0,
                2.0,
                0.6,
                60.793_149_423_034_31,
            ),
        ];

        for (option_type, s, k, t, sigma, expected) in cases {
            let got = bs_price(option_type, s, k, t, sigma);
            assert!(
                approx_eq(got, expected, 2e-5),
                "bs_price({option_type:?}, {s}, {k}, {t}, {sigma}) expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn simulate_paths_zero_vol() {
        let drift = [0.01, 0.02, -0.03];
        let volatility = [0.0, 0.0, 0.0];
        let expected = [
            50.502_508_354_208_39,
            51.522_726_697_675_83,
            50.0,
            50.502_508_354_208_39,
            51.522_726_697_675_83,
            50.0,
        ];

        let shocks = simulate_paths(50.0, &drift, &volatility, 2, Some(123));

        assert_eq!(shocks.len(), expected.len());
        for (idx, (&got, &exp)) in shocks.iter().zip(expected.iter()).enumerate() {
            assert!(
                approx_eq(got, exp, 1e-12),
                "shocks[{idx}] expected {exp}, got {got}"
            );
        }
    }
}
