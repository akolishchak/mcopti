//! Monotone PCHIP interpolation for shape-preserving curves.

use std::{borrow::Cow, error::Error, fmt};

/// Piecewise Cubic Hermite Interpolating Polynomial preserving monotonicity.
#[derive(Clone, Debug)]
pub struct Pchip<'a> {
    x: Cow<'a, [f64]>,
    y: Cow<'a, [f64]>,
    d: Vec<f64>, // slopes at nodes
}

#[derive(Clone, Debug, PartialEq)]
pub enum PchipError {
    LengthMismatch,
    NonIncreasingX,
    NonFiniteInput,
}

impl fmt::Display for PchipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PchipError::LengthMismatch => f.write_str("x and y must have same length >= 2"),
            PchipError::NonIncreasingX => f.write_str("x must be strictly increasing"),
            PchipError::NonFiniteInput => f.write_str("x and y must be finite"),
        }
    }
}

impl Error for PchipError {}

impl<'a> Pchip<'a> {
    /// Build a monotone shape-preserving cubic interpolant.
    pub fn new<T>(x: T, y: T) -> Result<Self, PchipError>
    where
        T: Into<Cow<'a, [f64]>>,
    {
        let x = x.into();
        let y = y.into();

        let n = x.len();
        if n < 2 || y.len() != n {
            return Err(PchipError::LengthMismatch);
        }
        if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
            return Err(PchipError::NonFiniteInput);
        }

        // h and delta
        let mut h = Vec::with_capacity(n - 1);
        let mut delta = Vec::with_capacity(n - 1);
        for (xw, yw) in x.windows(2).zip(y.windows(2)) {
            let hi = xw[1] - xw[0];
            if hi <= 0.0 {
                return Err(PchipError::NonIncreasingX);
            }
            h.push(hi);
            delta.push((yw[1] - yw[0]) / hi);
        }

        let mut d = vec![0.0; n];

        // interior slopes
        for k in 1..n - 1 {
            let s1 = delta[k - 1];
            let s2 = delta[k];

            if s1 == 0.0 || s2 == 0.0 || s1.signum() != s2.signum() {
                d[k] = 0.0;
            } else {
                let h1 = h[k - 1];
                let h2 = h[k];
                let w1 = 2.0 * h2 + h1;
                let w2 = h2 + 2.0 * h1;
                d[k] = (w1 + w2) / (w1 / s1 + w2 / s2); // weighted harmonic mean
            }
        }

        // left endpoint
        d[0] = if n == 2 {
            delta[0]
        } else {
            let h0 = h[0];
            let h1 = h[1];
            let s0 = delta[0];
            let s1 = delta[1];
            let mut d0 = ((2.0 * h0 + h1) * s0 - h0 * s1) / (h0 + h1);
            if d0.signum() != s0.signum() {
                d0 = 0.0;
            } else if s0.signum() != s1.signum() && d0.abs() > 3.0 * s0.abs() {
                d0 = 3.0 * s0;
            }
            d0
        };

        // right endpoint
        d[n - 1] = if n == 2 {
            delta[0]
        } else {
            let h_nm2 = h[n - 2];
            let h_nm3 = h[n - 3];
            let s_nm2 = delta[n - 2];
            let s_nm3 = delta[n - 3];
            let mut dn = ((2.0 * h_nm2 + h_nm3) * s_nm2 - h_nm2 * s_nm3) / (h_nm2 + h_nm3);
            if dn.signum() != s_nm2.signum() {
                dn = 0.0;
            } else if s_nm2.signum() != s_nm3.signum() && dn.abs() > 3.0 * s_nm2.abs() {
                dn = 3.0 * s_nm2;
            }
            dn
        };

        Ok(Pchip { x, y, d })
    }

    /// Evaluate the interpolant at `xq`, extending the endpoint cubic segments for extrapolation
    /// (matches SciPy's `PchipInterpolator(..., extrapolate=True)`).
    /// $p(x) = y_k (1 - t)^2 (1 + 2t) + y_{k+1} t^2 (3 - 2t) + h_k d_k t (1 - t)^2 - h_k d_{k+1} t^2 (1 - t)$
    pub fn eval(&self, xq: f64) -> f64 {
        let n = self.x.len();

        // binary search for interval k such that x[k] <= xq < x[k+1]
        let k = if xq <= self.x[0] {
            0
        } else if xq >= self.x[n - 1] {
            n - 2
        } else {
            self.x.partition_point(|&xi| xi <= xq).saturating_sub(1)
        };

        let xk = self.x[k];
        let xk1 = self.x[k + 1];
        let h = xk1 - xk;
        let t = (xq - xk) / h;

        let yk = self.y[k];
        let yk1 = self.y[k + 1];
        let dk = self.d[k];
        let dk1 = self.d[k + 1];

        let t2 = t * t;
        let t3 = t2 * t;

        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        h00 * yk + h10 * h * dk + h01 * yk1 + h11 * h * dk1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_data_is_reproduced() {
        let p = Pchip::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]).unwrap();
        let y = p.eval(1.5);
        assert!((y - 1.5).abs() < 1e-12);
    }

    #[test]
    fn rejects_non_increasing_x() {
        let err = Pchip::new(vec![0.0, 0.0, 1.0], vec![0.0, 1.0, 2.0]).unwrap_err();
        assert_eq!(err, PchipError::NonIncreasingX);
    }

    #[test]
    fn rejects_non_finite_input() {
        let err = Pchip::new(vec![0.0, f64::NAN], vec![0.0, 1.0]).unwrap_err();
        assert_eq!(err, PchipError::NonFiniteInput);
    }

    // Reference values generated from SciPy's PchipInterpolator (scipy 1.15.3, numpy 2.1.3).
    // Code used to generate the fixtures:
    // import numpy as np
    // from scipy.interpolate import PchipInterpolator
    //
    // rng = np.random.default_rng(12345)
    // x = np.cumsum(rng.uniform(0.2, 1.0, size=8))
    // y = rng.normal(loc=0.0, scale=1.0, size=8)
    // q = np.linspace(x[0], x[-1], num=15)
    // expected = PchipInterpolator(x, y, extrapolate=True)(q)
    const SCIPY_X: [f64; 8] = [
        0.381_868_817_973_735_73,
        0.835_275_489_741_538,
        1.673_167_855_607_725_4,
        2.414_171_592_208_505,
        2.927_059_232_690_032_4,
        3.393_310_374_983_14,
        4.071_957_377_852_892,
        4.421_344_726_335_863,
    ];
    const SCIPY_Y: [f64; 8] = [
        0.361_058_113_054_895,
        -1.952_863_063_012_19,
        2.347_409_654_378_852,
        0.968_496_905_751_923_6,
        -0.759_387_180_424_506_6,
        0.902_198_274_212_251_7,
        -0.466_953_173_320_550_25,
        -0.060_689_518_737_027_98,
    ];
    const SCIPY_QUERIES: [f64; 15] = [
        0.381_868_817_973_735_73,
        0.670_402_811_428_173_4,
        0.958_936_804_882_611,
        1.247_470_798_337_048_8,
        1.536_004_791_791_486_3,
        1.824_538_785_245_923_8,
        2.113_072_778_700_362,
        2.401_606_772_154_799_3,
        2.690_140_765_609_237,
        2.978_674_759_063_674_4,
        3.267_208_752_518_112,
        3.555_742_745_972_55,
        3.844_276_739_426_987_4,
        4.132_810_732_881_425,
        4.421_344_726_335_863,
    ];
    const SCIPY_EXPECTED: [f64; 15] = [
        0.361_058_113_054_895,
        -1.589_311_601_968_926_4,
        -1.699_509_593_313_216_3,
        0.145_307_036_241_871_8,
        2.039_425_893_317_693,
        2.258_334_183_586_908_4,
        1.725_434_898_038_608,
        0.999_301_840_523_396_7,
        -0.137_614_628_080_667_88,
        -0.702_806_444_174_261,
        0.603_317_918_332_819_2,
        0.704_440_202_667_001,
        -0.108_040_913_634_959_27,
        -0.453_911_592_656_411_73,
        -0.060_689_518_737_027_95,
    ];

    const SCIPY_OUT_QUERIES: [f64; 6] = [
        -0.618_131_182_026_264_3,
        0.281_868_817_973_735_75,
        0.381_868_817_973_735_73,
        4.421_344_726_335_863,
        4.521_344_726_335_863,
        5.421_344_726_335_863,
    ];

    const SCIPY_OUT_EXPECTED: [f64; 6] = [
        6.313_692_341_802_915,
        1.269_431_844_116_979,
        0.361_058_113_054_895,
        -0.060_689_518_737_027_95,
        0.191_592_192_907_248_1,
        4.370_634_652_056_136,
    ];

    #[test]
    fn matches_scipy_fixture_to_1e12() {
        let p = Pchip::new(SCIPY_X.to_vec(), SCIPY_Y.to_vec()).expect("fixture input is valid");
        let max_diff = SCIPY_QUERIES
            .iter()
            .zip(SCIPY_EXPECTED)
            .map(|(&xq, expected)| (p.eval(xq) - expected).abs())
            .fold(0.0, f64::max);

        assert!(
            max_diff < 1e-12,
            "max abs diff {} exceeded tolerance",
            max_diff
        );
    }

    #[test]
    fn matches_scipy_extrapolated_fixture_to_1e12() {
        let p = Pchip::new(SCIPY_X.to_vec(), SCIPY_Y.to_vec()).expect("fixture input is valid");
        let max_diff = SCIPY_OUT_QUERIES
            .iter()
            .zip(SCIPY_OUT_EXPECTED)
            .map(|(&xq, expected)| (p.eval(xq) - expected).abs())
            .fold(0.0, f64::max);

        assert!(
            max_diff < 1e-12,
            "max abs diff {} exceeded tolerance",
            max_diff
        );
    }
}
