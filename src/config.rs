//! Simulation configuration defaults and tuning parameters.

pub struct Config {
    pub paths: usize,
    pub step_minutes: i64,
    pub factor_clamp: (f64, f64),
    pub iv_level_clamp: (f64, f64),
    pub epsilon: f64,
    pub iv_floor: f64,
    pub seed: u64,
}

pub const DEFAULT_CONFIG: Config = Config::default();

impl Config {
    const fn default() -> Self {
        Self {
            paths: 100_000,
            step_minutes: 15,
            factor_clamp: (0.5, 3.0),
            iv_level_clamp: (0.8, 2.0),
            epsilon: 1e-12,
            iv_floor: 0.90,
            seed: 7,
        }
    }
}
