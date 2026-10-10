//! Simulation configuration defaults and tuning parameters.

/// Simulation settings. Start with [`DEFAULT_CONFIG`] and override individual fields.
pub struct Config {
    /// Number of Monte Carlo paths; must be positive. Default: 100,000.
    pub paths: usize,
    /// Positive intraday time step, in minutes. Default: 15.
    pub step_minutes: i64,
    /// Bounds on the realized-to-implied volatility ratio. Default: `(0.5, 3.0)`.
    pub factor_clamp: (f64, f64),
    /// Bounds on each step's implied volatility multiplier. Default: `(0.8, 2.0)`.
    pub iv_level_clamp: (f64, f64),
    /// Numerical tolerance setting. Default: `1e-12`.
    pub epsilon: f64,
    /// Spot multiplier for the strike used to floor implied volatility. Default: `0.90`.
    pub iv_floor: f64,
    /// Seed for reproducible Monte Carlo paths. Default: 7.
    pub seed: u64,
    /// Default minimum entry barrier ratio for backtests. Default: `0.75`.
    pub entry_barrier_ratio_threshold: f64,
    /// Default minimum expected return divided by simulated risk. Default: `0.03`.
    pub ror_threshold: f64,
}

/// Default simulation and screening settings.
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
            entry_barrier_ratio_threshold: 0.75,
            ror_threshold: 0.03,
        }
    }
}
