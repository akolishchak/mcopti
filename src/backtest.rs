use crate::{
    Context, EntryBarriers, LegUniverse, OpenPosition, OptionChainDb, OptionChainDbError,
    OptionsDbMode, Position, RawOptionChain, Scenario, Simulator,
};

use chrono::NaiveDate;
use rayon::prelude::*;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::rc::Rc;

#[derive(Clone, Copy, Debug)]
pub struct BacktestParameters {
    pub entry_barrier_ratio_threshold: f64,
    pub ror_threshold: f64,
}

#[derive(Clone, Debug)]
pub struct BacktestSummary {
    pub parameters: BacktestParameters,
    pub closed_pnl: f64,
    pub max_drawdown: f64,
    pub open_positions: usize,
    pub wins: usize,
    pub losses: usize,
}

#[derive(Clone)]
struct PreparedCandidate {
    ticker: Rc<str>,
    position: Position,
    expected_value: f64,
    risk: f64,
    ror: f64,
    barrier_ratio: f64,
}

struct Track {
    parameters: BacktestParameters,
    max_positions: usize,
    profit_take: f64,
    stop_loss: f64,
    open_positions: Vec<OpenPosition>,
    pnl: f64,
    pnl_peak: f64,
    max_drawdown: f64,
    wins: usize,
    losses: usize,
}

impl Track {
    fn new(
        parameters: BacktestParameters,
        max_positions: usize,
        profit_take: f64,
        stop_loss: f64,
    ) -> Self {
        Self {
            parameters,
            max_positions,
            profit_take,
            stop_loss,
            open_positions: Vec::new(),
            pnl: 0.0,
            pnl_peak: 0.0,
            max_drawdown: 0.0,
            wins: 0,
            losses: 0,
        }
    }

    fn update_open_positions(
        &mut self,
        chain: &OptionChainDb,
        date: NaiveDate,
        verbose: bool,
    ) -> Result<(), BacktestError> {
        let mut i = 0;
        while i < self.open_positions.len() {
            let should_remove = {
                let pos = &mut self.open_positions[i];
                let price = chain.position_mark_mid(&pos.ticker, &pos.position)?;
                if let Some(metrics) = pos.update(date, price) {
                    self.pnl += metrics.pnl;
                    self.pnl_peak = self.pnl_peak.max(self.pnl);
                    self.max_drawdown = self.max_drawdown.max(self.pnl_peak - self.pnl);
                    if verbose {
                        println!(
                            "[CLOSE] {date} {} {{{}}} mark={:.4} pnl={:.4} pf={:.4} dd={:.4} cum_pnl={:.4}",
                            pos.ticker,
                            pos.position,
                            price,
                            metrics.pnl,
                            metrics.profit_factor,
                            metrics.drawdown,
                            self.pnl
                        );
                    }
                    if metrics.pnl.is_sign_positive() {
                        self.wins += 1;
                    } else {
                        self.losses += 1;
                    }
                    true
                } else {
                    false
                }
            };

            if should_remove {
                self.open_positions.swap_remove(i);
            } else {
                i += 1;
            }
        }

        Ok(())
    }

    fn open_positions(&mut self, candidates: &[PreparedCandidate], date: NaiveDate, verbose: bool) {
        let mut opened = 0;
        for candidate in candidates {
            if opened == self.max_positions {
                break;
            }
            if candidate.ror < self.parameters.ror_threshold {
                continue;
            }
            if candidate.barrier_ratio < self.parameters.entry_barrier_ratio_threshold {
                continue;
            }

            let side = candidate.position.side();
            let pt_mark = candidate.position.premium * (1.0 + self.profit_take.max(0.0) * side);
            let sl_mark = candidate.position.premium * (1.0 - self.stop_loss.max(0.0) * side);
            if verbose {
                println!(
                    "[OPEN ] {date} {} {{{}}} premium={:.4} ev={:.4} risk={:.4} ror={:.4} pt={:.4} sl={:.4}",
                    candidate.ticker,
                    candidate.position,
                    candidate.position.premium,
                    candidate.expected_value,
                    candidate.risk,
                    candidate.ror,
                    pt_mark,
                    sl_mark
                );
            }
            self.open_positions.push(OpenPosition::new(
                candidate.ticker.to_string(),
                candidate.position.clone(),
                pt_mark,
                sl_mark,
            ));
            opened += 1;
        }
    }

    fn summary(self) -> BacktestSummary {
        BacktestSummary {
            parameters: self.parameters,
            closed_pnl: self.pnl,
            max_drawdown: self.max_drawdown,
            open_positions: self.open_positions.len(),
            wins: self.wins,
            losses: self.losses,
        }
    }
}

impl BacktestSummary {
    pub fn score(&self) -> f64 {
        self.closed_pnl - self.max_drawdown
    }
}

pub struct Backtest {
    // premium coefficients to enter/exit a trade
    profit_take: f64,
    stop_loss: f64,
    // data days
    days: Vec<PathBuf>,
    //
    max_positions: usize,
}

pub struct ScreenerCandidate {
    pub ticker: String,
    pub raw_chain: RawOptionChain,
    pub positions: Vec<Position>,
}

impl ScreenerCandidate {
    pub fn new(ticker: String, raw_chain: RawOptionChain, positions: Vec<Position>) -> Self {
        Self {
            ticker,
            raw_chain,
            positions,
        }
    }
}

pub trait ChainScreener {
    fn screen(&self, option_chains_path: &Path) -> Vec<ScreenerCandidate>;
}

#[derive(Debug)]
pub enum BacktestError {
    Io(io::Error),
    Db(OptionChainDbError),
    NoData,
}

impl From<io::Error> for BacktestError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<crate::option_chain_db::OptionChainDbError> for BacktestError {
    fn from(value: crate::option_chain_db::OptionChainDbError) -> Self {
        Self::Db(value)
    }
}

impl Display for BacktestError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "file error: {e}"),
            Self::Db(e) => write!(f, "db error: {e}"),
            Self::NoData => write!(f, "no input data found"),
        }
    }
}

impl Error for BacktestError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Db(e) => Some(e),
            Self::NoData => None,
        }
    }
}

impl Backtest {
    pub fn new(
        data_paths: &[PathBuf],
        profit_take: f64,
        stop_loss: f64,
    ) -> Result<Self, BacktestError> {
        let mut days = Vec::new();
        for data_path in data_paths {
            days.extend(
                fs::read_dir(data_path)?
                    .flatten()
                    .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
                    .map(|e| e.path()),
            );
        }

        if days.is_empty() {
            return Err(BacktestError::NoData);
        }

        days.sort_unstable_by(|a, b| a.file_name().cmp(&b.file_name()));

        Ok(Self {
            profit_take,
            stop_loss,
            days,
            max_positions: 2,
        })
    }

    pub fn run(
        &self,
        generator: impl ChainScreener + Sync,
        parameters: BacktestParameters,
    ) -> Result<(), BacktestError> {
        let parameters = [parameters];
        self.run_tracks(&generator, &parameters, true)?;
        Ok(())
    }

    pub fn run_tracks(
        &self,
        generator: &(impl ChainScreener + Sync),
        parameters: &[BacktestParameters],
        verbose: bool,
    ) -> Result<Vec<BacktestSummary>, BacktestError> {
        if parameters.is_empty() {
            return Ok(Vec::new());
        }

        let mut tracks: Vec<_> = parameters
            .iter()
            .copied()
            .map(|parameters| {
                Track::new(
                    parameters,
                    self.max_positions,
                    self.profit_take,
                    self.stop_loss,
                )
            })
            .collect();
        let jobs: Vec<_> = self
            .days
            .par_iter()
            .map(|day| {
                generator
                    .screen(day)
                    .into_iter()
                    .map(|candidate| {
                        let ticker = candidate.ticker;
                        let context = Context::from_raw_option_chain(&ticker, &candidate.raw_chain);
                        let universe = LegUniverse::from_positions(candidate.positions);
                        (ticker, context, universe)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        for (day, job) in self.days.iter().zip(jobs) {
            let date = NaiveDate::parse_from_str(
                day.file_name().and_then(|n| n.to_str()).unwrap(),
                "%Y-%m-%d",
            )
            .unwrap();
            let chain = OptionChainDb::new(day, OptionsDbMode::Read)?;
            let candidates = self.prepare_candidates(job);

            for track in &mut tracks {
                track.update_open_positions(&chain, date, verbose)?;
                track.open_positions(&candidates, date, verbose);
            }
        }

        let mut summaries: Vec<_> = tracks.into_iter().map(Track::summary).collect();
        summaries.sort_unstable_by(|a, b| {
            b.score()
                .total_cmp(&a.score())
                .then_with(|| b.closed_pnl.total_cmp(&a.closed_pnl))
                .then_with(|| a.max_drawdown.total_cmp(&b.max_drawdown))
                .then_with(|| {
                    let a_trades = a.wins + a.losses;
                    let b_trades = b.wins + b.losses;
                    let a_win_rate = if a_trades == 0 {
                        0.0
                    } else {
                        a.wins as f64 / a_trades as f64
                    };
                    let b_win_rate = if b_trades == 0 {
                        0.0
                    } else {
                        b.wins as f64 / b_trades as f64
                    };
                    b_win_rate.total_cmp(&a_win_rate)
                })
        });
        if verbose {
            for summary in &summaries {
                println!(
                    "[SUMMARY] barrier_threshold={:.2} ror_threshold={:.2} closed_pnl={:.4} max_drawdown={:.4} open_positions={}, wins={}, losses={}",
                    summary.parameters.entry_barrier_ratio_threshold,
                    summary.parameters.ror_threshold,
                    summary.closed_pnl,
                    summary.max_drawdown,
                    summary.open_positions,
                    summary.wins,
                    summary.losses
                );
            }
        }

        Ok(summaries)
    }

    fn prepare_candidates(
        &self,
        job: Vec<(String, Context, LegUniverse)>,
    ) -> Vec<PreparedCandidate> {
        let mut candidates = Vec::with_capacity(job.len() * 5);
        for (ticker, context, universe) in job {
            let ticker: Rc<str> = Rc::from(ticker);
            let Ok(scenario) = Scenario::new(&context, &universe)
                .inspect_err(|err| eprintln!("{}: senario error ={}", ticker, err))
            else {
                continue;
            };

            let sim = Simulator::default();
            let Ok(stats) = sim
                .run(&context, universe, &scenario)
                .inspect_err(|err| eprintln!("{}: simulator error = {}", ticker, err))
            else {
                continue;
            };

            for stat in stats {
                let barrier_ratio = EntryBarriers::new(
                    &context,
                    &stat.position,
                    &scenario,
                    self.profit_take,
                    self.stop_loss,
                )
                .ratio();
                candidates.push(PreparedCandidate {
                    ticker: Rc::clone(&ticker),
                    position: stat.position,
                    expected_value: stat.expected_value,
                    risk: stat.risk,
                    ror: stat.ror,
                    barrier_ratio,
                });
            }
        }

        candidates.sort_unstable_by(|a, b| b.ror.total_cmp(&a.ror));
        candidates
    }
}
