use crate::{
    Context, LegUniverse, OpenPosition, OptionChainDb, OptionChainDbError, OptionsDbMode, Position,
    RawOptionChain, Scenario, Simulator,
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
        data_path: impl AsRef<Path>,
        profit_take: f64,
        stop_loss: f64,
    ) -> Result<Self, BacktestError> {
        let mut days: Vec<_> = fs::read_dir(data_path)?
            .flatten()
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .map(|e| e.path())
            .collect();

        if days.is_empty() {
            return Err(BacktestError::NoData);
        }

        days.sort_unstable();

        Ok(Self {
            profit_take,
            stop_loss,
            days,
            max_positions: 2,
        })
    }

    pub fn run(&self, generator: impl ChainScreener + Sync) -> Result<(), BacktestError> {
        let mut open_positions: Vec<OpenPosition> = Vec::new();
        let mut pnl = 0.0;

        let simulator_jobs: Vec<_> = self
            .days
            .par_iter()
            .map(|day| {
                let candidates = generator.screen(day);
                candidates
                    .into_iter()
                    .map(|candidate| {
                        let ticker = candidate.ticker;
                        let context =
                            Context::from_raw_option_chain(&ticker, &candidate.raw_chain);
                        let universe = LegUniverse::from_positions(candidate.positions);
                        (ticker, context, universe)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        for (day, job) in self.days.iter().zip(simulator_jobs) {
            //
            let date = NaiveDate::parse_from_str(
                day.file_name().and_then(|n| n.to_str()).unwrap(),
                "%Y-%m-%d",
            )
            .unwrap();

            // println!("{date}");
            //
            let chain = OptionChainDb::new(day, OptionsDbMode::Read)?;
            //
            // evaluate open positions
            //
            let mut i = 0;
            while i < open_positions.len() {
                let should_remove = {
                    let pos = &mut open_positions[i];
                    // mark position
                    let price = chain.position_mark_mid(&pos.ticker, &pos.position)?;
                    if let Some(metrics) = pos.update(date, price) {
                        // exit, update performance counters
                        pnl += metrics.pnl;
                        println!(
                            "[CLOSE] {date} {} mark={:.4} pnl={:.4} pf={:.4} dd={:.4} cum_pnl={:.4}",
                            pos.ticker,
                            price,
                            metrics.pnl,
                            metrics.profit_factor,
                            metrics.drawdown,
                            pnl
                        );
                        true // remove position
                    } else {
                        false // keep position
                    }
                };

                if should_remove {
                    open_positions.swap_remove(i);
                } else {
                    i += 1;
                }
            }

            //
            // check for new positions
            //
            let mut candidates = Vec::with_capacity(job.len() * 5);
            for (ticker, context, universe) in job {
                // println!(
                //     "{}: {} candidates, simulating...",
                //     gen_pos.ticker,
                //     gen_pos.positions.len()
                // );
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
                // println!("{:?}", stats);
                candidates.extend(stats.into_iter().map(|stat| (Rc::clone(&ticker), stat)));
            }
            //
            // rank stats by top RoR
            // open max_positions from the list
            //
            let max_positions = self.max_positions.min(candidates.len());
            if max_positions == 0 {
                continue;
            }
            candidates.select_nth_unstable_by(max_positions - 1, |a, b| {
                let ror_a = a.1.expected_value / a.1.risk;
                let ror_b = b.1.expected_value / b.1.risk;
                ror_b.total_cmp(&ror_a)
            });

            for (ticker, stat) in candidates.into_iter().take(max_positions) {
                let side = stat.position.side();
                let pt_mark = stat.position.premium * (1.0 + self.profit_take.max(0.0) * side);
                let sl_mark = stat.position.premium * (1.0 - self.stop_loss.max(0.0) * side);
                let ror = if stat.risk != 0.0 {
                    stat.expected_value / stat.risk
                } else {
                    f64::NAN
                };
                println!(
                    "[OPEN ] {date} {ticker} legs={} premium={:.4} ev={:.4} risk={:.4} ror={:.4} pt={:.4} sl={:.4}",
                    stat.position.legs.len(),
                    stat.position.premium,
                    stat.expected_value,
                    stat.risk,
                    ror,
                    pt_mark,
                    sl_mark
                );
                let position =
                    OpenPosition::new(ticker.to_string(), stat.position, pt_mark, sl_mark);
                open_positions.push(position);
            }
        }

        println!(
            "[SUMMARY] closed_pnl={:.4} open_positions={}",
            pnl,
            open_positions.len()
        );

        Ok(())
    }
}
