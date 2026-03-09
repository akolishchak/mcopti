use chrono::{NaiveDate, Utc};
use mcopti::{Backtest, MarketData, OptionChainDb, spread_screener::SpreadScreener};
use std::env;
use std::error::Error;
use std::io::{Error as IoError, ErrorKind};
use std::path::Path;
use std::time::SystemTime;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args.len() > 4 {
        eprintln!("Usage: backtest_spreads <data_dir> [profit_take] [stop_loss]");
        std::process::exit(2);
    }

    let data_dir = Path::new(&args[1]);
    if !data_dir.exists() {
        return Err(Box::new(IoError::new(
            ErrorKind::NotFound,
            format!("data directory does not exist: {}", data_dir.display()),
        )));
    }
    if !data_dir.is_dir() {
        return Err(Box::new(IoError::new(
            ErrorKind::InvalidInput,
            format!("path is not a directory: {}", data_dir.display()),
        )));
    }

    fix_market_data_db(data_dir)?;
    fix_options_db(data_dir)?;

    let profit_take = if let Some(raw) = args.get(2) {
        raw.parse::<f64>()?
    } else {
        0.50
    };
    let stop_loss = if let Some(raw) = args.get(3) {
        raw.parse::<f64>()?
    } else {
        0.50
    };

    let backtest = Backtest::new(data_dir, profit_take, stop_loss)?;
    let screener = SpreadScreener::default();
    backtest.run(screener)?;

    Ok(())
}

fn fix_options_db(data_dir: &Path) -> Result<(), Box<dyn Error>> {
    for entry in std::fs::read_dir(data_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }

        let day_dir = entry.path();
        let db_path = day_dir.join("options.db");
        if db_path.exists() {
            continue;
        }

        let mut db = OptionChainDb::default_write(day_dir.to_string_lossy().as_ref())?;
        db.ingest_from_json()?;
    }

    Ok(())
}

fn fix_market_data_db(data_dir: &Path) -> Result<(), Box<dyn Error>> {
    if Path::new("data/market_data_1d.db").exists() {
        return Ok(());
    }

    let mut tickers = Vec::new();
    let day_dir = std::fs::read_dir(data_dir)?
        .flatten()
        .find(|entry| entry.file_type().is_ok_and(|ft| ft.is_dir()))
        .map(|entry| entry.path());

    if let Some(day_dir) = day_dir {
        for file in std::fs::read_dir(day_dir)? {
            let file = file?;
            let path = file.path();
            if !path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
            {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Some(ticker) = stem.split('_').next() else {
                continue;
            };
            if !ticker.trim().is_empty() {
                tickers.push(ticker.trim().to_ascii_uppercase());
            }
        }
    }

    std::fs::create_dir_all("data")?;
    let mut write_db = MarketData::default_write("1d")?;
    let start = NaiveDate::from_ymd_opt(2020, 1, 1).expect("valid start date");
    let end = chrono::DateTime::<Utc>::from(SystemTime::now()).date_naive();
    println!("Fetching market data...");
    for ticker in tickers {
        println!("{ticker}");
        write_db.ingest(&ticker, "1d", start, end)?;
    }

    Ok(())
}
