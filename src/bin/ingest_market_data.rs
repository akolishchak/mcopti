use chrono::{Duration, Months, NaiveDate, Utc};
use mcopti::MarketData;
use std::env;
use std::error::Error;
use std::io::{Error as IoError, ErrorKind};
use std::time::SystemTime;
use std::fs::read_to_string;

const DB_PATH: &str = "data/market_data_1d.db";

fn parse_date(value: &str) -> Result<NaiveDate, IoError> {
    NaiveDate::parse_from_str(value, "%Y-%m-%d").map_err(|e| {
        IoError::new(
            ErrorKind::InvalidInput,
            format!("invalid date '{value}' (expected YYYY-MM-DD): {e}"),
        )
    })
}

fn today_utc_date() -> NaiveDate {
    chrono::DateTime::<Utc>::from(SystemTime::now()).date_naive()
}

fn default_last_12_months() -> (NaiveDate, NaiveDate) {
    let end = today_utc_date();
    let start = end
        .checked_sub_months(Months::new(12))
        .unwrap_or(end - Duration::days(365));
    (start, end)
}

fn tickers(identifier: &str) -> Result<Vec<String>, IoError> {
    if identifier.len() <= 5 {
        Ok(vec![identifier.to_string()])
    } else {
        // long ticker - must be file name
        let tickers = read_to_string(identifier)?
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        Ok(tickers)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args.len() > 4 {
        eprintln!("Usage: ingest_market_data <ticker|file_name> [start YYYY-MM-DD] [end YYYY-MM-DD]");
        std::process::exit(2);
    }

    let identifier = &args[1];
    let (start, end) = match args.len() {
        2 => default_last_12_months(),
        3 => {
            let start = parse_date(&args[2])?;
            (start, today_utc_date())
        }
        4 => (parse_date(&args[2])?, parse_date(&args[3])?),
        _ => unreachable!(),
    };

    if end < start {
        return Err(Box::new(IoError::new(
            ErrorKind::InvalidInput,
            "end date must be on or after start date",
        )));
    }

    let mut market_data = MarketData::default_write("1d")?;

    let tickers = tickers(identifier)?;
    let tickers_num = tickers.len();
    for ticker in &tickers {
        market_data.ingest(ticker, "1d", start, end)?;
        println!(
            "Updated ticker {} ({}..{})",
            ticker,
            start,
            end
        );
    }
    
    if tickers_num > 0 {
        println!("{tickers_num} ticker(s) updated in {DB_PATH}");
    }
    
    Ok(())
}
