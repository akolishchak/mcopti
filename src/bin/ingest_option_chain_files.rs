use mcopti::option_chain_db::OptionChainDb;
use mcopti::raw_option_chain::parse_option_chain_file;
use std::error::Error;
use std::io::{Error as IoError, ErrorKind};
use std::path::{Path, PathBuf};

fn ticker_from_file_name(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?.trim();
    let ticker = stem.split('_').next().unwrap_or(stem).trim();
    if ticker.is_empty() {
        None
    } else {
        Some(ticker.to_ascii_uppercase())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let files: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if files.is_empty() {
        eprintln!("Usage: ingest_option_chain_files <json_file>...");
        std::process::exit(2);
    }

    let mut current_dir: Option<PathBuf> = None;
    let mut db: Option<OptionChainDb> = None;
    let mut ingested = 0usize;

    for file in files {
        if !file.exists() {
            return Err(Box::new(IoError::new(
                ErrorKind::NotFound,
                format!("file does not exist: {}", file.display()),
            )));
        }
        if !file.is_file() {
            return Err(Box::new(IoError::new(
                ErrorKind::InvalidInput,
                format!("path is not a file: {}", file.display()),
            )));
        }

        let parent = file.parent().ok_or_else(|| {
            IoError::new(
                ErrorKind::InvalidInput,
                format!("file has no parent directory: {}", file.display()),
            )
        })?;
        if current_dir.as_deref() != Some(parent) {
            db = Some(OptionChainDb::default_write(
                parent.to_string_lossy().as_ref(),
            )?);
            current_dir = Some(parent.to_path_buf());
        }

        let ticker = ticker_from_file_name(&file).ok_or_else(|| {
            IoError::new(
                ErrorKind::InvalidInput,
                format!("cannot infer ticker from file name: {}", file.display()),
            )
        })?;
        let payload = parse_option_chain_file(&file)?;
        if payload.data.is_empty() {
            return Err(Box::new(IoError::new(
                ErrorKind::InvalidData,
                format!("empty option chain: {}", file.display()),
            )));
        }

        db.as_mut()
            .expect("db should be initialized")
            .ingest(&ticker, &payload)?;
        ingested += 1;
        println!(
            "Ingested {ticker} {} contract(s) into {}/options.db",
            payload.data.len(),
            parent.display()
        );
    }

    println!("Ingested {ingested} option chain file(s)");
    Ok(())
}
