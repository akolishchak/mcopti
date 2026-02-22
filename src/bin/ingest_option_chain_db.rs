use mcopti::option_chain_db::OptionChainDb;
use std::env;
use std::error::Error;
use std::io::{Error as IoError, ErrorKind};
use std::path::Path;

const DB_FILE: &str = "options.db";

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: ingest_option_chain_db <folder_path>");
        std::process::exit(2);
    }

    let folder = &args[1];
    let folder_path = Path::new(folder);
    if !folder_path.exists() {
        return Err(Box::new(IoError::new(
            ErrorKind::NotFound,
            format!("folder does not exist: {folder}"),
        )));
    }
    if !folder_path.is_dir() {
        return Err(Box::new(IoError::new(
            ErrorKind::InvalidInput,
            format!("path is not a folder: {folder}"),
        )));
    }

    let mut db = OptionChainDb::default_write(folder)?;
    let ingested = db.ingest_from_json()?;

    println!(
        "Ingested {ingested} option chain file(s) into {}/{}",
        folder_path.display(),
        DB_FILE
    );

    Ok(())
}
