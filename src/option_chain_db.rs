use crate::raw_option_chain::{OptionChainError, parse_option_chain_file};
use crate::{OptionType, RawOptionChain};
use rusqlite::{Connection, OpenFlags, Transaction, params};
use std::collections::HashMap;
use std::error::Error as StdError;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

const OPTION_DB_FILE: &str = "options.db";

const OPTIONS_DB_SCHEMA: &str = r#"
    PRAGMA foreign_keys = ON;
    CREATE TABLE IF NOT EXISTS symbols(
        id  INTEGER PRIMARY KEY,
        sym TEXT UNIQUE NOT NULL
    );
    CREATE TABLE IF NOT EXISTS snapshots(           -- one trade-date per symbol
        id  INTEGER PRIMARY KEY,
        symbol_id INTEGER NOT NULL,
        snap_date TEXT NOT NULL,
        last_price REAL NOT NULL,

        UNIQUE(symbol_id, snap_date)
    );
    CREATE TABLE IF NOT EXISTS expirations(         -- one row per expiry inside snapshot
        id  INTEGER PRIMARY KEY,
        snapshot_id INTEGER NOT NULL,
        exp_date TEXT NOT NULL,

        FOREIGN KEY (snapshot_id) REFERENCES snapshots(id),
        UNIQUE(snapshot_id, exp_date)
    );
    CREATE TABLE IF NOT EXISTS contracts(
        id            INTEGER PRIMARY KEY,
        snapshot_id   INTEGER NOT NULL,
        expiration_id INTEGER NOT NULL,
        opt_type  TEXT NOT NULL CHECK(opt_type IN('C','P')),
        strike  REAL NOT NULL,
        last    REAL,   mark REAL,
        bid     REAL,   bid_size  INT,
        ask     REAL,   ask_size  INT,
        volume  INT,    open_interest INT,
        iv REAL, delta REAL, gamma REAL, theta REAL, vega REAL, rho REAL,

        FOREIGN KEY (snapshot_id) REFERENCES snapshots(id),
        FOREIGN KEY (expiration_id) REFERENCES expirations(id),
        UNIQUE(snapshot_id, expiration_id, opt_type, strike)         -- 1 row per leg/strike
    );
    CREATE INDEX IF NOT EXISTS idx_puts
        ON contracts(snapshot_id, expiration_id, opt_type, strike)
        WHERE opt_type='P';
    CREATE INDEX IF NOT EXISTS idx_calls
        ON contracts(snapshot_id, expiration_id, opt_type, strike)
        WHERE opt_type='C';
    CREATE INDEX IF NOT EXISTS idx_contracts_delta
        ON contracts(snapshot_id, expiration_id, delta);
    CREATE INDEX IF NOT EXISTS idx_expirations_snapshot
        ON expirations(snapshot_id);
    CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_date
        ON snapshots(symbol_id, snap_date);
"#;

pub struct OptionChainDb {
    connection: Connection,
    data_path: PathBuf,
}

pub enum OptionsDbMode {
    Read,
    Write,
    Memory,
}

#[derive(Debug)]
pub enum OptionChainDbError {
    Db(rusqlite::Error),
    Io(std::io::Error),
    Parse(OptionChainError),
    InvalidPayload(String),
}

pub type OptionChainDbResult<T> = std::result::Result<T, OptionChainDbError>;

impl Display for OptionChainDbError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Db(err) => write!(f, "database error: {err}"),
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::Parse(err) => write!(f, "parse error: {err}"),
            Self::InvalidPayload(msg) => write!(f, "invalid payload: {msg}"),
        }
    }
}

impl StdError for OptionChainDbError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Db(err) => Some(err),
            Self::Io(err) => Some(err),
            Self::Parse(err) => Some(err),
            Self::InvalidPayload(_) => None,
        }
    }
}

impl From<rusqlite::Error> for OptionChainDbError {
    fn from(value: rusqlite::Error) -> Self {
        Self::Db(value)
    }
}

impl From<std::io::Error> for OptionChainDbError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<OptionChainError> for OptionChainDbError {
    fn from(value: OptionChainError) -> Self {
        Self::Parse(value)
    }
}

impl OptionChainDb {
    pub fn new(data_path: &str, dbmode: OptionsDbMode) -> OptionChainDbResult<Self> {
        let path = format!("{data_path}/{OPTION_DB_FILE}");

        let connection = match dbmode {
            OptionsDbMode::Read => {
                let uri = format!("file:{path}?mode=ro&immutable=1");

                let flags = OpenFlags::SQLITE_OPEN_READ_ONLY
                    | OpenFlags::SQLITE_OPEN_URI
                    | OpenFlags::SQLITE_OPEN_NO_MUTEX;

                let conn = Connection::open_with_flags(uri, flags)?;

                conn.execute_batch(
                    r#"
                    PRAGMA query_only=ON;
                    PRAGMA mmap_size=134217728;     -- 128MB
                    PRAGMA cache_size=-98304;       -- ~96MB
                    PRAGMA temp_store=MEMORY;
                "#,
                )?;
                conn
            }
            OptionsDbMode::Write => {
                let conn = Connection::open(path)?;
                conn.execute_batch(
                    r#"
                    PRAGMA journal_mode=WAL;
                    PRAGMA synchronous=NORMAL;
                    PRAGMA wal_autocheckpoint=0;
                    PRAGMA temp_store=MEMORY;
                    PRAGMA cache_size=-131072;      -- ~128MB
                "#,
                )?;
                conn.execute_batch(OPTIONS_DB_SCHEMA)?;
                conn
            }
            OptionsDbMode::Memory => {
                let conn = Connection::open_in_memory()?;
                conn.execute_batch(
                    r#"
                    PRAGMA temp_store=MEMORY;
                    PRAGMA cache_size=-131072;      -- ~128MB
                "#,
                )?;
                conn.execute_batch(OPTIONS_DB_SCHEMA)?;
                conn
            }
        };

        Ok(Self {
            connection,
            data_path: PathBuf::from(data_path),
        })
    }

    pub fn default_read(data_path: &str) -> OptionChainDbResult<Self> {
        Self::new(data_path, OptionsDbMode::Read)
    }

    pub fn default_write(data_path: &str) -> OptionChainDbResult<Self> {
        Self::new(data_path, OptionsDbMode::Write)
    }

    pub fn default_memory(data_path: &str) -> OptionChainDbResult<Self> {
        Self::new(data_path, OptionsDbMode::Memory)
    }

    pub fn ingest_from_json(&mut self) -> OptionChainDbResult<usize> {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(&self.data_path)? {
            let path = entry?.path();
            if path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
            {
                files.push(path);
            }
        }
        files.sort();

        let mut ingested = 0;
        for file in &files {
            let payload = parse_option_chain_file(file)?;
            let ticker = Self::ticker_from_file_name(file).ok_or_else(|| {
                OptionChainDbError::InvalidPayload(format!(
                    "cannot infer ticker from file name '{}'",
                    file.display()
                ))
            })?;
            self.ingest(&ticker, &payload)?;
            ingested += 1;
        }
        Ok(ingested)
    }

    pub fn ingest(&mut self, ticker: &str, payload: &RawOptionChain) -> OptionChainDbResult<()> {
        let symbol = ticker.trim().to_ascii_uppercase();
        if symbol.is_empty() {
            return Err(OptionChainDbError::InvalidPayload(
                "missing ticker".to_string(),
            ));
        }

        let trade_date = payload.date.format("%Y-%m-%d").to_string();
        let tx = self.connection.transaction()?;
        let symbol_id = Self::upsert_symbol(&tx, &symbol)?;
        let snapshot_id = Self::upsert_snapshot(&tx, symbol_id, &trade_date, payload.last_price)?;

        let mut expirations: HashMap<chrono::NaiveDate, i64> = HashMap::new();
        for leg in &payload.data {
            if let std::collections::hash_map::Entry::Vacant(entry) =
                expirations.entry(leg.expiration)
            {
                let exp_date = leg.expiration.format("%Y-%m-%d").to_string();
                let expiration_id = Self::upsert_expiration(&tx, snapshot_id, &exp_date)?;
                entry.insert(expiration_id);
            }
        }

        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO contracts
                (snapshot_id,expiration_id,opt_type,strike,last,mark,bid,bid_size,
                 ask,ask_size,volume,open_interest,iv,delta,gamma,theta,vega,rho)
                VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18)",
            )?;

            for leg in &payload.data {
                let expiration_id = *expirations.get(&leg.expiration).ok_or_else(|| {
                    OptionChainDbError::InvalidPayload(
                        "expiration id not found for option leg".to_string(),
                    )
                })?;
                let opt_type = Self::option_type_code(leg.option_type);

                stmt.execute(params![
                    snapshot_id,
                    expiration_id,
                    opt_type,
                    leg.strike,
                    leg.last,
                    leg.mark,
                    leg.bid,
                    leg.bid_size as i64,
                    leg.ask,
                    leg.ask_size as i64,
                    leg.volume as i64,
                    leg.open_interest as i64,
                    leg.implied_volatility,
                    leg.delta,
                    leg.gamma,
                    leg.theta,
                    leg.vega,
                    leg.rho,
                ])?;
            }
        }

        tx.commit()?;
        Ok(())
    }

    fn option_type_code(option_type: OptionType) -> &'static str {
        match option_type {
            OptionType::Call => "C",
            OptionType::Put => "P",
        }
    }

    fn ticker_from_file_name(path: &Path) -> Option<String> {
        let stem = path.file_stem()?.to_str()?.trim();
        let ticker = stem.split('_').next().unwrap_or(stem).trim();
        if ticker.is_empty() {
            None
        } else {
            Some(ticker.to_ascii_uppercase())
        }
    }

    fn upsert_symbol(tx: &Transaction<'_>, symbol: &str) -> OptionChainDbResult<i64> {
        let id = tx.query_row(
            "INSERT INTO symbols(sym) VALUES (?1)
             ON CONFLICT(sym) DO UPDATE SET id=id
             RETURNING id",
            params![symbol],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    fn upsert_snapshot(
        tx: &Transaction<'_>,
        symbol_id: i64,
        snap_date: &str,
        last_price: f64,
    ) -> OptionChainDbResult<i64> {
        let id = tx.query_row(
            "INSERT INTO snapshots(symbol_id, snap_date, last_price) VALUES (?1, ?2, ?3)
             ON CONFLICT(symbol_id, snap_date) DO UPDATE SET id=id
             RETURNING id",
            params![symbol_id, snap_date, last_price],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    fn upsert_expiration(
        tx: &Transaction<'_>,
        snapshot_id: i64,
        exp_date: &str,
    ) -> OptionChainDbResult<i64> {
        let id = tx.query_row(
            "INSERT INTO expirations(snapshot_id, exp_date) VALUES (?1, ?2)
             ON CONFLICT(snapshot_id, exp_date) DO UPDATE SET id=id
             RETURNING id",
            params![snapshot_id, exp_date],
            |row| row.get(0),
        )?;
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};

    #[test]
    fn ingests_all_json_files_in_tests_fixtures() {
        let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let json_files = option_chain_json_files(&fixture_dir);
        assert!(
            !json_files.is_empty(),
            "expected at least one option-chain json fixture in {}",
            fixture_dir.display()
        );

        let mut expected_symbols = HashSet::new();
        let mut expected_snapshots = HashSet::new();
        let mut expected_expirations = HashSet::new();
        let mut expected_contracts = HashSet::new();

        for file in &json_files {
            let payload = parse_option_chain_file(file).expect("failed to parse fixture json");
            let symbol = OptionChainDb::ticker_from_file_name(file)
                .expect("fixture file should include ticker in name");
            let trade_date = payload.date.format("%Y-%m-%d").to_string();
            expected_symbols.insert(symbol.clone());
            expected_snapshots.insert((symbol.clone(), trade_date.clone()));

            for leg in &payload.data {
                let expiration = leg.expiration.format("%Y-%m-%d").to_string();
                let opt_type = match leg.option_type {
                    OptionType::Call => "C".to_string(),
                    OptionType::Put => "P".to_string(),
                };
                expected_expirations.insert((
                    symbol.clone(),
                    trade_date.clone(),
                    expiration.clone(),
                ));
                expected_contracts.insert((
                    symbol.clone(),
                    trade_date.clone(),
                    expiration,
                    opt_type,
                    leg.strike.to_bits(),
                ));
            }
        }

        let mut db = OptionChainDb::default_memory(
            fixture_dir
                .to_str()
                .expect("fixture directory path should be valid UTF-8"),
        )
        .expect("failed to initialize in-memory options db");

        let ingested = db.ingest_from_json().expect("failed to ingest fixture");
        assert_eq!(ingested, json_files.len());
 
        let symbol_count: usize = db
            .connection
            .query_row("SELECT COUNT(*) FROM symbols", [], |row| row.get(0))
            .unwrap();
        assert_eq!(symbol_count, expected_symbols.len());

        let snapshot_count: usize = db
            .connection
            .query_row("SELECT COUNT(*) FROM snapshots", [], |row| row.get(0))
            .unwrap();
        assert_eq!(snapshot_count, expected_snapshots.len());

        let expirations_count: usize = db
            .connection
            .query_row("SELECT COUNT(*) FROM expirations", [], |row| row.get(0))
            .unwrap();
        assert_eq!(expirations_count, expected_expirations.len());

        let contracts_count: usize = db
            .connection
            .query_row("SELECT COUNT(*) FROM contracts", [], |row| row.get(0))
            .unwrap();
        assert_eq!(contracts_count, expected_contracts.len());
    }

    fn option_chain_json_files(dir: &Path) -> Vec<PathBuf> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .expect("failed to read fixtures directory")
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
            })
            .collect();
        files.sort();
        files
    }
}
