
use chrono::{DateTime, NaiveDate, Utc};
use rusqlite::{Connection, Error, OpenFlags, Result, params};


#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Column {
    Open,
    High,
    Low,
    Close,
    Volume,
    AdjOpen,
    AdjHigh,
    AdjLow,
    AdjClose,
    CoLog,
    OcLog,
    CcLog,
    CoLogAdj,
    OcLogAdj,
    CcLogAdj,
    Vol20,
    Vol20Adj,
    VolTrend20,
    AdjCloseTrend20,
    VolRank6,
    VolRank9,
    VolRank12,
    Iv,
    IvTrend20,
    IvRank6,
    IvRank9,
    IvRank12
}

const COLUMN_COUNT: usize = 27;
const COLUMN_NAMES: [&str; COLUMN_COUNT] = [
    "open", "high", "low", "close", "volume",
    "adjopen", "adjhigh", "adjlow", "adjclose",
    "co_log", "oc_log", "cc_log", "co_log_adj", "oc_log_adj", "cc_log_adj",
    "vol20", "vol20_adj", "vol_trend20", "adjclose_trend20",
    "vol_rank6", "vol_rank9", "vol_rank12",
    "iv", "iv_trend20", "iv_rank6", "iv_rank9", "iv_rank12",
];

const COLUMN_NAMES_ALL: &str = "open, high, low, close, volume,
    adjopen, adjhigh, adjlow, adjclose,
    co_log, oc_log, cc_log, co_log_adj, oc_log_adj,
    cc_log_adj, vol20, vol20_adj, vol_trend20,
    adjclose_trend20, vol_rank6, vol_rank9, vol_rank12,
    iv, iv_trend20, iv_rank6, iv_rank9, iv_rank12";


impl Column {
    #[inline] pub fn idx(self) -> usize { self as usize }
    #[inline] pub fn name(self) -> &'static str { COLUMN_NAMES[self.idx()] }
    #[inline] pub fn all() -> &'static str { COLUMN_NAMES_ALL }
}

pub enum DbMode {
    Read,
    Write,
}
pub struct MarketData {
    connection: Connection,
    columns: Option<String>,
    period_step: i64,
}

impl MarketData {
    pub fn new(directory: &str, resolution: &str, dbmode: DbMode) -> Result<Self> {
        let path = format!("{directory}/market_data_{resolution}.db");

        let connection = match dbmode {
            DbMode::Read => {
                let uri = format!("file:{path}?mode=ro&immutable=1");

                let flags = OpenFlags::SQLITE_OPEN_READ_ONLY
                    | OpenFlags::SQLITE_OPEN_URI
                    | OpenFlags::SQLITE_OPEN_NO_MUTEX;

                let conn = Connection::open_with_flags(uri, flags)?;

                conn.execute_batch(r#"
                    PRAGMA query_only=ON;
                    PRAGMA mmap_size=536870912;     -- 512MB (>= DB size)
                    PRAGMA cache_size=-65536;       -- ~64MB (raise to -131072 if few conns)
                    PRAGMA temp_store=MEMORY;
                "#)?;
                conn
            },
            DbMode::Write => {
                let conn = Connection::open(path)?;
                conn.execute_batch(r#"
                    PRAGMA journal_mode=WAL;
                    PRAGMA synchronous=NORMAL;
                    PRAGMA wal_autocheckpoint=0;
                    PRAGMA temp_store=MEMORY;
                    PRAGMA cache_size=-131072;      -- ~128MB
                "#)?;
                conn
            }
        };

        let period_step = Self::resolution_steps(resolution);

        Ok(Self {
            connection,
            columns: None,
            period_step,
        })
    }

    pub fn default_read(resolution: &str) -> Result<Self> {
        Self::new("data", resolution, DbMode::Read)
    }

    pub fn default_write(resolution: &str) -> Result<Self> {
        Self::new("data", resolution, DbMode::Write)
    }

    pub fn columns(mut self, columns: &[Column]) -> Self {
        let mut columns_str = String::with_capacity(columns.len()*12);
        for (i, c) in columns.iter().enumerate() {
            if i > 0 { columns_str.push(','); }
            columns_str.push('"');
            columns_str.push_str(c.name());
            columns_str.push('"');
        }
        self.columns = Some(columns_str);
        self
    }

    pub fn fetch(&self, ticker: &str, start: NaiveDate, end: NaiveDate) -> Result<Vec<(DateTime<Utc>, Vec<f64>)>> {
        let columns = self.columns.as_deref().unwrap_or(Column::all());
        let sql = format!("SELECT timestamp, {columns}
            FROM candles
            WHERE ticker=?1 AND timestamp >= ?2 AND timestamp < ?3
            ORDER BY timestamp");
        let mut stmt = self.connection.prepare_cached(&sql)?;

        let start_ts = start
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();
        let end_ts = end
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp() + self.period_step;

        stmt.query_map(params![ticker, start_ts, end_ts], |row| {
            let ts: i64 = row.get(0)?;
            let datetime = DateTime::<Utc>::from_timestamp(ts, 0)
                .ok_or(Error::IntegralValueOutOfRange(0, ts))?;

            let column_count = row.as_ref().column_count();
            let mut values = Vec::with_capacity(column_count.saturating_sub(1));
            for idx in 1..column_count {
                values.push(row.get::<_, f64>(idx)?);
            }
            Ok((datetime, values))
        })?
        .collect()
    }

    fn resolution_steps(resolution: &str) -> i64 {
        match resolution {
            "1d" => 86_400,
            _ => unimplemented!("{resolution} is not supported")
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    const FIXTURE_DIR: &str = "tests/fixtures";

    #[test]
    fn fetch_subset_columns_returns_rows() -> rusqlite::Result<()> {
        let md = MarketData::new(FIXTURE_DIR, "1d", DbMode::Read)?
            .columns(&[Column::Open, Column::Close, Column::Volume]);

        let start = NaiveDate::from_ymd_opt(2023, 1, 3).unwrap();
        let end = NaiveDate::from_ymd_opt(2023, 1, 5).unwrap();

        let rows = md.fetch("AAPL", start, end)?;
        assert_eq!(rows.len(), 3);

        let first_ts = Utc.with_ymd_and_hms(2023, 1, 3, 21, 0, 0).unwrap();
        let last_ts = Utc.with_ymd_and_hms(2023, 1, 5, 21, 0, 0).unwrap();
        assert_eq!(rows[0].0, first_ts);
        assert_eq!(rows[2].0, last_ts);

        let approx = |a: f64, b: f64| (a - b).abs() < 1e-9;
        assert!(approx(rows[0].1[0], 130.279_998_779_296_88));
        assert!(approx(rows[0].1[1], 125.069_999_694_824_22));
        assert!(approx(rows[0].1[2], 112_117_500.0));

        Ok(())
    }

    #[test]
    fn fetch_all_columns_errors_on_nulls() {
        let md = MarketData::new(FIXTURE_DIR, "1d", DbMode::Read).unwrap();
        let start = NaiveDate::from_ymd_opt(2023, 1, 3).unwrap();
        let end = NaiveDate::from_ymd_opt(2023, 1, 5).unwrap();

        match md.fetch("AAPL", start, end) {
            Ok(_) => panic!("expected fetch to fail when encountering NULL columns"),
            Err(rusqlite::Error::InvalidColumnType(idx, _, _)) => assert_eq!(idx, 16),
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }
}
