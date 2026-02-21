//! SQLite market data access for historical candles and indicators.

use chrono::{DateTime, Datelike, NaiveDate, TimeZone, Utc};
use chrono_tz::America::New_York;
use rusqlite::{Connection, Error, OpenFlags, Result, params};
use serde::Deserialize;
use std::collections::VecDeque;
use std::error::Error as StdError;
use std::fmt::{Display, Formatter};
use std::time::Duration;

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
    IvRank12,
}

const COLUMN_COUNT: usize = 27;
const COLUMN_NAMES: [&str; COLUMN_COUNT] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjopen",
    "adjhigh",
    "adjlow",
    "adjclose",
    "co_log",
    "oc_log",
    "cc_log",
    "co_log_adj",
    "oc_log_adj",
    "cc_log_adj",
    "vol20",
    "vol20_adj",
    "vol_trend20",
    "adjclose_trend20",
    "vol_rank6",
    "vol_rank9",
    "vol_rank12",
    "iv",
    "iv_trend20",
    "iv_rank6",
    "iv_rank9",
    "iv_rank12",
];

const COLUMN_NAMES_ALL: &str = "open, high, low, close, volume,
    adjopen, adjhigh, adjlow, adjclose,
    co_log, oc_log, cc_log, co_log_adj, oc_log_adj,
    cc_log_adj, vol20, vol20_adj, vol_trend20,
    adjclose_trend20, vol_rank6, vol_rank9, vol_rank12,
    iv, iv_trend20, iv_rank6, iv_rank9, iv_rank12";
const VOL_WINDOW: usize = 20;
const PCT6: usize = 126;
const PCT9: usize = 189;
const PCT12: usize = 252;

impl Column {
    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }
    #[inline]
    pub fn name(self) -> &'static str {
        COLUMN_NAMES[self.idx()]
    }
    #[inline]
    pub fn all() -> &'static str {
        COLUMN_NAMES_ALL
    }
}

pub enum DbMode {
    Read,
    Write,
}

#[derive(Debug)]
pub enum IngestError {
    Db(rusqlite::Error),
    Http(Box<ureq::Error>),
    Decode(std::io::Error),
}

impl Display for IngestError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Db(e) => write!(f, "database error: {e}"),
            Self::Http(e) => write!(f, "http request failed: {e}"),
            Self::Decode(e) => write!(f, "response decode failed: {e}"),
        }
    }
}

impl StdError for IngestError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Db(e) => Some(e),
            Self::Http(e) => Some(e),
            Self::Decode(e) => Some(e),
        }
    }
}

impl From<rusqlite::Error> for IngestError {
    fn from(value: rusqlite::Error) -> Self {
        Self::Db(value)
    }
}

impl From<ureq::Error> for IngestError {
    fn from(value: ureq::Error) -> Self {
        Self::Http(Box::new(value))
    }
}

impl From<std::io::Error> for IngestError {
    fn from(value: std::io::Error) -> Self {
        Self::Decode(value)
    }
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

                conn.execute_batch(
                    r#"
                    PRAGMA query_only=ON;
                    PRAGMA mmap_size=536870912;     -- 512MB (>= DB size)
                    PRAGMA cache_size=-65536;       -- ~64MB (raise to -131072 if few conns)
                    PRAGMA temp_store=MEMORY;
                "#,
                )?;
                conn
            }
            DbMode::Write => {
                let conn = Connection::open(path)?;
                conn.execute_batch(
                    r#"
                    PRAGMA journal_mode=WAL;
                    PRAGMA synchronous=NORMAL;
                    PRAGMA wal_autocheckpoint=0;
                    PRAGMA temp_store=MEMORY;
                    PRAGMA cache_size=-131072;      -- ~128MB

                    CREATE TABLE IF NOT EXISTS candles (
                        ticker TEXT,
                        timestamp INTEGER,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume REAL,
                        adjopen REAL,
                        adjhigh REAL,
                        adjlow REAL,
                        adjclose REAL,
                        co_log REAL,
                        oc_log REAL,
                        cc_log REAL,
                        co_log_adj REAL,
                        oc_log_adj REAL,
                        cc_log_adj REAL,
                        vol20 REAL,
                        vol20_adj REAL,
                        vol_trend20 REAL,
                        adjclose_trend20 REAL,
                        vol_rank6 REAL,
                        vol_rank9 REAL,
                        vol_rank12 REAL,
                        iv REAL,
                        iv_trend20 REAL,
                        iv_rank6 REAL,
                        iv_rank9 REAL,
                        iv_rank12 REAL,
                        PRIMARY KEY (ticker, timestamp)
                    );
                "#,
                )?;
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
        let mut columns_str = String::with_capacity(columns.len() * 12);
        for (i, c) in columns.iter().enumerate() {
            if i > 0 {
                columns_str.push(',');
            }
            columns_str.push('"');
            columns_str.push_str(c.name());
            columns_str.push('"');
        }
        self.columns = Some(columns_str);
        self
    }

    pub fn fetch(
        &self,
        ticker: &str,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<(DateTime<Utc>, Vec<f64>)>> {
        let columns = self.columns.as_deref().unwrap_or(Column::all());
        let sql = format!(
            "SELECT timestamp, {columns}
            FROM candles
            WHERE ticker=?1 AND timestamp >= ?2 AND timestamp < ?3
            ORDER BY timestamp"
        );
        let mut stmt = self.connection.prepare_cached(&sql)?;

        let start_ts = start.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
        let end_ts = end.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp() + self.period_step;

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

    pub fn ingest(
        &mut self,
        ticker: &str,
        resolution: &str,
        start: NaiveDate,
        end: NaiveDate,
    ) -> std::result::Result<(), IngestError> {
        let step = Self::resolution_steps(resolution);
        let start_ts = start.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
        let end_ts_exclusive = end.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp() + step;

        if end_ts_exclusive <= start_ts {
            return Ok(());
        }

        let mut period1 = start_ts;
        if resolution == "1d" {
            period1 -= 12 * 3600;
        }

        let mut body = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(15))
            .build()
            .get(&format!(
                "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            ))
            .query("period1", &period1.to_string())
            .query("period2", &end_ts_exclusive.to_string())
            .query("interval", resolution)
            .query("events", "history")
            .query("includeAdjustedClose", "true")
            .set("User-Agent", "Mozilla/5.0")
            .call()
            .map_err(IngestError::from)?
            .into_json::<YahooChartResponse>()
            .map_err(IngestError::Decode)?;
        if body.chart.result.is_empty() {
            return Ok(());
        }

        let mut result = body.chart.result.swap_remove(0);
        if result.indicators.quote.is_empty() {
            return Ok(());
        }

        let quote = result.indicators.quote.swap_remove(0);
        let timestamps = result.timestamp;
        let open = quote.open;
        let high = quote.high;
        let low = quote.low;
        let close = quote.close;
        let volume = quote.volume;
        let adjclose = if result.indicators.adjclose.is_empty() {
            Vec::new()
        } else {
            result.indicators.adjclose.swap_remove(0).adjclose
        };

        let n_base = *[
            timestamps.len(),
            open.len(),
            high.len(),
            low.len(),
            close.len(),
            volume.len(),
        ]
        .iter()
        .min()
        .unwrap_or(&0);

        if n_base == 0 {
            return Ok(());
        }

        let ingest_adj = resolution == "1d" && adjclose.len() >= n_base;
        let mut rows = Vec::with_capacity(n_base);
        if ingest_adj {
            let n = n_base.min(adjclose.len());
            for i in 0..n {
                let (o, h, l, c) = match (open[i], high[i], low[i], close[i]) {
                    (Some(o), Some(h), Some(l), Some(c)) => (o, h, l, c),
                    _ => continue,
                };
                let ac = match adjclose[i] {
                    Some(v) => v,
                    None => continue,
                };
                if c <= 0.0 {
                    continue;
                }
                let ts = if resolution == "1d" {
                    match Self::daily_close_timestamp(timestamps[i]) {
                        Some(v) => v,
                        None => continue,
                    }
                } else {
                    timestamps[i]
                };
                if ts < start_ts || ts >= end_ts_exclusive {
                    continue;
                }
                let ratio = ac / c;
                rows.push(IngestRow {
                    timestamp: ts,
                    open: o,
                    high: h,
                    low: l,
                    close: c,
                    volume: volume[i].unwrap_or(0.0),
                    adjopen: Some(o * ratio),
                    adjhigh: Some(h * ratio),
                    adjlow: Some(l * ratio),
                    adjclose: Some(ac),
                });
            }
        } else {
            for i in 0..n_base {
                let (o, h, l, c) = match (open[i], high[i], low[i], close[i]) {
                    (Some(o), Some(h), Some(l), Some(c)) => (o, h, l, c),
                    _ => continue,
                };
                let ts = if resolution == "1d" {
                    match Self::daily_close_timestamp(timestamps[i]) {
                        Some(v) => v,
                        None => continue,
                    }
                } else {
                    timestamps[i]
                };
                if ts < start_ts || ts >= end_ts_exclusive {
                    continue;
                }
                rows.push(IngestRow {
                    timestamp: ts,
                    open: o,
                    high: h,
                    low: l,
                    close: c,
                    volume: volume[i].unwrap_or(0.0),
                    adjopen: None,
                    adjhigh: None,
                    adjlow: None,
                    adjclose: None,
                });
            }
        }

        if rows.is_empty() {
            return Ok(());
        }

        let ticker_upper = ticker.to_ascii_uppercase();
        let derived = if resolution == "1d" {
            self.compute_daily_derived(&ticker_upper, &rows)
                .map_err(IngestError::Db)?
        } else {
            DailyDerived::new(rows.len())
        };

        let tx = self.connection.transaction().map_err(IngestError::Db)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR IGNORE INTO candles
                (ticker,timestamp,open,high,low,close,volume,
                 adjopen,adjhigh,adjlow,adjclose,
                 co_log,oc_log,cc_log,co_log_adj,oc_log_adj,cc_log_adj,
                 vol20,vol20_adj,vol_trend20,adjclose_trend20,
                 vol_rank6,vol_rank9,vol_rank12)
                VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24)"
            )
            .map_err(IngestError::Db)?;

            for (i, row) in rows.iter().enumerate() {
                stmt.execute(params![
                    &ticker_upper,
                    row.timestamp,
                    row.open,
                    row.high,
                    row.low,
                    row.close,
                    row.volume,
                    row.adjopen,
                    row.adjhigh,
                    row.adjlow,
                    row.adjclose,
                    derived.co_log[i],
                    derived.oc_log[i],
                    derived.cc_log[i],
                    derived.co_log_adj[i],
                    derived.oc_log_adj[i],
                    derived.cc_log_adj[i],
                    derived.vol20[i],
                    derived.vol20_adj[i],
                    derived.vol_trend20[i],
                    derived.adjclose_trend20[i],
                    derived.vol_rank6[i],
                    derived.vol_rank9[i],
                    derived.vol_rank12[i]
                ])
                .map_err(IngestError::Db)?;
            }
        }
        tx.commit().map_err(IngestError::Db)?;
        Ok(())
    }

    fn compute_daily_derived(&self, ticker: &str, rows: &[IngestRow]) -> Result<DailyDerived> {
        let n = rows.len();
        let mut derived = DailyDerived::new(n);
        let first_ts = rows[0].timestamp;

        let (mut prev_close, mut prev_adjclose) = self.get_prev_close_adj(ticker, first_ts)?;
        for (i, row) in rows.iter().enumerate() {
            let oc = Self::log_ratio(Some(row.close), Some(row.open));
            derived.oc_log[i] = oc;
            let oc_adj = Self::log_ratio(row.adjclose, row.adjopen);
            derived.oc_log_adj[i] = oc_adj;

            let co = Self::log_ratio(Some(row.open), prev_close);
            derived.co_log[i] = co;
            derived.cc_log[i] = match (co, oc) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            };

            let co_adj = Self::log_ratio(row.adjopen, prev_adjclose);
            derived.co_log_adj[i] = co_adj;
            derived.cc_log_adj[i] = match (co_adj, oc_adj) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            };

            prev_close = if row.close > 0.0 {
                Some(row.close)
            } else {
                None
            };
            prev_adjclose = match row.adjclose {
                Some(v) if v > 0.0 => Some(v),
                _ => None,
            };
        }

        let (mut prior_cc, mut prior_cc_adj) =
            self.get_prior_cc_logs(ticker, first_ts, (VOL_WINDOW - 1) as i64)?;
        let prior_cc_len = prior_cc.len();
        let prior_cc_adj_len = prior_cc_adj.len();
        prior_cc.extend_from_slice(&derived.cc_log);
        prior_cc_adj.extend_from_slice(&derived.cc_log_adj);
        derived.vol20 = Self::rolling_annvol_from_cc(&prior_cc, prior_cc_len, VOL_WINDOW);
        derived.vol20_adj =
            Self::rolling_annvol_from_cc(&prior_cc_adj, prior_cc_adj_len, VOL_WINDOW);

        let prior_vol20_adj = self.get_prior_vol20_adj(ticker, first_ts, PCT12 as i64)?;
        let mut pct6 = PercentileWindow::from_seed(PCT6, &prior_vol20_adj);
        let mut pct9 = PercentileWindow::from_seed(PCT9, &prior_vol20_adj);
        let mut pct12 = PercentileWindow::from_seed(PCT12, &prior_vol20_adj);
        for i in 0..n {
            if let Some(curr) = derived.vol20_adj[i] {
                derived.vol_rank6[i] = pct6.rank(curr);
                derived.vol_rank9[i] = pct9.rank(curr);
                derived.vol_rank12[i] = pct12.rank(curr);
                pct6.push(curr);
                pct9.push(curr);
                pct12.push(curr);
            }
        }

        let prior_vol_base =
            self.get_prior_voltrend_base(ticker, first_ts, (VOL_WINDOW - 1) as i64)?;
        let prior_vol_len = prior_vol_base.len();
        let mut combined_vol = prior_vol_base;
        combined_vol.reserve(n);
        for i in 0..n {
            combined_vol.push(derived.vol20_adj[i].or(derived.vol20[i]));
        }
        let vol_trend = Self::rolling_trend_tstat(&combined_vol, VOL_WINDOW);
        derived.vol_trend20[..n].copy_from_slice(&vol_trend[prior_vol_len..(prior_vol_len + n)]);

        let prior_adj_base =
            self.get_prior_adjclose_values(ticker, first_ts, (VOL_WINDOW - 1) as i64)?;
        let prior_adj_len = prior_adj_base.len();
        let mut combined_adj = prior_adj_base;
        combined_adj.reserve(n);
        for row in rows {
            combined_adj.push(row.adjclose.or(Some(row.close)));
        }
        let adj_trend = Self::rolling_trend_tstat(&combined_adj, VOL_WINDOW);
        derived.adjclose_trend20[..n]
            .copy_from_slice(&adj_trend[prior_adj_len..(prior_adj_len + n)]);

        Ok(derived)
    }

    fn get_prev_close_adj(
        &self,
        ticker: &str,
        first_ts: i64,
    ) -> Result<(Option<f64>, Option<f64>)> {
        let mut stmt = self.connection.prepare_cached(
            "SELECT close, adjclose
            FROM candles
            WHERE ticker=?1 AND timestamp < ?2
            ORDER BY timestamp DESC LIMIT 1",
        )?;
        let mut rows = stmt.query(params![ticker, first_ts])?;
        if let Some(row) = rows.next()? {
            Ok((row.get(0)?, row.get(1)?))
        } else {
            Ok((None, None))
        }
    }

    #[allow(clippy::type_complexity)]
    fn get_prior_cc_logs(
        &self,
        ticker: &str,
        first_ts: i64,
        limit: i64,
    ) -> Result<(Vec<Option<f64>>, Vec<Option<f64>>)> {
        let mut stmt = self.connection.prepare_cached(
            "SELECT cc_log, cc_log_adj
            FROM candles
            WHERE ticker=?1 AND timestamp < ?2
            ORDER BY timestamp DESC LIMIT ?3",
        )?;
        let mut raw = Vec::new();
        let mut adj = Vec::new();
        let rows = stmt.query_map(params![ticker, first_ts, limit], |row| {
            Ok((row.get::<_, Option<f64>>(0)?, row.get::<_, Option<f64>>(1)?))
        })?;

        for row in rows {
            let (cc, cc_adj) = row?;
            raw.push(cc);
            adj.push(cc_adj);
        }
        raw.reverse();
        adj.reverse();
        Ok((raw, adj))
    }

    fn get_prior_voltrend_base(
        &self,
        ticker: &str,
        first_ts: i64,
        limit: i64,
    ) -> Result<Vec<Option<f64>>> {
        let mut stmt = self.connection.prepare_cached(
            "SELECT vol20_adj, vol20
            FROM candles
            WHERE ticker=?1 AND timestamp < ?2
            ORDER BY timestamp DESC LIMIT ?3",
        )?;
        let mut values = Vec::new();
        let rows = stmt.query_map(params![ticker, first_ts, limit], |row| {
            Ok((row.get::<_, Option<f64>>(0)?, row.get::<_, Option<f64>>(1)?))
        })?;
        for row in rows {
            let (adj, raw) = row?;
            values.push(adj.or(raw));
        }
        values.reverse();
        Ok(values)
    }

    fn get_prior_adjclose_values(
        &self,
        ticker: &str,
        first_ts: i64,
        limit: i64,
    ) -> Result<Vec<Option<f64>>> {
        let mut stmt = self.connection.prepare_cached(
            "SELECT adjclose, close
            FROM candles
            WHERE ticker=?1 AND timestamp < ?2
            ORDER BY timestamp DESC LIMIT ?3",
        )?;
        let mut values = Vec::new();
        let rows = stmt.query_map(params![ticker, first_ts, limit], |row| {
            Ok((row.get::<_, Option<f64>>(0)?, row.get::<_, Option<f64>>(1)?))
        })?;
        for row in rows {
            let (adj, raw) = row?;
            values.push(adj.or(raw));
        }
        values.reverse();
        Ok(values)
    }

    fn get_prior_vol20_adj(&self, ticker: &str, first_ts: i64, limit: i64) -> Result<Vec<f64>> {
        let mut stmt = self.connection.prepare_cached(
            "SELECT vol20_adj
            FROM candles
            WHERE ticker=?1 AND timestamp < ?2 AND vol20_adj IS NOT NULL
            ORDER BY timestamp DESC LIMIT ?3",
        )?;
        let mut values = Vec::new();
        let rows = stmt.query_map(params![ticker, first_ts, limit], |row| row.get::<_, f64>(0))?;
        for row in rows {
            values.push(row?);
        }
        values.reverse();
        Ok(values)
    }

    fn log_ratio(num: Option<f64>, den: Option<f64>) -> Option<f64> {
        match (num, den) {
            (Some(n), Some(d)) if n > 0.0 && d > 0.0 => Some((n / d).ln()),
            _ => None,
        }
    }

    fn rolling_annvol_from_cc(
        values: &[Option<f64>],
        batch_offset: usize,
        window: usize,
    ) -> Vec<Option<f64>> {
        let batch_len = values.len().saturating_sub(batch_offset);
        let mut result = vec![None; batch_len];
        if batch_len == 0 {
            return result;
        }
        if window < 2 {
            return result;
        }

        let mut prefix_sum = Vec::with_capacity(values.len() + 1);
        let mut prefix_sum2 = Vec::with_capacity(values.len() + 1);
        let mut prefix_valid = Vec::with_capacity(values.len() + 1);
        prefix_sum.push(0.0);
        prefix_sum2.push(0.0);
        prefix_valid.push(0u32);

        let mut push_prefix = |value: Option<f64>| {
            let mut next_sum = *prefix_sum.last().unwrap();
            let mut next_sum2 = *prefix_sum2.last().unwrap();
            let mut next_valid = *prefix_valid.last().unwrap();
            if let Some(r) = value {
                next_sum += r;
                next_sum2 += r * r;
                next_valid += 1;
            }
            prefix_sum.push(next_sum);
            prefix_sum2.push(next_sum2);
            prefix_valid.push(next_valid);
        };

        for value in values.iter().copied() {
            push_prefix(value);
        }

        let window_f = window as f64;
        let required_valid = window as u32;
        for (batch_idx, out) in result.iter_mut().enumerate() {
            let all_idx = batch_offset + batch_idx;
            let end = all_idx + 1;
            if end < window {
                continue;
            }
            let start = end - window;
            let valid = prefix_valid[end] - prefix_valid[start];
            if valid != required_valid {
                continue;
            }

            let sum = prefix_sum[end] - prefix_sum[start];
            let sum2 = prefix_sum2[end] - prefix_sum2[start];
            let mean = sum / window_f;
            let var = (sum2 - window_f * mean * mean) / (window_f - 1.0);
            *out = Some(if var > 0.0 { (252.0 * var).sqrt() } else { 0.0 });
        }

        result
    }

    fn rolling_trend_tstat(values: &[Option<f64>], window: usize) -> Vec<Option<f64>> {
        let mut out = vec![None; values.len()];
        if window < 2 || values.is_empty() {
            return out;
        }

        let n = window as f64;
        let mean_t = (n - 1.0) / 2.0;
        let mut t = Vec::with_capacity(window);
        let mut tt = 0.0;
        for i in 0..window {
            let ti = i as f64 - mean_t;
            t.push(ti);
            tt += ti * ti;
        }
        if tt <= 0.0 {
            return out;
        }
        let tt_sqrt = tt.sqrt();
        let mut yq: VecDeque<f64> = VecDeque::with_capacity(window);

        for (i, value) in values.iter().copied().enumerate() {
            let Some(v) = value else {
                yq.clear();
                continue;
            };
            if v <= 0.0 {
                yq.clear();
                continue;
            }

            yq.push_back(v.ln());
            if yq.len() > window {
                yq.pop_front();
            }
            if yq.len() != window {
                continue;
            }

            let ymean = yq.iter().copied().sum::<f64>() / n;
            let mut dot_ty = 0.0;
            let mut sum_y2 = 0.0;
            for (j, y) in yq.iter().copied().enumerate() {
                dot_ty += t[j] * y;
                sum_y2 += y * y;
            }

            let beta = dot_ty / tt;
            let sst = sum_y2 - n * ymean * ymean;
            let ssr = beta * beta * tt;
            let sse = sst - ssr;
            let dof = n - 2.0;
            out[i] = if dof <= 0.0 || sse < 0.0 {
                Some(0.0)
            } else {
                let se_beta = (sse / dof).sqrt() / tt_sqrt;
                Some(if se_beta > 0.0 { beta / se_beta } else { 0.0 })
            };
        }

        out
    }

    fn daily_close_timestamp(timestamp: i64) -> Option<i64> {
        let dt_utc = DateTime::<Utc>::from_timestamp(timestamp, 0)?;
        let date = dt_utc.with_timezone(&New_York).date_naive();
        New_York
            .with_ymd_and_hms(date.year(), date.month(), date.day(), 16, 0, 0)
            .single()
            .map(|dt| dt.with_timezone(&Utc).timestamp())
    }

    fn resolution_steps(resolution: &str) -> i64 {
        match resolution {
            "1d" => 86_400,
            _ => unimplemented!("{resolution} is not supported"),
        }
    }
}

#[derive(Copy, Clone)]
struct IngestRow {
    timestamp: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    adjopen: Option<f64>,
    adjhigh: Option<f64>,
    adjlow: Option<f64>,
    adjclose: Option<f64>,
}

struct DailyDerived {
    co_log: Vec<Option<f64>>,
    oc_log: Vec<Option<f64>>,
    cc_log: Vec<Option<f64>>,
    co_log_adj: Vec<Option<f64>>,
    oc_log_adj: Vec<Option<f64>>,
    cc_log_adj: Vec<Option<f64>>,
    vol20: Vec<Option<f64>>,
    vol20_adj: Vec<Option<f64>>,
    vol_trend20: Vec<Option<f64>>,
    adjclose_trend20: Vec<Option<f64>>,
    vol_rank6: Vec<Option<f64>>,
    vol_rank9: Vec<Option<f64>>,
    vol_rank12: Vec<Option<f64>>,
}

impl DailyDerived {
    fn new(len: usize) -> Self {
        Self {
            co_log: vec![None; len],
            oc_log: vec![None; len],
            cc_log: vec![None; len],
            co_log_adj: vec![None; len],
            oc_log_adj: vec![None; len],
            cc_log_adj: vec![None; len],
            vol20: vec![None; len],
            vol20_adj: vec![None; len],
            vol_trend20: vec![None; len],
            adjclose_trend20: vec![None; len],
            vol_rank6: vec![None; len],
            vol_rank9: vec![None; len],
            vol_rank12: vec![None; len],
        }
    }
}

struct PercentileWindow {
    max_len: usize,
    queue: VecDeque<f64>,
    sorted: Vec<f64>,
}

impl PercentileWindow {
    fn from_seed(max_len: usize, values: &[f64]) -> Self {
        let mut window = Self {
            max_len,
            queue: VecDeque::with_capacity(max_len),
            sorted: Vec::with_capacity(max_len),
        };
        for &value in values {
            window.push(value);
        }
        window
    }

    fn rank(&self, value: f64) -> Option<f64> {
        if self.queue.len() == self.max_len {
            let cnt = self.sorted.partition_point(|x| *x <= value);
            Some(cnt as f64 / self.max_len as f64)
        } else {
            None
        }
    }

    fn push(&mut self, value: f64) {
        let pos = self.sorted.partition_point(|x| *x <= value);
        self.sorted.insert(pos, value);
        self.queue.push_back(value);

        if self.queue.len() <= self.max_len {
            return;
        }

        if let Some(old) = self.queue.pop_front() {
            let first_ge = self.sorted.partition_point(|x| *x < old);
            if first_ge < self.sorted.len() && self.sorted[first_ge] == old {
                self.sorted.remove(first_ge);
                return;
            }
            if let Some(idx) = self.sorted.iter().position(|x| *x == old) {
                self.sorted.remove(idx);
            }
        }
    }
}

#[derive(Default, Deserialize)]
struct YahooChartResponse {
    #[serde(default)]
    chart: YahooChart,
}

#[derive(Default, Deserialize)]
struct YahooChart {
    #[serde(default)]
    result: Vec<YahooResult>,
}

#[derive(Default, Deserialize)]
struct YahooResult {
    #[serde(default)]
    timestamp: Vec<i64>,
    #[serde(default)]
    indicators: YahooIndicators,
}

#[derive(Default, Deserialize)]
struct YahooIndicators {
    #[serde(default)]
    quote: Vec<YahooQuote>,
    #[serde(default)]
    adjclose: Vec<YahooAdjClose>,
}

#[derive(Default, Deserialize)]
struct YahooQuote {
    #[serde(default)]
    open: Vec<Option<f64>>,
    #[serde(default)]
    high: Vec<Option<f64>>,
    #[serde(default)]
    low: Vec<Option<f64>>,
    #[serde(default)]
    close: Vec<Option<f64>>,
    #[serde(default)]
    volume: Vec<Option<f64>>,
}

#[derive(Default, Deserialize)]
struct YahooAdjClose {
    #[serde(default)]
    adjclose: Vec<Option<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    const FIXTURE_DIR: &str = "tests/fixtures";

    #[test]
    fn fetch_subset_columns_returns_rows() -> rusqlite::Result<()> {
        let md = MarketData::new(FIXTURE_DIR, "1d", DbMode::Read)?.columns(&[
            Column::Open,
            Column::Close,
            Column::Volume,
        ]);

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

    #[test]
    fn rolling_annvol_from_cc_matches_expected() {
        let values = vec![
            None,
            Some(0.01),
            Some(0.02),
            Some(0.015),
            Some(0.03),
            Some(0.025),
            None,
            Some(0.01),
            Some(0.005),
            Some(0.02),
        ];
        let got = MarketData::rolling_annvol_from_cc(&values, 3, 4);
        let expected = vec![
            None,
            Some(0.135_554_417_117_259_57),
            Some(0.102_469_507_659_596_03),
            None,
            None,
            None,
            None,
        ];

        assert_option_vec_approx(&got, &expected, 1e-12);
    }

    #[test]
    fn rolling_trend_tstat_matches_expected() {
        let values = vec![
            Some(0.20),
            Some(0.21),
            Some(0.22),
            Some(0.23),
            Some(0.24),
            None,
            Some(0.25),
            Some(0.26),
            Some(0.27),
            Some(0.28),
            Some(0.29),
        ];
        let got = MarketData::rolling_trend_tstat(&values, 5);
        let expected = vec![
            None,
            None,
            None,
            None,
            Some(64.234_816_495_169_16),
            None,
            None,
            None,
            None,
            None,
            Some(78.905_736_717_422_43),
        ];

        assert_option_vec_approx(&got, &expected, 1e-8);
    }

    fn assert_option_vec_approx(actual: &[Option<f64>], expected: &[Option<f64>], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            match (a, e) {
                (Some(av), Some(ev)) => assert!(
                    (av - ev).abs() <= tol,
                    "idx={idx}: got {av}, expected {ev}, tol={tol}"
                ),
                (None, None) => {}
                _ => panic!("idx={idx}: got {:?}, expected {:?}", a, e),
            }
        }
    }
}
