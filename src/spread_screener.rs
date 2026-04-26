use crate::raw_option_chain::parse_option_chain_file;
use crate::{
    ChainScreener, LegBuilder, OptionChain, OptionChainDb, OptionType, OptionsDbMode, Position,
    ScreenerCandidate,
};
use chrono::NaiveDate;

use std::path::{Path, PathBuf};

pub struct SpreadScreener {
    queries: &'static [(OptionType, &'static str)],
}

const DEFAULT_QUERIES: [(OptionType, &str); 1] =
    [(OptionType::Put, include_str!("../queries/spread_query.sql"))];

impl Default for SpreadScreener {
    fn default() -> Self {
        Self {
            queries: &DEFAULT_QUERIES,
        }
    }
}

impl SpreadScreener {
    pub fn new(queries: &'static [(OptionType, &'static str)]) -> Self {
        Self { queries }
    }

    pub fn screen_db(
        &self,
        chain: &OptionChainDb,
        option_chains_path: &Path,
    ) -> Vec<ScreenerCandidate> {
        let spreads = self.query_spreads(chain);
        self.build_candidates(option_chains_path, spreads)
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

    fn json_index(path: &Path) -> Vec<(String, PathBuf)> {
        let mut out = Vec::new();
        let Ok(entries) = std::fs::read_dir(path) else {
            return out;
        };

        for entry in entries.flatten() {
            let file_path = entry.path();
            if !file_path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
            {
                continue;
            }
            let Some(ticker) = Self::ticker_from_file_name(&file_path) else {
                continue;
            };
            out.push((ticker, file_path));
        }

        out.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        out
    }

    fn query_spreads(&self, chain: &OptionChainDb) -> Vec<SpreadSpec> {
        let mut spreads = Vec::new();
        for (option_type, query) in self.queries {
            let rows = match chain.query_map(query, |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, f64>(3)?,
                    row.get::<_, f64>(4)?,
                ))
            }) {
                Ok(rows) => rows,
                Err(err) => {
                    eprintln!("screener: query failed ({option_type:?}): {err}");
                    continue;
                }
            };

            spreads.reserve(rows.len());
            for (symbol, expiry, short_strike, long_strike) in rows {
                let Ok(expiry) = NaiveDate::parse_from_str(&expiry, "%Y-%m-%d") else {
                    eprintln!("screener: bad expiry '{expiry}' for {symbol}");
                    continue;
                };

                spreads.push(SpreadSpec {
                    symbol,
                    option_type: *option_type,
                    expiry,
                    short_strike,
                    long_strike,
                });
            }
        }

        spreads
    }

    fn build_candidates(
        &self,
        option_chains_path: &Path,
        mut spreads: Vec<SpreadSpec>,
    ) -> Vec<ScreenerCandidate> {
        if spreads.is_empty() {
            return Vec::new();
        }
        spreads.sort_unstable_by(|a, b| a.symbol.cmp(&b.symbol));

        let json_index = Self::json_index(option_chains_path);
        if json_index.is_empty() {
            eprintln!(
                "screener: no chain json files found in {}",
                option_chains_path.display()
            );
            return Vec::new();
        }

        let mut out = Vec::with_capacity(spreads.len());
        for symbol_spreads in spreads.chunk_by(|a, b| a.symbol == b.symbol) {
            let symbol = symbol_spreads[0].symbol.as_str();
            let Some(idx) = json_index
                .binary_search_by(|(ticker, _)| ticker.as_str().cmp(symbol))
                .ok()
            else {
                eprintln!("{symbol}: chain json not found");
                continue;
            };
            let chain_path = json_index[idx].1.as_path();
            let Ok(raw_chain) = parse_option_chain_file(chain_path)
                .inspect_err(|err| eprintln!("{symbol}: failed to parse json chain: {err}"))
            else {
                continue;
            };

            let option_chain = OptionChain::from_raw(&raw_chain);
            let leg_builder = LegBuilder::from_option_chain(&option_chain);
            let mut positions = Vec::with_capacity(symbol_spreads.len());

            for spec in symbol_spreads {
                let Ok(short_leg) = leg_builder
                    .create(spec.option_type, spec.short_strike, spec.expiry)
                    .inspect_err(|err| eprintln!("{symbol}: leg build error: {err}"))
                else {
                    continue;
                };
                let Ok(long_leg) = leg_builder
                    .create(spec.option_type, spec.long_strike, spec.expiry)
                    .inspect_err(|err| eprintln!("{symbol}: leg build error: {err}"))
                else {
                    continue;
                };

                let mut position = Position::default();
                position.push(short_leg, -1);
                position.push(long_leg, 1);
                positions.push(position);
            }

            if !positions.is_empty() {
                out.push(ScreenerCandidate::new(
                    symbol.to_string(),
                    raw_chain,
                    positions,
                ));
            }
        }

        out
    }
}

#[derive(Debug)]
struct SpreadSpec {
    symbol: String,
    option_type: OptionType,
    expiry: NaiveDate,
    short_strike: f64,
    long_strike: f64,
}

impl ChainScreener for SpreadScreener {
    fn screen(&self, option_chains_path: &Path) -> Vec<ScreenerCandidate> {
        let Ok(chain) = OptionChainDb::new(option_chains_path, OptionsDbMode::Read)
            .inspect_err(|err| eprintln!("screener: failed to open options db: {err}"))
        else {
            return Vec::new();
        };
        self.screen_db(&chain, option_chains_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use std::fs;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    const PUT_QUERY_ROW: &str = "
        SELECT
            'ARM' AS symbol,
            '2025-09-05' AS snap_date,
            '2027-12-17' AS expiry,
            150.0 AS short_strike,
            145.0 AS long_strike
    ";

    const CALL_QUERY_ROW: &str = "
        SELECT
            'ARM' AS symbol,
            '2025-09-05' AS snap_date,
            '2025-09-12' AS expiry,
            145.0 AS short_strike,
            150.0 AS long_strike
    ";

    const SINGLE_PUT_QUERY: &[(OptionType, &str)] = &[(OptionType::Put, PUT_QUERY_ROW)];

    const CALL_AND_PUT_QUERIES: &[(OptionType, &str)] = &[
        (OptionType::Call, CALL_QUERY_ROW),
        (OptionType::Put, PUT_QUERY_ROW),
    ];

    #[test]
    fn screen_returns_expected_put_spread_values() {
        let tmp = tempdir().expect("failed to create temp dir");
        copy_arm_fixture(tmp.path());
        build_options_db(tmp.path());

        let screener = SpreadScreener::new(SINGLE_PUT_QUERY);
        let out = screener.screen(tmp.path());

        assert_eq!(out.len(), 1, "expected one symbol candidate");
        let candidate = &out[0];
        assert_eq!(candidate.ticker, "ARM");
        assert_eq!(
            candidate.raw_chain.date,
            NaiveDate::from_ymd_opt(2025, 9, 5).expect("invalid date"),
        );
        assert_eq!(candidate.positions.len(), 1);

        let pos = &candidate.positions[0];
        assert!(
            matches_credit_spread(
                pos,
                OptionType::Put,
                NaiveDate::from_ymd_opt(2027, 12, 17).expect("invalid date"),
                150.0,
                145.0
            ),
            "expected ARM 2027-12-17 put spread 150/145",
        );
        // expected premium = long(145 mid=38.25) - short(150 mid=40.525) = -2.275
        assert!(
            (pos.premium - (-2.275)).abs() < 1e-9,
            "unexpected premium {}",
            pos.premium
        );
    }

    #[test]
    fn screen_returns_expected_values_for_mixed_call_and_put_queries() {
        let tmp = tempdir().expect("failed to create temp dir");
        copy_arm_fixture(tmp.path());
        build_options_db(tmp.path());

        let screener = SpreadScreener::new(CALL_AND_PUT_QUERIES);
        let out = screener.screen(tmp.path());

        assert_eq!(out.len(), 1);
        let candidate = &out[0];
        assert_eq!(candidate.ticker, "ARM");
        assert_eq!(candidate.positions.len(), 2);

        let call_exp = NaiveDate::from_ymd_opt(2025, 9, 12).expect("invalid date");
        let put_exp = NaiveDate::from_ymd_opt(2027, 12, 17).expect("invalid date");

        let call_pos = candidate
            .positions
            .iter()
            .find(|p| matches_credit_spread(p, OptionType::Call, call_exp, 145.0, 150.0))
            .expect("expected call spread 145/150");
        let put_pos = candidate
            .positions
            .iter()
            .find(|p| matches_credit_spread(p, OptionType::Put, put_exp, 150.0, 145.0))
            .expect("expected put spread 150/145");

        // call premium = long(150 mid=0.33) - short(145 mid=0.92) = -0.59
        assert!(
            (call_pos.premium - (-0.59)).abs() < 1e-9,
            "unexpected call premium"
        );
        // put premium = long(145 mid=38.25) - short(150 mid=40.525) = -2.275
        assert!(
            (put_pos.premium - (-2.275)).abs() < 1e-9,
            "unexpected put premium"
        );
    }

    #[test]
    fn screen_db_uses_existing_db() {
        let tmp = tempdir().expect("failed to create temp dir");
        copy_arm_fixture(tmp.path());
        let mut db = OptionChainDb::default_memory(
            tmp.path()
                .to_str()
                .expect("temp directory path should be valid UTF-8"),
        )
        .expect("failed to create memory options db");
        db.ingest_from_json()
            .expect("failed to ingest fixture json into memory options db");

        let screener = SpreadScreener::new(SINGLE_PUT_QUERY);
        let out = screener.screen_db(&db, tmp.path());

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].ticker, "ARM");
        assert_eq!(out[0].positions.len(), 1);
    }

    fn copy_arm_fixture(dst_dir: &Path) {
        let fixture = arm_fixture_path();
        let dst = dst_dir.join("ARM_option_chain_20250908_160038.json");
        fs::copy(&fixture, &dst).unwrap_or_else(|err| {
            panic!(
                "failed to copy fixture from {} to {}: {err}",
                fixture.display(),
                dst.display()
            )
        });
    }

    fn build_options_db(dir: &Path) {
        let mut db = OptionChainDb::default_write(
            dir.to_str()
                .expect("temp directory path should be valid UTF-8"),
        )
        .expect("failed to create options db");
        db.ingest_from_json()
            .expect("failed to ingest fixture json into options db");
    }

    fn arm_fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("ARM_option_chain_20250908_160038.json")
    }

    fn matches_credit_spread(
        position: &Position,
        option_type: OptionType,
        expiry: NaiveDate,
        short_strike: f64,
        long_strike: f64,
    ) -> bool {
        if position.legs.len() != 2 {
            return false;
        }
        let Some((short_leg, short_qty)) = position.legs.iter().find(|(_, qty)| *qty == -1) else {
            return false;
        };
        let Some((long_leg, long_qty)) = position.legs.iter().find(|(_, qty)| *qty == 1) else {
            return false;
        };

        *short_qty == -1
            && *long_qty == 1
            && short_leg.option_type == option_type
            && long_leg.option_type == option_type
            && short_leg.expire == expiry
            && long_leg.expire == expiry
            && short_leg.strike == short_strike
            && long_leg.strike == long_strike
    }
}
