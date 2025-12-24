use chrono::NaiveDate;
use serde::de::{Error as DeError, Visitor};
use serde::{Deserialize, Deserializer};
use std::error::Error;
use std::fmt;
use std::path::Path;


/// Root option chain payload.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RawOptionChain {
    pub date: NaiveDate,
    #[serde(deserialize_with = "de_f64")]
    pub last_price: f64,
    #[serde(default)]
    pub data: Vec<OptionContract>,
}

/// A single option quote row.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct OptionContract {
    // #[serde(rename = "contractID")]
    // pub contract_id: String,
    // pub symbol: String,
    pub expiration: NaiveDate,
    #[serde(deserialize_with = "de_f64")]
    pub strike: f64,
    #[serde(rename = "type")]
    pub option_type: OptionType,
    #[serde(deserialize_with = "de_f64")]
    pub last: f64,
    #[serde(deserialize_with = "de_f64")]
    pub mark: f64,
    #[serde(deserialize_with = "de_f64")]
    pub bid: f64,
    #[serde(default, deserialize_with = "de_f64")]
    pub bid_size: f64,
    #[serde(default, deserialize_with = "de_f64")]
    pub ask: f64,
    #[serde(default, deserialize_with = "de_f64")]
    pub ask_size: f64,
    #[serde(default, deserialize_with = "de_f64")]
    pub volume: f64,
    #[serde(default, deserialize_with = "de_f64")]
    pub open_interest: f64,
    pub date: NaiveDate,
    #[serde(deserialize_with = "de_f64")]
    pub implied_volatility: f64,
    #[serde(deserialize_with = "de_f64")]
    pub delta: f64,
    #[serde(deserialize_with = "de_f64")]
    pub gamma: f64,
    #[serde(deserialize_with = "de_f64")]
    pub theta: f64,
    #[serde(deserialize_with = "de_f64")]
    pub vega: f64,
    #[serde(deserialize_with = "de_f64")]
    pub rho: f64,
    // #[serde(default, deserialize_with = "de_f64")]
    // pub theoretical: f64,
    // #[serde(default, deserialize_with = "de_f64")]
    // pub intrinsic_value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptionType {
    Call,
    Put,
}

/// Errors encountered while parsing filenames or JSON payloads.
#[derive(Debug)]
pub enum OptionChainError {
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl fmt::Display for OptionChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptionChainError::Io(err) => write!(f, "io error: {err}"),
            OptionChainError::Json(err) => write!(f, "json error: {err}"),
        }
    }
}

impl Error for OptionChainError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            OptionChainError::Io(err) => Some(err),
            OptionChainError::Json(err) => Some(err),
        }
    }
}

impl From<std::io::Error> for OptionChainError {
    fn from(err: std::io::Error) -> Self {
        OptionChainError::Io(err)
    }
}

impl From<serde_json::Error> for OptionChainError {
    fn from(err: serde_json::Error) -> Self {
        OptionChainError::Json(err)
    }
}

/// Read a file and deserialize the JSON payload.
///
/// Read a file and deserialize the JSON payload.
///
/// Eagerly reads into memory and uses `from_slice` for speed.
/// Fails if required numeric fields are missing/empty (see `de_f64`).
pub fn parse_option_chain_file(path: impl AsRef<Path>) -> Result<RawOptionChain, OptionChainError> {
    let buf = std::fs::read(path)?;
    let chain: RawOptionChain = serde_json::from_slice(&buf)?;
    Ok(chain)
}

/// Deserialize a chain from any reader (useful for testing).
///
/// Streaming variant; otherwise identical to `parse_option_chain_file`.
pub fn parse_option_chain_reader<R: std::io::Read>(
    reader: R,
) -> Result<RawOptionChain, serde_json::Error> {
    serde_json::from_reader(reader)
}

fn de_f64<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    struct NumVisitor;

    impl<'de> Visitor<'de> for NumVisitor {
        type Value = f64;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("number or numeric string")
        }

        fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E> {
            Ok(v)
        }

        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v as f64)
        }

        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
            Ok(v as f64)
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: DeError,
        {
            v.parse::<f64>().map_err(DeError::custom)
        }

        fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
        where
            E: DeError,
        {
            self.visit_str(&v)
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: DeError,
        {
            Ok(0.0)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: DeError,
        {
            Ok(0.0)
        }
    }

    deserializer.deserialize_any(NumVisitor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn parses_numeric_fields_from_strings() {
        let json = r#"
        {
            "date": "2025-09-08",
            "last_price": "100.5",
            "data": [{
                "contractID": "AAPL251001C00150000",
                "symbol": "AAPL",
                "expiration": "2025-10-01",
                "strike": "150",
                "type": "call",
                "last": "1.23",
                "mark": 1.2,
                "bid": "1.0",
                "bid_size": "2",
                "ask": 1.4,
                "ask_size": 0,
                "volume": "12",
                "open_interest": 5,
                "date": "2025-09-08",
                "implied_volatility": "0.5",
                "delta": "0.6",
                "gamma": "0.1",
                "theta": "-0.01",
                "vega": "0.2",
                "rho": "0.05",
                "theoretical": "1.25",
                "intrinsic_value": "-10.0"
            }]
        }"#;

        let chain: RawOptionChain = serde_json::from_str(json).unwrap();
        assert_eq!(chain.date, NaiveDate::from_ymd_opt(2025, 9, 8).unwrap());
        assert!((chain.last_price - 100.5).abs() < 1e-12);
        assert_eq!(chain.data.len(), 1);
        let opt = &chain.data[0];
        assert_eq!(opt.option_type, OptionType::Call);
        assert!((opt.strike - 150.0).abs() < 1e-12);
        // bid_size/open_interest are currently unused; ensure other fields parsed.
        assert!((opt.implied_volatility - 0.5).abs() < 1e-12);
    }

    #[test]
    fn parses_file() {
        let mut tmp = NamedTempFile::new().unwrap();
        let json = r#"{"date":"2025-09-08","last_price":101.0,"data":[]}"#;
        tmp.write_all(json.as_bytes()).unwrap();

        let parsed = parse_option_chain_file(tmp.path()).unwrap();
        assert_eq!(parsed.date, NaiveDate::from_ymd_opt(2025, 9, 8).unwrap());
        assert_eq!(parsed.last_price, 101.0);
    }

    #[test]
    fn parses_chain_fragment() {
        let json = r#"
        {
          "date": "2025-09-08",
          "last_price": "190.55",
          "data": [
            {
              "contractID": "AAPL250912C00190000",
              "symbol": "AAPL",
              "expiration": "2025-09-12",
              "strike": "190",
              "type": "call",
              "last": "5.1",
              "mark": "5.0",
              "bid": "4.9",
              "bid_size": "12",
              "ask": "5.1",
              "ask_size": "15",
              "volume": "220",
              "open_interest": "310",
              "date": "2025-09-08",
              "implied_volatility": "0.32",
              "delta": "0.62",
              "gamma": "0.04",
              "theta": "-0.06",
              "vega": "0.12",
              "rho": "0.05",
              "theoretical": "5.05",
              "intrinsic_value": "0.0"
            },
            {
              "contractID": "AAPL250912P00190000",
              "symbol": "AAPL",
              "expiration": "2025-09-12",
              "strike": "190",
              "type": "put",
              "last": "4.3",
              "mark": "4.25",
              "bid": "4.1",
              "bid_size": "9",
              "ask": "4.4",
              "ask_size": "11",
              "volume": "180",
              "open_interest": "275",
              "date": "2025-09-08",
              "implied_volatility": "0.35",
              "delta": "-0.38",
              "gamma": "0.05",
              "theta": "-0.04",
              "vega": "0.10",
              "rho": "-0.03",
              "theoretical": "4.28",
              "intrinsic_value": "0.0"
            },
            {
              "contractID": "AAPL250919C00195000",
              "symbol": "AAPL",
              "expiration": "2025-09-19",
              "strike": "195",
              "type": "call",
              "last": "7.8",
              "mark": "7.75",
              "bid": "7.6",
              "bid_size": "8",
              "ask": "7.9",
              "ask_size": "10",
              "volume": "90",
              "open_interest": "140",
              "date": "2025-09-08",
              "implied_volatility": "0.31",
              "delta": "0.58",
              "gamma": "0.03",
              "theta": "-0.05",
              "vega": "0.11",
              "rho": "0.04",
              "theoretical": "7.7",
              "intrinsic_value": "0.0"
            },
            {
              "contractID": "AAPL250919P00195000",
              "symbol": "AAPL",
              "expiration": "2025-09-19",
              "strike": "195",
              "type": "put",
              "last": "6.6",
              "mark": "6.55",
              "bid": "6.4",
              "bid_size": "7",
              "ask": "6.7",
              "ask_size": "9",
              "volume": "75",
              "open_interest": "160",
              "date": "2025-09-08",
              "implied_volatility": "0.34",
              "delta": "-0.42",
              "gamma": "0.04",
              "theta": "-0.05",
              "vega": "0.12",
              "rho": "-0.04",
              "theoretical": "6.6",
              "intrinsic_value": "0.0"
            }
          ]
        }"#;

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(json.as_bytes()).unwrap();
        let chain = parse_option_chain_file(tmp.path()).unwrap();

        assert_eq!(chain.date, NaiveDate::from_ymd_opt(2025, 9, 8).unwrap());
        assert!((chain.last_price - 190.55).abs() < 1e-12);
        assert_eq!(chain.data.len(), 4);

        let calls: Vec<_> = chain.data.iter().filter(|c| c.option_type == OptionType::Call).collect();
        let puts: Vec<_> = chain.data.iter().filter(|c| c.option_type == OptionType::Put).collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(puts.len(), 2);

        let exp_dates: std::collections::HashSet<_> =
            chain.data.iter().map(|c| c.expiration).collect();
        assert_eq!(exp_dates.len(), 2, "expected two expiration dates");

        // Spot check one call and one put to ensure numeric parsing.
        let call = calls.iter().find(|c| c.strike == 190.0).unwrap();
        assert!((call.mark - 5.0).abs() < 1e-12);
        // bid_size/volume are unused in current pipeline; omitted from struct.
        let put = puts.iter().find(|c| c.strike == 195.0).unwrap();
        assert!((put.delta + 0.42).abs() < 1e-12);
    }
}
