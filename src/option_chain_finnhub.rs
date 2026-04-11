//! Finnhub option-chain client normalized into `RawOptionChain`.

use crate::raw_option_chain::{OptionContract, OptionType, RawOptionChain};
use crate::serde_helpers::de_f64;
use chrono::NaiveDate;
use serde::Deserialize;
use std::error::Error as StdError;
use std::fmt::{self, Display, Formatter};
use std::time::Duration;

const DEFAULT_URL: &str = "https://finnhub.io/api/v1/stock/option-chain";
const DEFAULT_RETRY_DELAY_SECS: u64 = 15;

#[derive(Debug)]
pub enum FinnhubOptionChainError {
    InvalidSymbol,
    Http(Box<ureq::Error>),
    Decode(std::io::Error),
}

impl Display for FinnhubOptionChainError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSymbol => write!(f, "symbol is empty"),
            Self::Http(err) => write!(f, "http request failed: {err}"),
            Self::Decode(err) => write!(f, "response decode failed: {err}"),
        }
    }
}

impl StdError for FinnhubOptionChainError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::InvalidSymbol => None,
            Self::Http(err) => Some(err),
            Self::Decode(err) => Some(err),
        }
    }
}

impl From<ureq::Error> for FinnhubOptionChainError {
    fn from(value: ureq::Error) -> Self {
        Self::Http(Box::new(value))
    }
}

impl From<std::io::Error> for FinnhubOptionChainError {
    fn from(value: std::io::Error) -> Self {
        Self::Decode(value)
    }
}

#[derive(Clone)]
pub struct FinnhubOptionChainClient {
    agent: ureq::Agent,
    token: String,
    url: String,
    retry_delay: Duration,
    trade_date: NaiveDate,
}

impl FinnhubOptionChainClient {
    pub fn new(token: impl Into<String>, trade_date: NaiveDate) -> Self {
        Self {
            agent: ureq::Agent::new(),
            token: token.into(),
            url: DEFAULT_URL.to_string(),
            retry_delay: Duration::from_secs(DEFAULT_RETRY_DELAY_SECS),
            trade_date,
        }
    }

    pub fn retry_delay(mut self, retry_delay: Duration) -> Self {
        self.retry_delay = retry_delay;
        self
    }

    pub fn trade_date(&self) -> NaiveDate {
        self.trade_date
    }

    pub fn fetch(&self, symbol: &str) -> Result<RawOptionChain, FinnhubOptionChainError> {
        let symbol = normalized_symbol(symbol)?;

        loop {
            match self.fetch_once(&symbol) {
                Ok(chain) => return Ok(chain),
                Err(err @ FinnhubOptionChainError::Decode(_)) => return Err(err),
                Err(err @ FinnhubOptionChainError::InvalidSymbol) => return Err(err),
                Err(FinnhubOptionChainError::Http(err)) => {
                    eprintln!("{err}");
                    eprintln!("Waiting {} seconds...", self.retry_delay.as_secs());
                    std::thread::sleep(self.retry_delay);
                }
            }
        }
    }

    fn fetch_once(&self, symbol: &str) -> Result<RawOptionChain, FinnhubOptionChainError> {
        let payload = self
            .agent
            .get(&self.url)
            .query("symbol", symbol)
            .query("token", &self.token)
            .call()?
            .into_json::<FinnhubOptionChainPayload>()?;

        Ok(normalize_payload(payload, self.trade_date))
    }
}

fn normalized_symbol(symbol: &str) -> Result<String, FinnhubOptionChainError> {
    let symbol = symbol.trim().to_ascii_uppercase();
    if symbol.is_empty() {
        Err(FinnhubOptionChainError::InvalidSymbol)
    } else {
        Ok(symbol)
    }
}

fn normalize_payload(payload: FinnhubOptionChainPayload, trade_date: NaiveDate) -> RawOptionChain {
    let data_len = payload
        .data
        .iter()
        .map(|exp| exp.options.calls.len() + exp.options.puts.len())
        .sum();
    let mut data = Vec::with_capacity(data_len);

    for expiration in payload.data {
        let expire = expiration.expiration_date;
        for call in expiration.options.calls {
            data.push(call.into_contract(expire, trade_date, OptionType::Call));
        }
        for put in expiration.options.puts {
            data.push(put.into_contract(expire, trade_date, OptionType::Put));
        }
    }

    RawOptionChain {
        date: trade_date,
        last_price: payload.last_trade_price,
        data,
    }
}

#[derive(Debug, Deserialize)]
struct FinnhubOptionChainPayload {
    #[serde(rename = "lastTradePrice", deserialize_with = "de_f64")]
    last_trade_price: f64,
    #[serde(default)]
    data: Vec<FinnhubExpirationData>,
}

#[derive(Debug, Deserialize)]
struct FinnhubExpirationData {
    #[serde(rename = "expirationDate")]
    expiration_date: NaiveDate,
    #[serde(default)]
    options: FinnhubOptionBuckets,
}

#[derive(Debug, Default, Deserialize)]
struct FinnhubOptionBuckets {
    #[serde(rename = "CALL", default)]
    calls: Vec<FinnhubContract>,
    #[serde(rename = "PUT", default)]
    puts: Vec<FinnhubContract>,
}

#[derive(Debug, Deserialize)]
struct FinnhubContract {
    #[serde(deserialize_with = "de_f64")]
    strike: f64,
    #[serde(rename = "lastPrice", default, deserialize_with = "de_f64")]
    last_price: f64,
    #[serde(default, deserialize_with = "de_f64")]
    bid: f64,
    #[serde(default, deserialize_with = "de_f64")]
    ask: f64,
    #[serde(default, deserialize_with = "de_f64")]
    volume: f64,
    #[serde(rename = "openInterest", default, deserialize_with = "de_f64")]
    open_interest: f64,
    #[serde(rename = "impliedVolatility", default, deserialize_with = "de_f64")]
    implied_volatility_percent: f64,
    #[serde(default, deserialize_with = "de_f64")]
    delta: f64,
    #[serde(default, deserialize_with = "de_f64")]
    gamma: f64,
    #[serde(default, deserialize_with = "de_f64")]
    theta: f64,
    #[serde(default, deserialize_with = "de_f64")]
    vega: f64,
    #[serde(default, deserialize_with = "de_f64")]
    rho: f64,
}

impl FinnhubContract {
    fn into_contract(
        self,
        expiration: NaiveDate,
        trade_date: NaiveDate,
        option_type: OptionType,
    ) -> OptionContract {
        let mark = if self.bid > 0.0 || self.ask > 0.0 {
            0.5 * (self.bid + self.ask)
        } else {
            self.last_price
        };

        OptionContract {
            expiration,
            strike: self.strike,
            option_type,
            last: self.last_price,
            mark,
            bid: self.bid,
            bid_size: 0.0,
            ask: self.ask,
            ask_size: 0.0,
            volume: self.volume,
            open_interest: self.open_interest,
            date: trade_date,
            implied_volatility: self.implied_volatility_percent / 100.0,
            delta: self.delta,
            gamma: self.gamma,
            theta: self.theta,
            vega: self.vega,
            rho: self.rho,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> NaiveDate {
        NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
    }

    #[test]
    fn normalizes_finnhub_payload_into_raw_option_chain() {
        let json = r#"
        {
            "lastTradePrice": "180.25",
            "data": [
                {
                    "expirationDate": "2025-09-20",
                    "options": {
                        "PUT": [{
                            "strike": "175",
                            "lastPrice": "2.3",
                            "bid": "2.1",
                            "ask": "2.5",
                            "volume": 11,
                            "openInterest": 90,
                            "impliedVolatility": 24.0,
                            "delta": -0.25,
                            "gamma": 0.02,
                            "theta": -0.04,
                            "vega": 0.11,
                            "rho": -0.03
                        }],
                        "CALL": [{
                            "strike": "185",
                            "lastPrice": "1.8",
                            "bid": null,
                            "ask": null,
                            "volume": "7",
                            "openInterest": "33",
                            "impliedVolatility": "20.5",
                            "delta": "0.42",
                            "gamma": "0.03",
                            "theta": "-0.05",
                            "vega": "0.12",
                            "rho": "0.02"
                        }]
                    }
                }
            ]
        }"#;

        let payload: FinnhubOptionChainPayload = serde_json::from_str(json).unwrap();
        let chain = normalize_payload(payload, d("2025-09-08"));

        assert_eq!(chain.date, d("2025-09-08"));
        assert_eq!(chain.last_price, 180.25);
        assert_eq!(chain.data.len(), 2);

        let call = &chain.data[0];
        assert_eq!(call.option_type, OptionType::Call);
        assert_eq!(call.expiration, d("2025-09-20"));
        assert_eq!(call.strike, 185.0);
        assert_eq!(call.last, 1.8);
        assert_eq!(call.mark, 1.8);
        assert_eq!(call.bid, 0.0);
        assert_eq!(call.ask, 0.0);
        assert_eq!(call.implied_volatility, 0.205);

        let put = &chain.data[1];
        assert_eq!(put.option_type, OptionType::Put);
        assert_eq!(put.strike, 175.0);
        assert_eq!(put.mark, 2.3);
        assert_eq!(put.open_interest, 90.0);
    }

    #[test]
    fn preserves_provider_order() {
        let json = r#"
        {
            "lastTradePrice": 100.0,
            "data": [
                {
                    "expirationDate": "2025-10-17",
                    "options": {
                        "CALL": [{
                            "strike": 110,
                            "lastPrice": 1.0,
                            "bid": 0.9,
                            "ask": 1.1,
                            "impliedVolatility": 18.0
                        }, {
                            "strike": 105,
                            "lastPrice": 2.0,
                            "bid": 1.9,
                            "ask": 2.1,
                            "impliedVolatility": 19.0
                        }]
                    }
                },
                {
                    "expirationDate": "2025-09-19",
                    "options": {
                        "CALL": [{
                            "strike": 100,
                            "lastPrice": 3.0,
                            "bid": 2.9,
                            "ask": 3.1,
                            "impliedVolatility": 20.0
                        }]
                    }
                }
            ]
        }"#;

        let payload: FinnhubOptionChainPayload = serde_json::from_str(json).unwrap();
        let chain = normalize_payload(payload, d("2025-09-08"));

        let strikes: Vec<f64> = chain.data.iter().map(|contract| contract.strike).collect();
        let expiries: Vec<NaiveDate> = chain
            .data
            .iter()
            .map(|contract| contract.expiration)
            .collect();

        assert_eq!(strikes, vec![110.0, 105.0, 100.0]);
        assert_eq!(
            expiries,
            vec![d("2025-10-17"), d("2025-10-17"), d("2025-09-19")]
        );
    }

    #[test]
    fn constructor_uses_explicit_trade_date() {
        let client = FinnhubOptionChainClient::new("token", d("2025-09-08"));
        assert_eq!(client.trade_date(), d("2025-09-08"));
    }

    #[test]
    fn client_is_cloneable() {
        let client = FinnhubOptionChainClient::new("token", d("2025-09-08"));
        let cloned = client.clone();
        assert_eq!(cloned.trade_date(), d("2025-09-08"));
    }
}
