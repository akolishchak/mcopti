//! Option leg definition and builder for validated strikes and expiries.

use chrono::NaiveDate;
use std::{error::Error, fmt};

use crate::{Context, OptionChain, OptionType};

/// One option contract with its entry mid price and implied volatility.
#[derive(Clone, Copy, Debug)]
pub struct Leg {
    /// Call or put.
    pub option_type: OptionType,
    /// Strike price, in the same units as the underlying spot.
    pub strike: f64,
    /// Contract expiration date.
    pub expire: NaiveDate,
    /// Expiry index within the corresponding call or put chain.
    pub expire_id: usize,
    /// Entry mid price, `(bid + ask) / 2`, without a contract multiplier.
    pub mid: f64,
    /// Annual implied volatility as a decimal, such as `0.20` for 20%.
    pub iv: f64,
}

/// Build legs by looking up exact strikes and expiries in an option chain.
pub struct LegBuilder<'a> {
    option_chain: &'a OptionChain,
}

/// A requested contract is absent from the option chain.
#[derive(Debug)]
pub enum LegBuilderError {
    /// The requested expiry is absent from the selected option side.
    WrongExpire(NaiveDate),
    /// The requested strike is absent at the selected expiry.
    WrongStrike(f64),
}

impl fmt::Display for LegBuilderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongExpire(date) => write!(f, "invalid expiration date: {date}"),
            Self::WrongStrike(strike) => write!(f, "invalid strike: {strike}"),
        }
    }
}

impl Error for LegBuilderError {}

impl<'a> LegBuilder<'a> {
    /// Borrow the option chain from a simulation context.
    pub fn new(context: &'a Context) -> Self {
        Self::from_option_chain(&context.option_chain)
    }

    /// Borrow an option chain directly.
    pub fn from_option_chain(option_chain: &'a OptionChain) -> Self {
        Self { option_chain }
    }

    /// Look up an exact strike and expiry and copy its quote into a leg.
    ///
    /// # Errors
    ///
    /// Returns [`LegBuilderError`] if the selected side lacks the expiry or strike.
    pub fn create(
        &self,
        option_type: OptionType,
        strike: f64,
        expire: NaiveDate,
    ) -> Result<Leg, LegBuilderError> {
        let option_side = match option_type {
            OptionType::Call => &self.option_chain.calls,
            OptionType::Put => &self.option_chain.puts,
        };

        let Some(expire_id) = option_side.expire_id(expire) else {
            return Err(LegBuilderError::WrongExpire(expire));
        };

        let Some((mid, iv)) = option_side.strike(strike, expire_id) else {
            return Err(LegBuilderError::WrongStrike(strike));
        };

        Ok(Leg {
            option_type,
            strike,
            expire,
            expire_id,
            mid,
            iv,
        })
    }
}

impl fmt::Display for Leg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{:?}, {:.0}, {}",
            self.option_type, self.strike, self.expire
        )
    }
}
