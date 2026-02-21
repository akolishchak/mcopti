//! Option leg definition and builder for validated strikes and expiries.

use chrono::NaiveDate;
use std::{error::Error, fmt};

use crate::{Context, OptionType};

#[derive(Clone, Copy)]
pub struct Leg {
    pub option_type: OptionType,
    pub strike: f64,
    pub expire: NaiveDate,
    pub expire_id: usize,
    pub mid: f64,
    pub iv: f64,
}

pub struct LegBuilder<'a> {
    context: &'a Context,
}

#[derive(Debug)]
pub enum LegBuilderError {
    WrongExpire(NaiveDate),
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
    pub fn new(context: &'a Context) -> Self {
        Self { context }
    }

    pub fn create(
        &self,
        option_type: OptionType,
        strike: f64,
        expire: NaiveDate,
    ) -> Result<Leg, LegBuilderError> {
        let option_side = match option_type {
            OptionType::Call => &self.context.option_chain.calls,
            OptionType::Put => &self.context.option_chain.puts,
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
