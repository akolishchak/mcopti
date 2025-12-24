use crate::{OptionType, RawOptionChain, raw_option_chain::OptionContract};
use chrono::NaiveDate;

const EPSILON: f64 = 1e-12;

pub struct OptionChain {
    pub spot: f64,
    pub date: NaiveDate,
    pub calls: OptionChainSide,
    pub puts: OptionChainSide,
}

impl OptionChain {
    pub fn from_raw(raw_option_chain: RawOptionChain) -> Self {
        let spot = raw_option_chain.last_price;
        let date = raw_option_chain.date;

        let n_calls = raw_option_chain.data.iter().filter(|c| matches!(c.option_type, OptionType::Call)).count();
        let n_puts = raw_option_chain.data.len() - n_calls;

        let mut calls = OptionChainSide::new(spot, date, n_calls);
        let mut puts = OptionChainSide::new(spot, date, n_puts);

        for contract in &raw_option_chain.data {
            match contract.option_type {
                OptionType::Call => &mut calls,
                OptionType::Put => &mut puts,
            }
            .push(contract);
        }

        Self {
            spot,
            date,
            calls,
            puts,
        }
    }
}



pub struct OptionChainSide {
    tau: Vec<f64>,
    expire: Vec<NaiveDate>,
    strike: Vec<f64>,
    k: Vec<f64>,
    mid: Vec<f64>,
    iv: Vec<f64>,
    spot: f64,
    date: NaiveDate,
    tau_range: Vec<(usize, usize)>,
    index: usize,
    last_tau: Option<f64>,
}

/// View over all quotes that share the same expiry/τ.
pub struct OptionSlice<'a> {
    pub expire: NaiveDate,
    pub tau: f64,
    pub strike: &'a [f64],
    pub k: &'a [f64],
    pub mid: &'a [f64],
    pub iv: &'a [f64],
}

impl OptionChainSide {
    pub(crate) fn tau(&self) -> &[f64] {
        &self.tau
    }

    fn new(spot: f64, date: NaiveDate, capacity: usize) -> Self {
        Self {
            expire: Vec::with_capacity(capacity),
            strike: Vec::with_capacity(capacity),
            tau: Vec::with_capacity(capacity),
            k: Vec::with_capacity(capacity),
            mid: Vec::with_capacity(capacity),
            iv: Vec::with_capacity(capacity),
            spot,
            date: date,
            tau_range: Vec::with_capacity(capacity),
            index: 0,
            last_tau: None,
        }
    }

    fn push(&mut self,  contract: &OptionContract) {

        let tau = (contract.expiration - self.date).num_days() as f64 / 365.0;
        let k = (contract.strike / self.spot).ln();
        let iv = contract.implied_volatility;
        let mid = (contract.ask + contract.bid) * 0.5;

        let is_new_tau = self.last_tau.map_or(true, |last_tau| (tau - last_tau).abs() > EPSILON);

        if is_new_tau {
            self.expire.push(contract.expiration);
            self.tau.push(tau);
            self.tau_range.push((self.index, self.index + 1));
            self.last_tau = Some(tau);
        } else if let Some(last) = self.tau_range.last_mut() {
            last.1 = self.index + 1;
        }
        self.strike.push(contract.strike);
        self.k.push(k);
        self.mid.push(mid);
        self.iv.push(iv);
        self.index += 1;
    }

    /// Iterate over each maturity, yielding slices into the per-quote arrays.
    pub fn slices(&self) -> impl Iterator<Item = OptionSlice<'_>> {
        self.tau_range
            .iter()
            .enumerate()
            .map(move |(i, &(start, end))| OptionSlice {
                expire: self.expire[i],
                tau: self.tau[i],
                strike: &self.strike[start..end],
                k: &self.k[start..end],
                mid: &self.mid[start..end],
                iv: &self.iv[start..end],
            })
    }

}



#[cfg(test)]
mod tests {
    use super::*;
}
