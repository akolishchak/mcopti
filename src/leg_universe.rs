//! Deduplicated leg universe with slices for simulation workloads.

use chrono::NaiveDate;

use crate::{Leg, OptionType, Position};

/// Deduplicated contracts and position indexes used for batch pricing.
///
/// Positions that share a contract refer to the same unique leg, allowing
/// [`crate::Simulator`] to reuse its price at each simulated path step.
pub struct LegUniverse {
    /// Unique legs sorted by expiry index, option type, and strike.
    pub legs: Vec<Leg>,
    /// Latest leg expiry, or [`NaiveDate::MIN`] when there are no legs.
    pub max_expire: NaiveDate,
    /// Whether any position contains a call.
    pub call_present: bool,
    /// Whether any position contains a put.
    pub put_present: bool,
    /// Per-position `(leg index, signed quantity)` pairs indexing [`Self::legs`].
    pub positions_idx: Vec<Vec<(usize, i64)>>,
    /// Original positions in their input order.
    pub positions: Vec<Position>,
    type_range: Vec<(usize, usize)>,
    expiry_range: Vec<(usize, usize)>,
}

/// Legs sharing an expiry index and option type.
pub struct LegUniverseTypeSlice<'a> {
    /// Option type shared by the slice.
    pub option_type: OptionType,
    /// Expiry index shared by the slice.
    pub expire_id: usize,
    /// Borrowed legs in the group, in universe order.
    pub legs: &'a [Leg],
}

/// Legs sharing an expiry index.
pub struct LegUniverseExpirySlice<'a> {
    /// Expiry index shared by the slice.
    pub expire_id: usize,
    /// Borrowed legs in the group, in universe order.
    pub legs: &'a [Leg],
}

impl LegUniverse {
    /// Deduplicate legs and build indexes while preserving position order.
    ///
    /// Legs must use consistent expiry indexes from the same option chain.
    pub fn from_positions(positions: Vec<Position>) -> Self {
        let capacity = positions.iter().map(|p| p.legs.len()).sum();
        let mut legs: Vec<Leg> = Vec::with_capacity(capacity);
        let mut positions_idx = Vec::with_capacity(positions.len());
        let mut max_expire = NaiveDate::MIN;
        let mut call_present = false;
        let mut put_present = false;

        for position in positions.iter() {
            for (leg, _) in position.legs.iter() {
                match &leg.option_type {
                    OptionType::Call => call_present = true,
                    OptionType::Put => put_present = true,
                }
                max_expire = max_expire.max(leg.expire);
                legs.push(*leg);
            }
        }

        legs.sort_unstable_by(|a, b| {
            (a.expire_id, a.option_type)
                .cmp(&(b.expire_id, b.option_type))
                .then_with(|| a.strike.total_cmp(&b.strike))
        });
        legs.dedup_by(|a, b| {
            a.expire_id == b.expire_id && a.option_type == b.option_type && a.strike == b.strike
        });

        for position in positions.iter() {
            let mut position_legs = Vec::with_capacity(position.legs.len());
            for (leg, qty) in position.legs.iter() {
                let idx = legs
                    .binary_search_by(|l| {
                        (l.expire_id, l.option_type)
                            .cmp(&(leg.expire_id, leg.option_type))
                            .then_with(|| l.strike.total_cmp(&leg.strike))
                    })
                    .expect("leg not found after universe build");
                position_legs.push((idx, *qty));
            }
            positions_idx.push(position_legs);
        }

        let mut type_range = Vec::new();
        let mut last_type_range = None;
        let mut expiry_range = Vec::new();
        let mut last_expiry_range = None;
        for (i, leg) in legs.iter().enumerate() {
            let is_new_type = match last_type_range {
                Some((exp, opt)) => exp != leg.expire_id || opt != leg.option_type,
                None => true,
            };
            if is_new_type {
                type_range.push((i, i + 1));
                last_type_range = Some((leg.expire_id, leg.option_type));
            } else if let Some(last) = type_range.last_mut() {
                last.1 = i + 1;
            }

            let is_new_expiry = match last_expiry_range {
                Some(exp) => exp != leg.expire_id,
                None => true,
            };
            if is_new_expiry {
                expiry_range.push((i, i + 1));
                last_expiry_range = Some(leg.expire_id);
            } else if let Some(last) = expiry_range.last_mut() {
                last.1 = i + 1;
            }
        }

        Self {
            legs,
            max_expire,
            call_present,
            put_present,
            positions_idx,
            positions,
            type_range,
            expiry_range,
        }
    }

    /// Iterate over groups sharing an expiry index and option type.
    pub fn type_slices(&self) -> impl Iterator<Item = LegUniverseTypeSlice<'_>> {
        self.type_range
            .iter()
            .map(|&(start, end)| LegUniverseTypeSlice {
                option_type: self.legs[start].option_type,
                expire_id: self.legs[start].expire_id,
                legs: &self.legs[start..end],
            })
    }

    /// Iterate over groups sharing an expiry index.
    pub fn expiry_slices(&self) -> impl Iterator<Item = LegUniverseExpirySlice<'_>> {
        self.expiry_range
            .iter()
            .map(|&(start, end)| LegUniverseExpirySlice {
                expire_id: self.legs[start].expire_id,
                legs: &self.legs[start..end],
            })
    }

    /// Return the number of distinct expiry index groups.
    pub fn expiries(&self) -> usize {
        self.expiry_range.len()
    }
}
