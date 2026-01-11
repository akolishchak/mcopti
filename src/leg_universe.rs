use chrono::NaiveDate;

use crate::{Leg, OptionType, Position};

pub struct LegUniverse {
    pub legs: Vec<Leg>,
    pub max_expire: NaiveDate,
    pub call_present: bool,
    pub put_present: bool,
    pub positions_idx: Vec<Vec<(usize, i64)>>,
    pub positions: Vec<Position>,
    type_range: Vec<(usize, usize)>,
    expiry_range: Vec<(usize, usize)>,
}

pub struct LegUniverseTypeSlice<'a> {
    pub option_type: OptionType,
    pub expire_id: usize,
    pub legs: &'a [Leg],
}

pub struct LegUniverseExpirySlice<'a> {
    pub expire_id: usize,
    pub legs: &'a [Leg],
}

impl LegUniverse {
    pub fn from_positions(positions: Vec<Position>) -> Self {
        let capacity = positions
            .iter()
            .map(|p| p.legs.len())
            .sum();
        let mut legs: Vec<Leg> = Vec::with_capacity(capacity);
        let mut positions_idx = Vec::with_capacity(positions.len());
        let mut max_expire = NaiveDate::MIN;
        let mut call_present = false;
        let mut put_present = false;

        for position in positions.iter() {
            for &(leg, _) in position.legs.iter() {
                match &leg.option_type {
                    OptionType::Call => call_present = true,
                    OptionType::Put => put_present = true,
                }
                max_expire = max_expire.max(leg.expire);
                legs.push(leg);
            }
        }

        legs.sort_unstable_by(|a, b| {
            (a.expire_id, a.option_type)
                .cmp(&(b.expire_id, b.option_type))
                .then_with(|| a.strike.total_cmp(&b.strike))
        });
        legs.dedup_by(|a, b| {
            a.expire_id == b.expire_id
                && a.option_type == b.option_type
                && a.strike == b.strike
        });

        for position in positions.iter() {
            let mut position_legs = Vec::with_capacity(position.legs.len());
            for &(leg, qty) in position.legs.iter() {
                let idx = legs
                    .binary_search_by(|l| {
                        (l.expire_id, l.option_type)
                            .cmp(&(leg.expire_id, leg.option_type))
                            .then_with(|| l.strike.total_cmp(&leg.strike))
                    })
                    .expect("leg not found after universe build");
                position_legs.push((idx, qty));
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
                last_type_range = Some((leg.expire_id, leg.option_type, ));
            } else if let Some(last) = type_range.last_mut() {
                last.1 = i + 1;
            }

            let is_new_expiry = match last_expiry_range {
                Some(exp) => exp != leg.expire_id,
                None => true,
            };
            if is_new_expiry {
                expiry_range.push((i, i+1));
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

    pub fn type_slices(&self) -> impl Iterator<Item = LegUniverseTypeSlice<'_>> {
        self.type_range
            .iter()
            .map(|&(start, end)| LegUniverseTypeSlice {
                option_type: self.legs[start].option_type,
                expire_id: self.legs[start].expire_id,
                legs: &self.legs[start..end]
            })
    }

    pub fn expiry_slices(&self) -> impl Iterator<Item = LegUniverseExpirySlice<'_>> {
        self.expiry_range
            .iter()
            .map(|&(start, end)| LegUniverseExpirySlice {
                expire_id: self.legs[start].expire_id,
                legs: &self.legs[start..end]
            })
    }
}
