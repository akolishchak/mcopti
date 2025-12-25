use crate::{Leg, OptionType, Position};

pub struct LegUniverse {
    pub legs: Vec<Leg>,
    pub range: Vec<(usize, usize)>, 
}

pub struct LegUniverseSlice<'a> {
    pub option_type: OptionType,
    pub expire_id: usize,
    pub legs: &'a [Leg],
}

impl LegUniverse {
    pub fn from_positions(positions: &[Position]) -> Self {
        let capacity = positions
            .iter()
            .map(|p| p.legs.len())
            .sum();
        let mut legs: Vec<Leg> = Vec::with_capacity(capacity);

        for position in positions {
            for &(leg, _) in position.legs.iter() {
                let exists = legs.iter().any(|existing| {
                    existing.option_type == leg.option_type
                    && existing.expire_id == leg.expire_id
                    && existing.strike == leg.strike
                });

                if !exists {
                    legs.push(leg);
                }
            }
        }

        legs.sort_by_key(|leg| (leg.option_type, leg.expire_id));

        let mut range = Vec::new();
        let mut last_group = None;
        for (i, leg) in legs.iter().enumerate() {
            let is_new_group = last_group.map_or(true, |(opt, exp)| {
                opt != leg.option_type || exp != leg.expire_id
            });
            if is_new_group {
                range.push((i, i + 1));
                last_group = Some((leg.option_type, leg.expire_id));
            } else if let Some(last) = range.last_mut() {
                last.1 = i + 1;
            }
        }

        Self { 
            legs,
            range,
        }
    }

    pub fn slices(&self) -> impl Iterator<Item = LegUniverseSlice<'_>> {
        self.range
            .iter()
            .map(|&(start, end)| LegUniverseSlice {
                option_type: self.legs[start].option_type,
                expire_id: self.legs[start].expire_id,
                legs: &self.legs[start..end]
            })
    }
}
