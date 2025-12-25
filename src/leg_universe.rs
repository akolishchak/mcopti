use crate::{Leg, Position, OptionType};

pub struct LegUniverse {
    pub legs: Vec<Leg>,
    pub expire_group: Vec<ExpireGroup>, 
}

impl LegUniverse {
    pub fn from_positions(positions: &[Position]) -> Self {
        let mut legs: Vec<Leg> = Vec::with_capacity(positions.len());
        let mut expire_group = Vec::new();

        for position in positions {
            for &(leg, _) in position.legs.iter() {
                let exists = legs.iter().any(|existing| {
                    existing.option_type == leg.option_type && existing.strike == leg.strike
                });

                if !exists {
                    legs.push(leg);
                    // update expire_group
                }
            }
        }

        Self { 
            legs,
            expire_group,
        }
    }
}

pub struct ExpireGroup {
    expiry_id: usize,
    option_type: OptionType,
    leg_indices: Vec<usize>,
}
