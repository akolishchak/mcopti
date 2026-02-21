//! Multi-leg position container

use crate::Leg;

#[derive(Default)]
pub struct Position {
    pub legs: Vec<(Leg, i64)>,
    pub premium: f64,
}

impl Position {
    pub fn push(&mut self, leg: Leg, qty: i64) {
        assert_ne!(qty, 0);
        self.legs.push((leg, qty));
        self.premium += leg.mid * qty as f64;
    }

    pub fn add(mut self, leg: Leg, qty: i64) -> Self {
        self.push(leg, qty);
        self
    }
}
