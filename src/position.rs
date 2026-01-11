use crate::Leg;

pub struct Position {
    pub legs: Vec<(Leg, i64)>,
    pub premium: f64,
}

impl Position {
    pub fn new() -> Self {
        Self {
            legs: Vec::new(),
            premium: 0.0,
        }
    }

    pub fn push(&mut self, leg: Leg, qty: i64) {
        assert_ne!(qty, 0);
        self.legs.push((leg, qty));
        self.premium += leg.mid;
    }

    pub fn add(mut self, leg: Leg, qty: i64) -> Self {
        self.push(leg, qty);
        self
    }
}
