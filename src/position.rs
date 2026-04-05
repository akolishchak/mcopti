//! Multi-leg position container

use crate::Leg;
use core::fmt;

#[derive(Clone, Default, Debug)]
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

    pub fn side(&self) -> f64 {
        // TODO: consider alternative approaches to detect side
        self.premium.signum()
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, (leg, qty)) in self.legs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{qty}*[{leg}]")?;
        }

        Ok(())
    }
}
