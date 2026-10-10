//! Multi-leg position container

use crate::Leg;
use core::fmt;

/// Signed option legs and their net entry premium. No contract multiplier is applied.
#[derive(Clone, Default, Debug)]
pub struct Position {
    /// Legs with signed quantities: positive for long, negative for short.
    pub legs: Vec<(Leg, i64)>,
    /// Sum of entry mid prices times quantities: positive for a debit, negative for a credit.
    pub premium: f64,
}

impl Position {
    /// Append a leg and update the net entry premium.
    ///
    /// # Panics
    ///
    /// Panics if `qty` is zero.
    pub fn push(&mut self, leg: Leg, qty: i64) {
        assert_ne!(qty, 0);
        self.legs.push((leg, qty));
        self.premium += leg.mid * qty as f64;
    }

    /// Append a leg and return the position for chained construction.
    ///
    /// # Panics
    ///
    /// Panics if `qty` is zero.
    pub fn add(mut self, leg: Leg, qty: i64) -> Self {
        self.push(leg, qty);
        self
    }

    /// Return the sign of the entry premium via [`f64::signum`].
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
