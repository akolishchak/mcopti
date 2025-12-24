use crate::OptionType;


pub struct Leg {
    option_type: OptionType,
    strike: f64,
    expire_id: usize,
}

