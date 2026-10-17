use super::{bit_width, call_each, check_field_bits};
use crate::circuit::{Circuit, Gadget, GadgetCalls, GadgetId, Wire};
use crate::error::FlpError;
use crate::gadgets::RangeProduct;
use ark_ff::Field;
use std::ops::RangeInclusive;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Range {
    pub len: usize,
    pub bounds: RangeInclusive<u64>,
}

impl Range {
    #[must_use]
    pub fn binary(len: usize) -> Self {
        Self { len, bounds: 0..=1 }
    }
}

impl<F: Field> Circuit<F> for Range {
    fn input_len(&self) -> usize {
        self.len
    }

    fn gadgets(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![Box::new(RangeProduct {
            bounds: self.bounds.clone(),
        })]
    }

    fn eval<W: Wire<F>>(&self, input: &[W], calls: &mut impl GadgetCalls<W>) -> Vec<W> {
        call_each(calls, GadgetId(0), input)
    }

    fn check_params(&self) -> Result<(), FlpError> {
        if self.bounds.is_empty() {
            return Err(FlpError::EmptyBounds);
        }
        check_field_bits::<F>(bit_width(*self.bounds.end()))
    }
}
