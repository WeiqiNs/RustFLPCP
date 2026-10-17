use super::{BIT, binary_sum, bit_gadget, call_each, check_field_bits, low_bits, u64_bits};
use crate::circuit::{Circuit, Gadget, GadgetCalls, Wire};
use crate::error::FlpError;
use ark_ff::Field;
use std::ops::RangeInclusive;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BitRange {
    pub len: usize,
    pub bounds: RangeInclusive<u64>,
}

impl BitRange {
    #[must_use]
    pub fn encode<F: Field>(&self, measurement: &[u64]) -> Vec<F> {
        let (lower, upper) = (*self.bounds.start(), *self.bounds.end());
        let (lower_width, upper_width) = (self.gap_bits(), self.upper_gap_bits());
        let values = measurement.iter().map(|&value| F::from(value));
        let lower_gaps = measurement
            .iter()
            .flat_map(|&value| low_bits(u64_bits(value.wrapping_sub(lower)), lower_width));
        let upper_gaps = measurement
            .iter()
            .flat_map(|&value| low_bits(u64_bits(upper.wrapping_sub(value)), upper_width));
        values.chain(lower_gaps).chain(upper_gaps).collect()
    }

    fn span(&self) -> u64 {
        self.bounds.end().saturating_sub(*self.bounds.start())
    }

    fn gap_bits(&self) -> usize {
        self.span().bit_width().max(1) as usize
    }

    fn upper_gap_bits(&self) -> usize {
        let span = self.span();
        if span != 0 && span.checked_add(1).is_none_or(u64::is_power_of_two) {
            0
        } else {
            self.gap_bits()
        }
    }
}

impl<F: Field> Circuit<F> for BitRange {
    fn input_len(&self) -> usize {
        self.len * (1 + self.gap_bits() + self.upper_gap_bits())
    }

    fn gadgets(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![bit_gadget()]
    }

    fn eval<W: Wire<F>>(&self, input: &[W], calls: &mut impl GadgetCalls<W>) -> Vec<W> {
        let bits = self.gap_bits();
        let (values, gaps) = input.split_at(self.len);
        let (lower_gaps, upper_gaps) = gaps.split_at(self.len * bits);
        let lower = W::constant(F::from(*self.bounds.start()));
        let upper = W::constant(F::from(*self.bounds.end()));

        let mut outputs = call_each(calls, BIT, gaps);
        outputs.extend(
            values
                .iter()
                .zip(lower_gaps.chunks(bits))
                .map(|(value, gap)| value.clone() - lower.clone() - binary_sum(gap)),
        );
        outputs.extend(
            values
                .iter()
                .zip(upper_gaps.chunks(bits))
                .map(|(value, gap)| upper.clone() - value.clone() - binary_sum(gap)),
        );
        outputs
    }

    fn check_params(&self) -> Result<(), FlpError> {
        if self.bounds.is_empty() {
            return Err(FlpError::EmptyBounds);
        }
        check_field_bits::<F>(self.gap_bits() + 1)?;
        check_field_bits::<F>(self.bounds.end().bit_width() as usize)
    }
}
