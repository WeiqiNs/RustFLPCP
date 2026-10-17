use super::{
    BIT, binary_sum, bit_gadget, call_each, check_field_bits, low_bits, square_via_bit, u64_bits,
};
use crate::circuit::{Circuit, Gadget, GadgetCalls, Wire};
use crate::error::FlpError;
use ark_ff::Field;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeanVariance {
    pub len: usize,
    pub input_bits: usize,
}

impl MeanVariance {
    #[must_use]
    pub fn encode<F: Field>(&self, measurement: &[u64]) -> Vec<F> {
        let values = measurement.iter().map(|&value| F::from(value));
        let squares = measurement.iter().map(|&value| F::from(value).square());
        let value_bits = measurement
            .iter()
            .flat_map(|&value| low_bits(u64_bits(value), self.input_bits));
        values.chain(squares).chain(value_bits).collect()
    }
}

impl<F: Field> Circuit<F> for MeanVariance {
    fn input_len(&self) -> usize {
        self.len * (2 + self.input_bits)
    }

    fn gadgets(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![bit_gadget()]
    }

    fn eval<W: Wire<F>>(&self, input: &[W], calls: &mut impl GadgetCalls<W>) -> Vec<W> {
        let (values, rest) = input.split_at(self.len);
        let (squares, value_bits) = rest.split_at(self.len);

        let mut outputs = call_each(calls, BIT, value_bits);
        for (index, (value, square)) in values.iter().zip(squares).enumerate() {
            let bits = &value_bits[index * self.input_bits..][..self.input_bits];
            outputs.push(value.clone() - binary_sum(bits));
            outputs.push(square_via_bit(calls, value) - square.clone());
        }
        outputs
    }

    fn check_params(&self) -> Result<(), FlpError> {
        check_field_bits::<F>(self.input_bits.saturating_mul(2).saturating_add(1))
    }
}
