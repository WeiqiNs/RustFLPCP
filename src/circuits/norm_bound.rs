use super::{BIT, binary_sum, bit_gadget, call_each, check_field_bits, low_bits, square_via_bit};
use crate::circuit::{Circuit, Gadget, GadgetCalls, GadgetId, Wire};
use crate::error::FlpError;
use crate::gadgets::{ParallelSum, Square};
use ark_ff::{AdditiveGroup, BigInteger, Field, PrimeField};
use std::num::NonZeroUsize;

const SQUARES: GadgetId = GadgetId(1);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NormBound {
    pub len: usize,
    pub input_bits: usize,
    pub norm_bits: usize,
    pub square_chunk: Option<NonZeroUsize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SignedNormBound {
    pub len: usize,
    pub input_bits: NonZeroUsize,
    pub norm_bits: usize,
    pub square_chunk: Option<NonZeroUsize>,
}

impl NormBound {
    #[must_use]
    pub fn encode<F: Field>(&self, measurement: &[u64]) -> Vec<F> {
        let values: Vec<F::BasePrimeField> = measurement
            .iter()
            .map(|&value| F::BasePrimeField::from(value))
            .collect();
        self.encode_shifted(&values, F::BasePrimeField::ZERO)
    }

    fn encode_shifted<F: Field>(
        &self,
        values: &[F::BasePrimeField],
        offset: F::BasePrimeField,
    ) -> Vec<F> {
        let lifted = values.iter().map(|&value| F::from_base_prime_field(value));
        let value_bits = values.iter().flat_map(|&value| {
            low_bits((value + offset).into_bigint().to_bits_le(), self.input_bits)
        });
        let norm: F::BasePrimeField = values.iter().map(Field::square).sum();
        lifted
            .chain(value_bits)
            .chain(low_bits(norm.into_bigint().to_bits_le(), self.norm_bits))
            .collect()
    }

    fn eval_shifted<F: Field, W: Wire<F>>(
        &self,
        input: &[W],
        calls: &mut impl GadgetCalls<W>,
        offset: F,
    ) -> Vec<W> {
        let (values, bits) = input.split_at(self.len);
        let (value_bits, norm_bits) = bits.split_at(self.len * self.input_bits);
        let zero = W::constant(F::ZERO);

        let mut outputs = call_each(calls, BIT, bits);
        let squares = match self.square_chunk {
            None => values.iter().fold(zero.clone(), |sum, value| {
                sum + square_via_bit(calls, value)
            }),
            Some(chunk) => values.chunks(chunk.get()).fold(zero.clone(), |sum, group| {
                let mut padded = group.to_vec();
                padded.resize(chunk.get(), zero.clone());
                sum + calls.call(SQUARES, &padded)
            }),
        };
        for (index, value) in values.iter().enumerate() {
            let bits = &value_bits[index * self.input_bits..][..self.input_bits];
            outputs.push(value.clone() + W::constant(offset) - binary_sum(bits));
        }
        outputs.push(squares - binary_sum(norm_bits));
        outputs
    }
}

impl SignedNormBound {
    #[must_use]
    pub fn encode<F: Field>(&self, measurement: &[i64]) -> Vec<F> {
        let values: Vec<F::BasePrimeField> = measurement
            .iter()
            .map(|&value| F::BasePrimeField::from(value))
            .collect();
        self.unsigned().encode_shifted(&values, self.offset())
    }

    fn unsigned(&self) -> NormBound {
        NormBound {
            len: self.len,
            input_bits: self.input_bits.get(),
            norm_bits: self.norm_bits,
            square_chunk: self.square_chunk,
        }
    }

    fn offset<F: Field>(&self) -> F {
        F::from(2u64).pow([(self.input_bits.get() - 1) as u64])
    }
}

impl<F: Field> Circuit<F> for NormBound {
    fn input_len(&self) -> usize {
        self.len * (1 + self.input_bits) + self.norm_bits
    }

    fn gadgets(&self) -> Vec<Box<dyn Gadget<F>>> {
        let mut gadgets = vec![bit_gadget()];
        if let Some(chunk) = self.square_chunk {
            gadgets.push(Box::new(ParallelSum {
                inner: Square,
                count: chunk.get(),
            }));
        }
        gadgets
    }

    fn eval<W: Wire<F>>(&self, input: &[W], calls: &mut impl GadgetCalls<W>) -> Vec<W> {
        self.eval_shifted(input, calls, F::ZERO)
    }

    fn check_params(&self) -> Result<(), FlpError> {
        if let Some(chunk) = self.square_chunk.filter(|chunk| chunk.get() > self.len) {
            return Err(FlpError::ChunkTooLarge {
                chunk: chunk.get(),
                len: self.len,
            });
        }
        let len_bits = self.len.bit_width() as usize;
        check_field_bits::<F>(self.input_bits.saturating_mul(2).saturating_add(len_bits))?;
        check_field_bits::<F>(self.norm_bits)
    }
}

impl<F: Field> Circuit<F> for SignedNormBound {
    fn input_len(&self) -> usize {
        Circuit::<F>::input_len(&self.unsigned())
    }

    fn gadgets(&self) -> Vec<Box<dyn Gadget<F>>> {
        Circuit::<F>::gadgets(&self.unsigned())
    }

    fn eval<W: Wire<F>>(&self, input: &[W], calls: &mut impl GadgetCalls<W>) -> Vec<W> {
        self.unsigned().eval_shifted(input, calls, self.offset())
    }

    fn check_params(&self) -> Result<(), FlpError> {
        Circuit::<F>::check_params(&self.unsigned())
    }
}
