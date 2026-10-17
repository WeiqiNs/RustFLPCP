mod bit_range;
mod mean_variance;
mod membership;
mod norm_bound;
mod one_hot;
mod range;

pub use bit_range::BitRange;
pub use mean_variance::MeanVariance;
pub use membership::Membership;
pub use norm_bound::{NormBound, SignedNormBound};
pub use one_hot::OneHot;
pub use range::Range;

use crate::circuit::{Gadget, GadgetCalls, GadgetId, Wire};
use crate::error::FlpError;
use crate::gadgets::RangeProduct;
use ark_ff::{Field, PrimeField};
use std::ops::Add;
use std::{iter, slice};

const BIT: GadgetId = GadgetId(0);

fn bit_gadget<F: Field>() -> Box<dyn Gadget<F>> {
    Box::new(RangeProduct { bounds: 0..=1 })
}

fn call_each<W>(calls: &mut impl GadgetCalls<W>, gadget: GadgetId, wires: &[W]) -> Vec<W> {
    wires
        .iter()
        .map(|wire| calls.call(gadget, slice::from_ref(wire)))
        .collect()
}

fn square_via_bit<W: Clone + Add<Output = W>>(calls: &mut impl GadgetCalls<W>, value: &W) -> W {
    calls.call(BIT, slice::from_ref(value)) + value.clone()
}

fn binary_sum<F: Field, W: Wire<F>>(bits: &[W]) -> W {
    let mut power = F::ONE;
    bits.iter().fold(W::constant(F::ZERO), |sum, bit| {
        let term = bit.clone() * power;
        power.double_in_place();
        sum + term
    })
}

fn low_bits<F: Field>(
    bits_le: impl IntoIterator<Item = bool>,
    width: usize,
) -> impl Iterator<Item = F> {
    bits_le
        .into_iter()
        .chain(iter::repeat(false))
        .take(width)
        .map(F::from)
}

fn u64_bits(value: u64) -> impl Iterator<Item = bool> {
    (0..u64::BITS).map(move |index| (value >> index) & 1 == 1)
}

fn check_field_bits<F: Field>(bits: usize) -> Result<(), FlpError> {
    let limit = F::BasePrimeField::MODULUS_BIT_SIZE as usize - 1;
    if bits > limit {
        return Err(FlpError::ExceedsField { bits, limit });
    }
    Ok(())
}
