use super::{BIT, bit_gadget, call_each};
use crate::circuit::{Circuit, Gadget, GadgetCalls, Wire};
use ark_ff::Field;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OneHot {
    pub len: usize,
}

impl<F: Field> Circuit<F> for OneHot {
    fn input_len(&self) -> usize {
        self.len
    }

    fn gadgets(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![bit_gadget()]
    }

    fn eval<W: Wire<F>>(&self, input: &[W], calls: &mut impl GadgetCalls<W>) -> Vec<W> {
        let mut outputs = call_each(calls, BIT, input);
        let hot_count = input
            .iter()
            .fold(W::constant(F::ZERO), |sum, bit| sum + bit.clone());
        outputs.push(hot_count - W::constant(F::ONE));
        outputs
    }
}
