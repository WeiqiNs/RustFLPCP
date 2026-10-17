use super::call_each;
use crate::circuit::{Circuit, Gadget, GadgetCalls, GadgetId, Wire};
use crate::error::FlpError;
use crate::gadgets::SetProduct;
use ark_ff::Field;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Membership {
    pub len: usize,
    pub values: BTreeSet<u64>,
}

impl<F: Field> Circuit<F> for Membership {
    fn input_len(&self) -> usize {
        self.len
    }

    fn gadgets(&self) -> Vec<Box<dyn Gadget<F>>> {
        vec![Box::new(SetProduct {
            values: self.values.clone(),
        })]
    }

    fn eval<W: Wire<F>>(&self, input: &[W], calls: &mut impl GadgetCalls<W>) -> Vec<W> {
        call_each(calls, GadgetId(0), input)
    }

    fn check_params(&self) -> Result<(), FlpError> {
        if self.values.is_empty() {
            return Err(FlpError::EmptySet);
        }
        Ok(())
    }
}
