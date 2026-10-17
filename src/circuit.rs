use crate::error::FlpError;
use ark_ff::Field;
use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GadgetId(pub usize);

pub trait Wire<F>: Clone + Add<Output = Self> + Sub<Output = Self> + Mul<F, Output = Self> {
    fn constant(value: F) -> Self;
}

impl<F: Field> Wire<F> for F {
    fn constant(value: F) -> Self {
        value
    }
}

pub trait Gadget<F>: Debug + Send + Sync {
    fn arity(&self) -> usize;
    fn degree(&self) -> usize;
    fn eval(&self, inputs: &[F]) -> F;
}

pub trait GadgetCalls<W> {
    fn call(&mut self, gadget: GadgetId, inputs: &[W]) -> W;
}

pub trait Circuit<F: Field> {
    fn input_len(&self) -> usize;
    fn gadgets(&self) -> Vec<Box<dyn Gadget<F>>>;
    fn eval<W: Wire<F>>(&self, input: &[W], calls: &mut impl GadgetCalls<W>) -> Vec<W>;
    fn check_params(&self) -> Result<(), FlpError> {
        Ok(())
    }
}
