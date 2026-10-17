use crate::circuit::Gadget;
use ark_ff::Field;
use std::collections::BTreeSet;
use std::ops::RangeInclusive;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RangeProduct {
    pub bounds: RangeInclusive<u64>,
}

impl<F: Field> Gadget<F> for RangeProduct {
    fn arity(&self) -> usize {
        1
    }

    fn degree(&self) -> usize {
        if self.bounds.is_empty() {
            return 0;
        }
        usize::try_from(self.bounds.end() - self.bounds.start())
            .unwrap_or(usize::MAX)
            .saturating_add(1)
    }

    fn eval(&self, inputs: &[F]) -> F {
        vanishing_product(inputs[0], self.bounds.clone())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SetProduct {
    pub values: BTreeSet<u64>,
}

impl<F: Field> Gadget<F> for SetProduct {
    fn arity(&self) -> usize {
        1
    }

    fn degree(&self) -> usize {
        self.values.len()
    }

    fn eval(&self, inputs: &[F]) -> F {
        vanishing_product(inputs[0], self.values.iter().copied())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Square;

impl<F: Field> Gadget<F> for Square {
    fn arity(&self) -> usize {
        1
    }

    fn degree(&self) -> usize {
        2
    }

    fn eval(&self, inputs: &[F]) -> F {
        inputs[0].square()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParallelSum<G> {
    pub inner: G,
    pub count: usize,
}

impl<F: Field, G: Gadget<F>> Gadget<F> for ParallelSum<G> {
    fn arity(&self) -> usize {
        self.count.saturating_mul(self.inner.arity())
    }

    fn degree(&self) -> usize {
        self.inner.degree()
    }

    fn eval(&self, inputs: &[F]) -> F {
        inputs
            .chunks(self.inner.arity())
            .map(|chunk| self.inner.eval(chunk))
            .sum()
    }
}

fn vanishing_product<F: Field>(x: F, roots: impl IntoIterator<Item = u64>) -> F {
    roots
        .into_iter()
        .fold(F::ONE, |product, root| product * (x - F::from(root)))
}
