use crate::circuit::{GadgetId, Wire};
use ark_ff::Field;
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Variable {
    Input(usize),
    GadgetOutput { gadget: GadgetId, call: usize },
}

#[derive(Clone, Debug)]
pub(crate) struct LinearForm<F> {
    pub(crate) terms: Vec<(Variable, F)>,
    pub(crate) constant: F,
}

impl<F: Field> LinearForm<F> {
    pub(crate) fn variable(variable: Variable) -> Self {
        Self {
            terms: vec![(variable, F::ONE)],
            constant: F::ZERO,
        }
    }
}

impl<F: Field> Add for LinearForm<F> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        self.terms.extend(rhs.terms);
        self.constant += rhs.constant;
        self
    }
}

impl<F: Field> Sub for LinearForm<F> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self {
        self.terms.extend(
            rhs.terms
                .into_iter()
                .map(|(variable, coeff)| (variable, -coeff)),
        );
        self.constant -= rhs.constant;
        self
    }
}

impl<F: Field> Mul<F> for LinearForm<F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self {
        for (_, coeff) in &mut self.terms {
            *coeff *= rhs;
        }
        self.constant *= rhs;
        self
    }
}

impl<F: Field> Wire<F> for LinearForm<F> {
    fn constant(value: F) -> Self {
        Self {
            terms: Vec::new(),
            constant: value,
        }
    }
}
