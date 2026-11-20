use super::{Flp, GadgetPlan};
use crate::circuit::{Circuit, GadgetCalls, GadgetId};
use crate::error::FlpError;
use ark_ff::{FftField, Zero};
use ark_poly::EvaluationDomain;
use ark_std::rand::Rng;
use ark_std::{cfg_into_iter, cfg_iter};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

struct ProverCalls<'a, F: FftField> {
    plans: &'a [GadgetPlan<F>],
    wires: Vec<Vec<Vec<F>>>,
    counts: Vec<usize>,
}

impl<F: FftField> GadgetCalls<F> for ProverCalls<'_, F> {
    fn call(&mut self, gadget: GadgetId, inputs: &[F]) -> F {
        let count = &mut self.counts[gadget.0];
        *count += 1;
        for (wire, &input) in self.wires[gadget.0].iter_mut().zip(inputs) {
            wire[*count] = input;
        }
        self.plans[gadget.0].gadget.eval(inputs)
    }
}

impl<F: FftField, C: Circuit<F>> Flp<F, C> {
    pub fn prove(&self, input: &[F], rng: &mut impl Rng) -> Result<Vec<F>, FlpError> {
        if input.len() != self.input_len {
            return Err(FlpError::InputLength {
                expected: self.input_len,
                actual: input.len(),
            });
        }

        let wires = self
            .plans
            .iter()
            .map(|plan| {
                (0..plan.gadget.arity())
                    .map(|_| {
                        let mut wire = vec![F::ZERO; plan.wire_domain.size()];
                        wire[0] = F::rand(rng);
                        wire
                    })
                    .collect()
            })
            .collect();
        let mut calls = ProverCalls {
            plans: &self.plans,
            wires,
            counts: vec![0; self.plans.len()],
        };
        self.circuit.eval(input, &mut calls);

        let mut vector = Vec::with_capacity(self.vector_len);
        vector.extend_from_slice(input);
        for (index, (plan, wires)) in self.plans.iter().zip(&calls.wires).enumerate() {
            vector.extend(wires.iter().map(|wire| wire[0]));
            vector.extend(gadget_poly(plan, wires).ok_or(FlpError::DegreeMismatch {
                gadget: GadgetId(index),
            })?);
        }
        Ok(vector)
    }
}

fn gadget_poly<F: FftField>(plan: &GadgetPlan<F>, wires: &[Vec<F>]) -> Option<Vec<F>> {
    let wire_evals: Vec<Vec<F>> = cfg_iter!(wires)
        .map(|wire| plan.poly_domain.fft(&plan.wire_domain.ifft(wire)))
        .collect();
    let gadget_evals: Vec<F> = cfg_into_iter!(0..plan.poly_domain.size())
        .map(|index| {
            let point: Vec<F> = wire_evals.iter().map(|evals| evals[index]).collect();
            plan.gadget.eval(&point)
        })
        .collect();
    let mut coeffs = plan.poly_domain.ifft(&gadget_evals);
    if !coeffs[plan.poly_len..].iter().all(Zero::is_zero) {
        return None;
    }
    coeffs.truncate(plan.poly_len);
    Some(coeffs)
}
