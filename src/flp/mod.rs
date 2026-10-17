mod prove;
mod query;

pub use query::{LinearQuery, Query};

use crate::circuit::{Circuit, Gadget, GadgetCalls, GadgetId};
use crate::error::FlpError;
use crate::linear_form::{LinearForm, Variable};
use ark_ff::{FftField, Field};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use std::collections::BTreeMap;

type CallInputs<F> = Vec<LinearForm<F>>;

#[derive(Debug)]
pub struct Flp<F: FftField, C> {
    circuit: C,
    input_len: usize,
    plans: Vec<GadgetPlan<F>>,
    outputs: Vec<LinearForm<F>>,
    vector_len: usize,
}

#[derive(Debug)]
struct GadgetPlan<F: FftField> {
    gadget: Box<dyn Gadget<F>>,
    calls: Vec<CallInputs<F>>,
    wire_domain: Radix2EvaluationDomain<F>,
    poly_domain: Radix2EvaluationDomain<F>,
    seed_offset: usize,
    poly_offset: usize,
    poly_len: usize,
}

struct SymbolicCalls<F> {
    calls: BTreeMap<GadgetId, Vec<CallInputs<F>>>,
}

impl<F: Field> GadgetCalls<LinearForm<F>> for SymbolicCalls<F> {
    fn call(&mut self, gadget: GadgetId, inputs: &[LinearForm<F>]) -> LinearForm<F> {
        let calls = self.calls.entry(gadget).or_default();
        let call = calls.len();
        calls.push(inputs.to_vec());
        LinearForm::variable(Variable::GadgetOutput { gadget, call })
    }
}

impl<F: FftField, C: Circuit<F>> Flp<F, C> {
    pub fn new(circuit: C) -> Result<Self, FlpError> {
        circuit.check_params()?;
        let input_len = circuit.input_len();
        let input: Vec<LinearForm<F>> = (0..input_len)
            .map(|index| LinearForm::variable(Variable::Input(index)))
            .collect();
        let mut symbolic = SymbolicCalls {
            calls: BTreeMap::new(),
        };
        let outputs = circuit.eval(&input, &mut symbolic);
        if outputs.is_empty() {
            return Err(FlpError::NoOutputs);
        }

        let gadgets = circuit.gadgets();
        if let Some(&gadget) = symbolic.calls.keys().find(|id| id.0 >= gadgets.len()) {
            return Err(FlpError::InvalidGadget { gadget });
        }
        let mut plans = Vec::with_capacity(gadgets.len());
        let mut offset = input_len;
        for (index, gadget) in gadgets.into_iter().enumerate() {
            let id = GadgetId(index);
            let calls = symbolic.calls.remove(&id).unwrap_or_default();
            let plan = GadgetPlan::new(id, gadget, calls, offset)?;
            offset = plan.poly_offset + plan.poly_len;
            plans.push(plan);
        }

        Ok(Self {
            circuit,
            input_len,
            plans,
            outputs,
            vector_len: offset,
        })
    }

    pub fn vector_len(&self) -> usize {
        self.vector_len
    }

    pub fn proof_len(&self) -> usize {
        self.vector_len - self.input_len
    }

    pub fn query_count(&self) -> usize {
        self.plans
            .iter()
            .map(|plan| plan.gadget.arity())
            .sum::<usize>()
            + 1
    }

    pub fn decide(&self, query: &Query<F>, answers: &[F]) -> bool {
        if answers.len() != self.query_count() || query.gadget_weights.len() != self.plans.len() {
            return false;
        }
        let (mut wires, combined) = answers.split_at(answers.len() - 1);
        let mut gadget_sum = F::ZERO;
        for (plan, &weight) in self.plans.iter().zip(&query.gadget_weights) {
            let (inputs, tail) = wires.split_at(plan.gadget.arity());
            gadget_sum += weight * plan.gadget.eval(inputs);
            wires = tail;
        }
        combined == [gadget_sum]
    }

    pub fn verify(&self, query: &Query<F>, vector: &[F]) -> bool {
        let fits = |len: usize| len == self.vector_len;
        fits(vector.len())
            && query
                .linear_queries()
                .iter()
                .all(|linear| fits(linear.coeffs.len()))
            && self.decide(query, &query.answer(vector))
    }
}

impl<F: FftField> GadgetPlan<F> {
    fn new(
        id: GadgetId,
        gadget: Box<dyn Gadget<F>>,
        calls: Vec<CallInputs<F>>,
        seed_offset: usize,
    ) -> Result<Self, FlpError> {
        let arity = gadget.arity();
        if arity == 0 {
            return Err(FlpError::ZeroArity { gadget: id });
        }
        if gadget.degree() == 0 {
            return Err(FlpError::ZeroDegree { gadget: id });
        }
        gadget.check_params()?;
        if calls.is_empty() {
            return Err(FlpError::UnusedGadget { gadget: id });
        }
        if let Some(call) = calls.iter().position(|inputs| inputs.len() != arity) {
            return Err(FlpError::ArityMismatch {
                gadget: id,
                call,
                expected: arity,
                actual: calls[call].len(),
            });
        }

        let domains = || {
            let wire_size = (calls.len() + 1).next_power_of_two();
            let poly_len = gadget.degree().checked_mul(wire_size - 1)?.checked_add(1)?;
            let poly_size = poly_len.checked_next_power_of_two()?;
            Some((
                Radix2EvaluationDomain::new(wire_size)?,
                Radix2EvaluationDomain::new(poly_size)?,
                poly_len,
            ))
        };
        let (wire_domain, poly_domain, poly_len) =
            domains().ok_or(FlpError::DomainTooLarge { gadget: id })?;
        let poly_offset = seed_offset + arity;
        Ok(Self {
            gadget,
            calls,
            wire_domain,
            poly_domain,
            seed_offset,
            poly_offset,
            poly_len,
        })
    }
}
