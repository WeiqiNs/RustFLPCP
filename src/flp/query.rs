use super::{Flp, GadgetPlan};
use crate::circuit::Circuit;
use crate::linear_form::{LinearForm, Variable};
use ark_ff::{FftField, Field, Zero};
use ark_poly::EvaluationDomain;
use ark_std::rand::Rng;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LinearQuery<F> {
    pub coeffs: Vec<F>,
    pub constant: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Query<F> {
    linear: Vec<LinearQuery<F>>,
    pub(super) gadget_weights: Vec<F>,
}

struct QueryCoins<F> {
    point: F,
    gadget_weights: Vec<F>,
    output_weights: Vec<F>,
}

struct QueryAccumulator<'a, F: FftField> {
    plans: &'a [GadgetPlan<F>],
    coeffs: Vec<F>,
    node_weights: Vec<Vec<F>>,
    constant: F,
}

impl<F: Field> Query<F> {
    #[must_use]
    pub fn linear_queries(&self) -> &[LinearQuery<F>] {
        &self.linear
    }

    pub(crate) fn answer(&self, vector: &[F]) -> Vec<F> {
        self.linear
            .iter()
            .map(|query| {
                query
                    .coeffs
                    .iter()
                    .zip(vector)
                    .map(|(&coeff, &value)| coeff * value)
                    .sum::<F>()
                    + query.constant
            })
            .collect()
    }
}

impl<'a, F: FftField> QueryAccumulator<'a, F> {
    fn new(plans: &'a [GadgetPlan<F>], vector_len: usize) -> Self {
        Self {
            plans,
            coeffs: vec![F::ZERO; vector_len],
            node_weights: plans
                .iter()
                .map(|plan| vec![F::ZERO; plan.wire_domain.size()])
                .collect(),
            constant: F::ZERO,
        }
    }

    fn add_form(&mut self, form: &LinearForm<F>, scale: F) {
        for &(variable, coeff) in &form.terms {
            match variable {
                Variable::Input(index) => self.coeffs[index] += coeff * scale,
                Variable::GadgetOutput { gadget, call } => {
                    self.node_weights[gadget.0][call + 1] += coeff * scale;
                }
            }
        }
        self.constant += form.constant * scale;
    }

    fn finish(self) -> LinearQuery<F> {
        let Self {
            plans,
            mut coeffs,
            node_weights,
            constant,
        } = self;
        for (plan, weights) in plans.iter().zip(&node_weights) {
            if weights.iter().all(Zero::is_zero) {
                continue;
            }
            let evals = plan.wire_domain.fft(weights);
            let poly_coeffs = &mut coeffs[plan.poly_offset..][..plan.poly_len];
            for (coeff, eval) in poly_coeffs.iter_mut().zip(evals.iter().cycle()) {
                *coeff += eval;
            }
        }
        LinearQuery { coeffs, constant }
    }
}

impl<F: FftField, C: Circuit<F>> Flp<F, C> {
    pub fn query(&self, rng: &mut impl Rng) -> Query<F> {
        self.build_query(self.sample_coins(rng))
    }

    fn sample_coins(&self, rng: &mut impl Rng) -> QueryCoins<F> {
        let largest_wire_domain = self
            .plans
            .iter()
            .map(|plan| plan.wire_domain)
            .max_by_key(EvaluationDomain::size);
        let point =
            largest_wire_domain.map_or(F::ZERO, |domain| domain.sample_element_outside_domain(rng));
        let gadget_weights = self.plans.iter().map(|_| F::rand(rng)).collect();
        let output_weights = self.outputs.iter().map(|_| F::rand(rng)).collect();
        QueryCoins {
            point,
            gadget_weights,
            output_weights,
        }
    }

    fn build_query(&self, coins: QueryCoins<F>) -> Query<F> {
        let mut linear = Vec::with_capacity(self.query_count());
        for plan in &self.plans {
            let lagrange = plan
                .wire_domain
                .evaluate_all_lagrange_coefficients(coins.point);
            for slot in 0..plan.gadget.arity() {
                let mut accumulator = QueryAccumulator::new(&self.plans, self.vector_len);
                accumulator.coeffs[plan.seed_offset + slot] += lagrange[0];
                for (call, &weight) in plan.calls.iter().zip(&lagrange[1..]) {
                    accumulator.add_form(&call[slot], weight);
                }
                linear.push(accumulator.finish());
            }
        }

        let mut accumulator = QueryAccumulator::new(&self.plans, self.vector_len);
        for (plan, &weight) in self.plans.iter().zip(&coins.gadget_weights) {
            let mut term = weight;
            for coeff in &mut accumulator.coeffs[plan.poly_offset..][..plan.poly_len] {
                *coeff += term;
                term *= coins.point;
            }
        }
        for (output, &weight) in self.outputs.iter().zip(&coins.output_weights) {
            accumulator.add_form(output, weight);
        }
        linear.push(accumulator.finish());
        Query {
            linear,
            gadget_weights: coins.gadget_weights,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::QueryCoins;
    use crate::circuit::{Circuit, Gadget, GadgetCalls, GadgetId, Wire};
    use crate::circuits::{BitRange, MeanVariance, NormBound, Range, SignedNormBound};
    use crate::flp::{Flp, GadgetPlan};
    use crate::gadgets::RangeProduct;
    use ark_bls12_381::Fr;
    use ark_ff::{AdditiveGroup, UniformRand};
    use ark_poly::univariate::DensePolynomial;
    use ark_poly::{DenseUVPolynomial, EvaluationDomain, Polynomial};
    use std::num::NonZeroUsize;

    struct Composite;

    impl Circuit<Fr> for Composite {
        fn input_len(&self) -> usize {
            4
        }

        fn gadgets(&self) -> Vec<Box<dyn Gadget<Fr>>> {
            vec![
                Box::new(RangeProduct { bounds: 0..=1 }),
                Box::new(RangeProduct { bounds: 0..=2 }),
            ]
        }

        fn eval<W: Wire<Fr>>(&self, input: &[W], calls: &mut impl GadgetCalls<W>) -> Vec<W> {
            let a = calls.call(GadgetId(0), &[input[0].clone()]);
            let b = calls.call(GadgetId(1), &[a + input[1].clone()]);
            let c = calls.call(
                GadgetId(0),
                &[b.clone() * Fr::from(2u64) + W::constant(Fr::from(5u64))],
            );
            vec![c - input[2].clone(), b + input[3].clone() * Fr::from(3u64)]
        }
    }

    struct ReferenceCalls<'a> {
        plans: &'a [GadgetPlan<Fr>],
        vector: &'a [Fr],
        recorded: Vec<Vec<Vec<Fr>>>,
    }

    impl GadgetCalls<Fr> for ReferenceCalls<'_> {
        fn call(&mut self, gadget: GadgetId, inputs: &[Fr]) -> Fr {
            let plan = &self.plans[gadget.0];
            let calls = &mut self.recorded[gadget.0];
            calls.push(inputs.to_vec());
            let node = plan.wire_domain.element(calls.len());
            gadget_poly(plan, self.vector).evaluate(&node)
        }
    }

    fn gadget_poly(plan: &GadgetPlan<Fr>, vector: &[Fr]) -> DensePolynomial<Fr> {
        DensePolynomial::from_coefficients_slice(&vector[plan.poly_offset..][..plan.poly_len])
    }

    fn reference_answers<C: Circuit<Fr>>(
        flp: &Flp<Fr, C>,
        coins: &QueryCoins<Fr>,
        vector: &[Fr],
    ) -> Vec<Fr> {
        let mut calls = ReferenceCalls {
            plans: &flp.plans,
            vector,
            recorded: vec![Vec::new(); flp.plans.len()],
        };
        let outputs = flp.circuit.eval(&vector[..flp.input_len], &mut calls);

        let mut answers = Vec::new();
        for (plan, recorded) in flp.plans.iter().zip(&calls.recorded) {
            for slot in 0..plan.gadget.arity() {
                let mut nodes = vec![Fr::ZERO; plan.wire_domain.size()];
                nodes[0] = vector[plan.seed_offset + slot];
                for (call, inputs) in recorded.iter().enumerate() {
                    nodes[call + 1] = inputs[slot];
                }
                let wire = DensePolynomial::from_coefficients_vec(plan.wire_domain.ifft(&nodes));
                answers.push(wire.evaluate(&coins.point));
            }
        }
        let gadget_sum: Fr = flp
            .plans
            .iter()
            .zip(&coins.gadget_weights)
            .map(|(plan, weight)| gadget_poly(plan, vector).evaluate(&coins.point) * weight)
            .sum();
        let output_sum: Fr = outputs
            .iter()
            .zip(&coins.output_weights)
            .map(|(output, weight)| *output * weight)
            .sum();
        answers.push(gadget_sum + output_sum);
        answers
    }

    fn assert_matches_reference<C: Circuit<Fr>>(circuit: C) {
        let mut rng = ark_std::test_rng();
        let flp = Flp::<Fr, C>::new(circuit).unwrap();
        let vector: Vec<Fr> = (0..flp.vector_len).map(|_| Fr::rand(&mut rng)).collect();
        let coins = flp.sample_coins(&mut rng);
        let expected = reference_answers(&flp, &coins, &vector);
        assert_eq!(flp.build_query(coins).answer(&vector), expected);
    }

    #[test]
    fn query_answers_match_reference_evaluation() {
        assert_matches_reference(Range {
            len: 3,
            bounds: 2..=6,
        });
        assert_matches_reference(BitRange {
            len: 3,
            bounds: 10..=20,
        });
        for square_chunk in [None, NonZeroUsize::new(3)] {
            assert_matches_reference(NormBound {
                len: 4,
                input_bits: 3,
                norm_bits: 7,
                square_chunk,
            });
            assert_matches_reference(SignedNormBound {
                len: 4,
                input_bits: NonZeroUsize::new(3).unwrap(),
                norm_bits: 7,
                square_chunk,
            });
        }
        assert_matches_reference(MeanVariance {
            len: 3,
            input_bits: 4,
        });
        assert_matches_reference(Composite);
    }
}
