use ark_bls12_381::Fr;
use ark_ff::{AdditiveGroup, Field};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use rust_flpcp::{
    BitRange, Circuit, Flp, FlpError, Gadget, GadgetCalls, GadgetId, MeanVariance, Membership,
    NormBound, ParallelSum, Query, Range, RangeProduct, SignedNormBound, Square, Wire,
};
use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::ops::RangeInclusive;
use std::slice;

type Gadgets = Vec<Box<dyn Gadget<Fr>>>;
type MalformedCase = (fn() -> Gadgets, Vec<(usize, usize)>, Option<FlpError>);

struct ScriptedCircuit {
    gadgets: fn() -> Gadgets,
    script: Vec<(usize, usize)>,
}

impl Circuit<Fr> for ScriptedCircuit {
    fn input_len(&self) -> usize {
        2
    }

    fn gadgets(&self) -> Gadgets {
        (self.gadgets)()
    }

    fn eval<W: Wire<Fr>>(&self, input: &[W], calls: &mut impl GadgetCalls<W>) -> Vec<W> {
        self.script
            .iter()
            .map(|&(gadget, arity)| calls.call(GadgetId(gadget), &input[..arity]))
            .collect()
    }
}

struct BoundedSquareSum;

impl Circuit<Fr> for BoundedSquareSum {
    fn input_len(&self) -> usize {
        3
    }

    fn gadgets(&self) -> Gadgets {
        vec![
            Box::new(RangeProduct { bounds: 0..=3 }),
            Box::new(ParallelSum {
                inner: Square,
                count: 2,
            }),
        ]
    }

    fn eval<W: Wire<Fr>>(&self, input: &[W], calls: &mut impl GadgetCalls<W>) -> Vec<W> {
        let [x, y, claimed] = input else {
            unreachable!()
        };
        vec![
            calls.call(GadgetId(0), slice::from_ref(x)),
            calls.call(GadgetId(0), slice::from_ref(y)),
            calls.call(GadgetId(1), &[x.clone(), y.clone()]) - claimed.clone(),
        ]
    }
}

struct SplitBitChecks;

impl Circuit<Fr> for SplitBitChecks {
    fn input_len(&self) -> usize {
        2
    }

    fn gadgets(&self) -> Gadgets {
        vec![bit(), bit()]
    }

    fn eval<W: Wire<Fr>>(&self, input: &[W], calls: &mut impl GadgetCalls<W>) -> Vec<W> {
        let [x0, x1] = input else { unreachable!() };
        let a = calls.call(GadgetId(0), slice::from_ref(x0));
        let b0 = calls.call(GadgetId(1), slice::from_ref(x1));
        calls.call(GadgetId(1), slice::from_ref(x1));
        vec![a, b0]
    }
}

struct Difference {
    input_len: usize,
}

impl Circuit<Fr> for Difference {
    fn input_len(&self) -> usize {
        self.input_len
    }

    fn gadgets(&self) -> Gadgets {
        Vec::new()
    }

    fn eval<W: Wire<Fr>>(&self, input: &[W], _: &mut impl GadgetCalls<W>) -> Vec<W> {
        vec![input[0].clone() - input[1].clone()]
    }
}

fn bit() -> Box<dyn Gadget<Fr>> {
    Box::new(RangeProduct { bounds: 0..=1 })
}

fn new_error<C: Circuit<Fr>>(circuit: C) -> Option<FlpError> {
    Flp::<Fr, C>::new(circuit).err()
}

fn accepts<C: Circuit<Fr>>(flp: &Flp<Fr, C>, input: &[u64]) -> bool {
    let mut rng = ark_std::test_rng();
    let input: Vec<Fr> = input.iter().copied().map(Fr::from).collect();
    let vector = flp.prove(&input, &mut rng).unwrap();
    flp.verify(&flp.query(&mut rng), &vector)
}

fn answers(query: &Query<Fr>, vector: &[Fr]) -> Vec<Fr> {
    query
        .linear_queries()
        .iter()
        .map(|linear| {
            linear
                .coeffs
                .iter()
                .zip(vector)
                .map(|(coeff, value)| *coeff * value)
                .sum::<Fr>()
                + linear.constant
        })
        .collect()
}

#[test]
fn new_rejects_malformed_circuits() {
    let cases: [MalformedCase; 7] = [
        (
            || vec![bit()],
            vec![(0, 1), (0, 2)],
            Some(FlpError::ArityMismatch {
                gadget: GadgetId(0),
                call: 1,
                expected: 1,
                actual: 2,
            }),
        ),
        (
            || vec![bit()],
            vec![(1, 1)],
            Some(FlpError::InvalidGadget {
                gadget: GadgetId(1),
            }),
        ),
        (
            || vec![bit(), bit()],
            vec![(0, 1)],
            Some(FlpError::UnusedGadget {
                gadget: GadgetId(1),
            }),
        ),
        (
            || {
                vec![
                    bit(),
                    Box::new(ParallelSum {
                        inner: Square,
                        count: 0,
                    }),
                ]
            },
            vec![(0, 1)],
            Some(FlpError::ZeroArity {
                gadget: GadgetId(1),
            }),
        ),
        (
            || {
                vec![
                    bit(),
                    Box::new(RangeProduct {
                        bounds: RangeInclusive::new(1, 0),
                    }),
                ]
            },
            vec![(0, 1), (1, 1)],
            Some(FlpError::ZeroDegree {
                gadget: GadgetId(1),
            }),
        ),
        (Vec::new, vec![], Some(FlpError::NoOutputs)),
        (|| vec![bit()], vec![(0, 1)], None),
    ];
    for (gadgets, script, expected) in cases {
        assert_eq!(new_error(ScriptedCircuit { gadgets, script }), expected);
    }
}

#[test]
fn new_rejects_unbuildable_parameters() {
    let empty = || RangeInclusive::new(5, 3);
    let norm_bound = |input_bits, norm_bits| NormBound {
        len: 1,
        input_bits,
        norm_bits,
        square_chunk: None,
    };
    assert_eq!(
        new_error(Range {
            len: 2,
            bounds: empty()
        }),
        Some(FlpError::EmptyBounds)
    );
    assert_eq!(
        new_error(BitRange {
            len: 2,
            bounds: empty()
        }),
        Some(FlpError::EmptyBounds)
    );
    assert_eq!(
        new_error(Membership {
            len: 2,
            values: BTreeSet::new()
        }),
        Some(FlpError::EmptySet)
    );
    for (len, end) in [(1, u64::MAX), (2, u64::MAX), (1, 1 << 63), (1, 1 << 32)] {
        assert_eq!(
            new_error(Range {
                len,
                bounds: 0..=end
            }),
            Some(FlpError::DomainTooLarge {
                gadget: GadgetId(0)
            }),
            "{len} {end}"
        );
    }
    assert_eq!(
        new_error(NormBound {
            len: 4,
            input_bits: 3,
            norm_bits: 7,
            square_chunk: NonZeroUsize::new(5),
        }),
        Some(FlpError::ChunkTooLarge { chunk: 5, len: 4 })
    );
    assert_eq!(
        new_error(SignedNormBound {
            len: 4,
            input_bits: NonZeroUsize::new(3).unwrap(),
            norm_bits: 7,
            square_chunk: NonZeroUsize::new(5),
        }),
        Some(FlpError::ChunkTooLarge { chunk: 5, len: 4 })
    );
    assert_eq!(new_error(norm_bound(126, 254)), None);
    assert_eq!(
        new_error(norm_bound(127, 254)),
        Some(FlpError::ExceedsField {
            bits: 255,
            limit: 254
        })
    );
    assert_eq!(
        new_error(norm_bound(126, 255)),
        Some(FlpError::ExceedsField {
            bits: 255,
            limit: 254
        })
    );
    let mean_variance = |input_bits| MeanVariance { len: 1, input_bits };
    assert_eq!(new_error(mean_variance(126)), None);
    assert_eq!(
        new_error(mean_variance(127)),
        Some(FlpError::ExceedsField {
            bits: 255,
            limit: 254
        })
    );
}

#[test]
fn error_messages_are_distinct_lowercase_sentences() {
    let gadget = GadgetId(1);
    let errors = [
        FlpError::ZeroArity { gadget },
        FlpError::ZeroDegree { gadget },
        FlpError::InvalidGadget { gadget },
        FlpError::ArityMismatch {
            gadget,
            call: 2,
            expected: 1,
            actual: 3,
        },
        FlpError::UnusedGadget { gadget },
        FlpError::NoOutputs,
        FlpError::DomainTooLarge { gadget },
        FlpError::InputLength {
            expected: 4,
            actual: 5,
        },
        FlpError::EmptyBounds,
        FlpError::EmptySet,
        FlpError::ExceedsField {
            bits: 255,
            limit: 254,
        },
        FlpError::ChunkTooLarge { chunk: 5, len: 4 },
    ];
    let messages: BTreeSet<String> = errors.iter().map(ToString::to_string).collect();
    assert_eq!(messages.len(), errors.len());
    for message in &messages {
        assert!(message.starts_with(char::is_lowercase), "{message}");
        assert!(!message.ends_with('.'), "{message}");
    }
}

#[test]
fn engine_types_are_thread_safe() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Flp<Fr, NormBound>>();
    assert_send_sync::<Query<Fr>>();
    assert_send_sync::<FlpError>();
}

#[test]
fn reports_layout_lengths() {
    for len in [3, 4] {
        let flp = Flp::<Fr, _>::new(Range::binary(len)).unwrap();
        let wire_size = (len + 1).next_power_of_two();
        let poly_len = 2 * (wire_size - 1) + 1;
        assert_eq!(flp.proof_len(), 1 + poly_len, "{len}");
    }
}

#[test]
fn custom_circuit_with_public_gadgets() {
    let flp = Flp::<Fr, _>::new(BoundedSquareSum).unwrap();
    assert!(accepts(&flp, &[2, 3, 13]));
    assert!(!accepts(&flp, &[2, 3, 12]));
    assert!(!accepts(&flp, &[4, 0, 16]));
}

#[test]
fn affine_circuit_without_gadgets() {
    let flp = Flp::<Fr, _>::new(Difference { input_len: 2 }).unwrap();
    assert_eq!(flp.query_count(), 1);
    assert!(accepts(&flp, &[5, 5]));
    assert!(!accepts(&flp, &[5, 6]));
}

#[test]
fn rejects_query_for_another_circuit() {
    let mut rng = ark_std::test_rng();
    let two_bits = Flp::<Fr, _>::new(ScriptedCircuit {
        gadgets: || vec![bit(), bit()],
        script: vec![(0, 1), (1, 1)],
    })
    .unwrap();
    let paired_squares = Flp::<Fr, _>::new(ScriptedCircuit {
        gadgets: || {
            vec![Box::new(ParallelSum {
                inner: Square,
                count: 2,
            })]
        },
        script: vec![(0, 2)],
    })
    .unwrap();
    assert_eq!(two_bits.query_count(), paired_squares.query_count());
    assert!(two_bits.decide(&two_bits.query(&mut rng), &[Fr::ZERO; 3]));
    assert!(!two_bits.decide(&paired_squares.query(&mut rng), &[Fr::ZERO; 3]));

    let narrow = Flp::<Fr, _>::new(Difference { input_len: 2 }).unwrap();
    let wide = Flp::<Fr, _>::new(Difference { input_len: 3 }).unwrap();
    let narrow_vector = narrow.prove(&[Fr::ZERO; 2], &mut rng).unwrap();
    let wide_vector = wide.prove(&[Fr::ZERO; 3], &mut rng).unwrap();
    assert!(narrow.verify(&narrow.query(&mut rng), &narrow_vector));
    assert!(!narrow.verify(&wide.query(&mut rng), &narrow_vector));
    assert!(!narrow.verify(&narrow.query(&mut rng), &wide_vector));
}

#[test]
fn rejects_cross_gadget_cancellation() {
    let mut rng = ark_std::test_rng();
    let flp = Flp::<Fr, _>::new(SplitBitChecks).unwrap();
    assert_eq!(flp.vector_len(), 14);

    let i = Radix2EvaluationDomain::<Fr>::new(4).unwrap().element(1);
    let two = Fr::from(2u64);
    let c1 = two / (-Fr::ONE - i);
    let c0 = -two * i / (-Fr::ONE - i);
    assert_eq!(c0 - c1, two);
    assert_eq!(c1 * i + c0, Fr::ZERO);

    let mut vector = flp.prove(&[two, Fr::ZERO], &mut rng).unwrap();
    vector[3] -= c0;
    vector[4] -= c1;
    vector[7] += c0;
    vector[8] += c1;
    assert!(!flp.verify(&flp.query(&mut rng), &vector));
}

#[test]
fn decides_external_answers_masked_by_fresh_seeds() {
    let mut rng = ark_std::test_rng();
    let flp = Flp::<Fr, _>::new(Range::binary(4)).unwrap();
    let input = [1u64, 0, 1, 1].map(Fr::from);
    let query = flp.query(&mut rng);
    let first = answers(&query, &flp.prove(&input, &mut rng).unwrap());
    let second = answers(&query, &flp.prove(&input, &mut rng).unwrap());
    assert!(flp.decide(&query, &first) && flp.decide(&query, &second));
    assert_ne!(first[0], second[0]);
    assert!(!flp.decide(&query, &first[..first.len() - 1]));
}

#[test]
fn rejects_proof_polynomial_forged_to_vanish_on_calls() {
    let mut rng = ark_std::test_rng();
    let flp = Flp::<Fr, _>::new(Range::binary(2)).unwrap();
    let input = [2u64, 0].map(Fr::from);
    let mut vector = flp.prove(&input, &mut rng).unwrap();
    for coeff in &mut vector[input.len() + 1..] {
        *coeff = Fr::ZERO;
    }
    let query = flp.query(&mut rng);
    let answers = answers(&query, &vector);
    assert_eq!(answers.last(), Some(&Fr::ZERO));
    assert!(!flp.decide(&query, &answers));
}

#[test]
fn prove_rejects_wrong_input_length() {
    let flp = Flp::<Fr, _>::new(Range::binary(3)).unwrap();
    assert_eq!(
        flp.prove(&[Fr::ZERO; 2], &mut ark_std::test_rng()),
        Err(FlpError::InputLength {
            expected: 3,
            actual: 2
        })
    );
}
