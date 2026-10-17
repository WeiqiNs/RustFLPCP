use ark_bls12_381::Fr;
use ark_ff::{AdditiveGroup, Field};
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main};
use rust_flpcp::{
    BitRange, Circuit, Flp, MeanVariance, Membership, NormBound, OneHot, Range, SignedNormBound,
};
use std::collections::BTreeSet;
use std::fmt::Display;
use std::hint::black_box;
use std::num::NonZeroUsize;

const LENS: [usize; 3] = [64, 256, 1024];
const DENOMINATIONS: [u64; 7] = [1, 2, 5, 10, 20, 50, 100];

fn bench_circuit<C: Circuit<Fr>>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    parameter: impl Display,
    circuit: C,
    input: &[Fr],
) {
    let flp = Flp::<Fr, C>::new(circuit).unwrap();
    let mut rng = ark_std::test_rng();
    let vector = flp.prove(input, &mut rng).unwrap();
    let query = flp.query(&mut rng);
    assert!(flp.verify(&query, &vector), "{parameter}");
    group.bench_function(BenchmarkId::new("prove", &parameter), |b| {
        b.iter(|| flp.prove(black_box(input), &mut rng).unwrap());
    });
    group.bench_function(BenchmarkId::new("query", &parameter), |b| {
        b.iter(|| flp.query(&mut rng));
    });
    group.bench_function(BenchmarkId::new("verify", &parameter), |b| {
        b.iter(|| flp.verify(black_box(&query), black_box(&vector)));
    });
}

fn bench_lens<C: Circuit<Fr>>(
    c: &mut Criterion,
    label: &str,
    setup: impl Fn(usize) -> (C, Vec<Fr>),
) {
    let mut group = c.benchmark_group(label);
    for len in LENS {
        let (circuit, input) = setup(len);
        bench_circuit(&mut group, len, circuit, &input);
    }
    group.finish();
}

fn measurement(len: usize, modulus: u64) -> Vec<u64> {
    (0..len as u64).map(|index| index % modulus).collect()
}

fn norm_bound(len: usize, square_chunk: Option<NonZeroUsize>) -> (NormBound, Vec<Fr>) {
    let circuit = NormBound {
        len,
        input_bits: 8,
        norm_bits: 27,
        square_chunk,
    };
    let input = circuit.encode(&measurement(len, 256));
    (circuit, input)
}

fn circuits(c: &mut Criterion) {
    bench_lens(c, "range_binary", |len| {
        let input = measurement(len, 2).into_iter().map(Fr::from).collect();
        (Range::binary(len), input)
    });
    bench_lens(c, "range_0_15", |len| {
        let input = measurement(len, 16).into_iter().map(Fr::from).collect();
        (
            Range {
                len,
                bounds: 0..=15,
            },
            input,
        )
    });
    bench_lens(c, "bit_range_0_1023", |len| {
        let circuit = BitRange {
            len,
            bounds: 0..=1023,
        };
        let input = circuit.encode(&measurement(len, 1024));
        (circuit, input)
    });
    bench_lens(c, "norm_bound", |len| norm_bound(len, None));

    let mut group = c.benchmark_group("norm_bound_chunk");
    for square_chunk in [
        None,
        NonZeroUsize::new(4),
        NonZeroUsize::new(16),
        NonZeroUsize::new(64),
    ] {
        let (circuit, input) = norm_bound(1024, square_chunk);
        bench_circuit(&mut group, format!("{square_chunk:?}"), circuit, &input);
    }
    group.finish();
}

fn len_1024(c: &mut Criterion) {
    let len = 1024;
    let mut group = c.benchmark_group("len_1024");

    let mut one_hot = vec![Fr::ZERO; len];
    one_hot[0] = Fr::ONE;
    bench_circuit(&mut group, "one_hot", OneHot { len }, &one_hot);

    let membership = Membership {
        len,
        values: BTreeSet::from(DENOMINATIONS),
    };
    let input: Vec<Fr> = DENOMINATIONS
        .iter()
        .cycle()
        .take(len)
        .map(|&value| Fr::from(value))
        .collect();
    bench_circuit(&mut group, "membership", membership, &input);

    let signed_norm_bound = SignedNormBound {
        len,
        input_bits: NonZeroUsize::new(8).unwrap(),
        norm_bits: 27,
        square_chunk: None,
    };
    let signed_values: Vec<i64> = (-128..128).cycle().take(len).collect();
    let input = signed_norm_bound.encode(&signed_values);
    bench_circuit(&mut group, "signed_norm_bound", signed_norm_bound, &input);

    let mean_variance = MeanVariance { len, input_bits: 8 };
    let input = mean_variance.encode(&measurement(len, 256));
    bench_circuit(&mut group, "mean_variance", mean_variance, &input);

    group.finish();
}

criterion_group!(benches, circuits, len_1024);
criterion_main!(benches);
