# Fully Linear PCPs in Rust (RustFLPCP)

[![CI](https://github.com/WeiqiNs/RustFLPCP/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/WeiqiNs/RustFLPCP/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/WeiqiNs/RustFLPCP/graph/badge.svg)](https://codecov.io/gh/WeiqiNs/RustFLPCP)

A fully linear PCP lets a client prove that its input is valid, such as a one-hot vote or a
vector with a bounded norm, to a verifier who only ever computes linear functions of the input
and the proof. That is exactly what a server can do with data it cannot read: under
inner-product functional encryption (IPFE) it decrypts inner products, and on secret-shared
data it adds shares. RustFLPCP implements the construction of Theorem 4.3 in
[Boneh, Boyle, Corrigan-Gibbs, Gilboa, and Ishai](https://eprint.iacr.org/2019/188) as a
generic engine: write a validity circuit once and get a prover, explicit linear queries, and a
decision procedure. It is the proof layer of [PEAR](https://eprint.iacr.org/2025/420), is
generic over any [arkworks](https://arkworks.rs) FFT-friendly field, and is tested on
BLS12-381 and Goldilocks.

```rust
use ark_bls12_381::Fr;
use rand::rngs::OsRng;
use rust_flpcp::{Flp, OneHot};

let rng = &mut OsRng;
let flp = Flp::<Fr, _>::new(OneHot { len: 4 })?;
let vector = flp.prove(&[0u64, 0, 1, 0].map(Fr::from), rng)?;

let query = flp.query(rng);
let answers: Vec<Fr> = query
    .linear_queries()
    .iter()
    .map(|linear| {
        let inner: Fr = linear.coeffs.iter().zip(&vector).map(|(c, v)| *c * v).sum();
        inner + linear.constant
    })
    .collect();
assert!(flp.decide(&query, &answers));
```

The inner products above run in the clear; under IPFE each `coeffs` becomes a function key and
the server learns only the answers. This is `examples/quickstart.rs`.

## What fully linear PCPs are for

Private aggregation systems collect inputs that no single server may see: telemetry
histograms, survey answers, federated-learning updates. A malicious client can still send
garbage, such as a vote of 1000 or a gradient that swamps the model, and the servers cannot
inspect the input to catch it. A fully linear PCP fixes that: the client attaches a short
proof, the verifier asks a few random linear queries of `input || proof`, and it accepts only
valid inputs while learning nothing else. Because every query is linear, the servers can answer
them on ciphertexts or shares.

## Highlights

- **One engine, any circuit.** A circuit is written once against a generic wire type; the same
  code builds the proof on field values and derives the queries symbolically. Multiplying two
  wires does not compile, so every circuit is affine apart from its gadgets.
- **Two queries per single-gadget check.** Every gadget identity folds into the output check
  with an independent random weight, so one IPFE decryption per gadget input plus one combined
  decryption decides a proof.
- **Fast prover.** Wire and proof polynomials live on radix-2 FFT domains, so proving is
  O(n log n).
- **Sound and zero knowledge.** A cheating vector passes with probability about the proof
  degree over the field size; the answers to one query reveal nothing about a valid input.
- **Ready-made circuits** for the Prio3-style checks in the table below.
- Safe Rust (`unsafe_code` is forbidden), MSRV 1.86, every source line covered by tests, and
  each security check backed by a test that fails when the check is removed.

## Circuits

| Circuit | Accepts | Typical use |
|---|---|---|
| `Range { len, bounds }` | every coordinate in `bounds`; `Range::binary(len)` is `0..=1` | small enumerations, bits |
| `BitRange { len, bounds }` | every coordinate in `bounds`, through bit decompositions | wide ranges, b-bit counters |
| `OneHot { len }` | exactly one coordinate is 1, the rest 0 | histograms, votes |
| `Membership { len, values }` | every coordinate is one of `values` | categorical answers |
| `NormBound { len, input_bits, norm_bits, square_chunk }` | coordinates below `2^input_bits`, sum of squares below `2^norm_bits` | bounded vectors |
| `SignedNormBound { len, input_bits, norm_bits, square_chunk }` | coordinates in `[-2^(input_bits - 1), 2^(input_bits - 1) - 1]`, sum of squares below `2^norm_bits` | federated-learning update clipping |
| `MeanVariance { len, input_bits }` | each claimed `y` equals `x^2` for `x` below `2^input_bits` | private mean and variance |

- `Range` and `Membership` use one gadget whose degree is the number of allowed values, so their
  proofs grow with it; prefer `BitRange` for wide ranges. When `bounds` holds a power-of-two
  number of values (more than one), `BitRange` drops the decomposition of the distance to the
  upper end.
- `BitRange`, `NormBound`, `SignedNormBound`, and `MeanVariance` take auxiliary bits; build
  their input with the circuit's `encode`. `SignedNormBound::encode` takes `i64` values, so
  scale fixed-point values to integers first.
- `NormBound` and `SignedNormBound` square through the bit gadget (`x^2 = x(x - 1) + x`) by
  default; `square_chunk: Some(c)` uses a `ParallelSum` of `c` `Square` gadgets instead, which
  changes the proof length and adds `c` queries. `c` may not exceed `len`.
- Parameters whose bit decompositions, bounds, or allowed values could wrap around the field
  modulus are rejected with `FlpError::ExceedsField`; this only matters for fields smaller
  than 64 bits. The check covers one client's input: when an aggregator sums many accepted
  inputs, keeping those sums below the modulus is the caller's job.

For the input, vector, proof, and query counts of each circuit, run
`cargo run --example costs`.

## Verifying through linear queries

The prover outputs one vector, `input || proof`. A `Query` is a list of
`LinearQuery { coeffs, constant }`, and each answer is `<coeffs, vector> + constant`: under
IPFE the verifier holds a function key for each `coeffs`, decrypts the inner product, and adds
`constant` afterwards. There is one query per gadget input wire plus one combined query, so
`query_count` is the sum of the gadget arities plus one. `decide(&query, &answers)` checks the
answers; `verify(&query, &vector)` is the plaintext shortcut. The `Query` holds the verifier's
private gadget weights, which `decide` needs and the prover must never see.

`decide` evaluates each gadget on its wire answers and accepts when the weighted sum of those
evaluations equals the combined answer: the same weighted sum of the gadget polynomials at the
random point, plus a randomly weighted sum of the circuit outputs. A cheating vector either has
a gadget polynomial that disagrees with its gadget at the random point, except with probability
at most that polynomial's degree over `|F| - |H|` (with `H` the largest wire domain, which the
random point avoids), or has a nonzero circuit output. The hidden, independent gadget and
output weights then make the whole sum vanish with probability at most `1/|F|`. The weights
must be drawn per gadget: with a shared weight, a prover could move an error from one gadget
polynomial to another so that the two differences cancel. For an honest proof the combined
answer is determined by the wire answers, so folding the gadget identities into it leaves zero
knowledge unchanged.

There is no Fiat–Shamir transform: soundness rests on the verifier's query randomness being
chosen independently of the proof and kept from the prover. Zero knowledge holds when each
vector is answered against a single `Query`: every wire polynomial carries one random seed, so
the wire answers of two queries at different points could be combined to cancel it and reveal a
linear function of the input.

## Writing a circuit

Implement `Circuit<F>`:

- `input_len` is the length of the input vector.
- `gadgets` lists the non-affine building blocks; a gadget's position is its `GadgetId`.
- `eval` maps the input wires to output wires that must all be zero for a valid input. It is
  generic over `W: Wire<F>`, which supports addition, subtraction, scaling by a field element,
  and constants. The only non-affine operation is `calls.call(GadgetId(i), inputs)`. The same
  `eval` runs symbolically to derive the linear queries and on field values to build the proof.
- `check_params` optionally rejects parameters before anything else runs.

The gadgets `RangeProduct`, `SetProduct`, `Square`, and `ParallelSum` are public, and any type
implementing `Gadget<F>` (`arity`, `degree`, `eval`) can be used; a gadget must also be
`Debug + Send + Sync`, so an `Flp` can be shared across verifier threads. `Flp::new` validates
the circuit once and returns an `FlpError` if a gadget is unused, called with the wrong number
of inputs, or too large for the field's FFT domains.

```rust
struct BoundedSquareSum;

impl Circuit<Fr> for BoundedSquareSum {
    fn input_len(&self) -> usize {
        3
    }

    fn gadgets(&self) -> Vec<Box<dyn Gadget<Fr>>> {
        vec![
            Box::new(RangeProduct { bounds: 0..=3 }),
            Box::new(ParallelSum { inner: Square, count: 2 }),
        ]
    }

    fn eval<W: Wire<Fr>>(&self, input: &[W], calls: &mut impl GadgetCalls<W>) -> Vec<W> {
        let [x, y, claimed] = input else { unreachable!() };
        vec![
            calls.call(GadgetId(0), slice::from_ref(x)),
            calls.call(GadgetId(0), slice::from_ref(y)),
            calls.call(GadgetId(1), &[x.clone(), y.clone()]) - claimed.clone(),
        ]
    }
}
```

A `ParallelSum` call proves only the sum of its inner gadget over the chunks. Use it where the
circuit needs that sum, such as a sum of squares compared to a claimed value; never use it to
batch zero checks. Terms can cancel over the field: with `x = (1 + sqrt(-23)) / 2`,
`3 * (3 - 1) + x * (x - 1) = 0` although neither value is a bit. Give each zero check its own
output: the verifier weights every output with independent randomness, and that is what makes
such a cancelling pair fail.

## Using it

```toml
[dependencies]
rust_flpcp = { git = "https://github.com/WeiqiNs/RustFLPCP" }
rand = "0.8"
```

- `prove` and `query` take `rng: &mut impl ark_std::rand::Rng`, the `rand` 0.8 trait that
  arkworks uses, so depend on `rand` 0.8 and pass a cryptographically secure generator such as
  `OsRng`. `ark_std::test_rng()` is deterministic and meant only for tests and benchmarks.
- Each gadget's wire polynomials are interpolated over a power-of-two domain with room for the
  gadget's calls plus one random seed. A gadget called `2^k - 1` times fills its domain
  exactly; one more call doubles it (compare the 63/64 and 1023/1024 rows of the costs
  example).
- The `parallel` feature (off by default) enables the rayon-backed parallelism of `ark-ff`,
  `ark-poly`, and `ark-std`.

## Build and test

```sh
cargo test --all
cargo clippy --all --all-targets -- -D warnings
cargo llvm-cov --all --fail-uncovered-lines 0
cargo deny check && cargo machete
cargo bench --bench flp
cargo run --example costs
```

CI runs formatting, clippy (pedantic, with and without `parallel`), the docs, the test suite
under coverage with an upload to Codecov, the MSRV build on Rust 1.86, and the dependency audit.

## Related work

[libprio-rs](https://github.com/divviup/libprio-rs) implements fully linear PCPs for the Prio3
VDAF. There the verifiers evaluate the queries directly on secret shares of the proof and
exchange the results, so the query vectors are never materialized. RustFLPCP exports explicit
coefficient vectors instead, so they can serve as IPFE function keys.
