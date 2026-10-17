# RustFLPCP

[![CI](https://github.com/WeiqiNs/RustFLPCP/actions/workflows/ci.yml/badge.svg)](https://github.com/WeiqiNs/RustFLPCP/actions/workflows/ci.yml)

A Rust implementation of fully linear probabilistically checkable proofs (FLPs) for arithmetic
circuits with gadgets, following the construction of Theorem 4.3 in
[Boneh, Boyle, Corrigan-Gibbs, Gilboa, and Ishai](https://eprint.iacr.org/2019/188). Any
circuit written against the `Circuit` trait gets a prover, a linear-query generator, and a
decision procedure, and the crate ships ready-made circuits for common client-input checks.
The query layout targets verification through inner-product functional encryption, as in
[PEAR](https://eprint.iacr.org/2025/420). Field arithmetic and FFTs come from
[arkworks](https://arkworks.rs); the library is generic over any `ark_ff::FftField`.

## Verifying through linear queries

The prover outputs one vector, `input || proof`. The verifier never needs that vector in the
clear: a `Query` is a list of `LinearQuery { coeffs, constant }`, and each answer is
`<coeffs, vector> + constant`. With inner-product functional encryption (IPFE) the verifier
holds a functional key for each `coeffs`, decrypts the inner product, adds `constant` after
decryption, and calls `decide` with the query and the answers. There is one linear query per
gadget input wire plus one combined query, so `query_count` is the sum of the gadget arities
plus one. The `Query` also holds the verifier's private gadget weights, which `decide` needs and
the prover must never see.

`decide` evaluates each gadget on its wire answers and accepts when the weighted sum of those
evaluations equals the combined answer: the same weighted sum of the gadget polynomials at the
random point, plus a randomly weighted sum of the circuit outputs. A cheating vector either
has a gadget polynomial that disagrees with its gadget at the random point, except with
probability at most that polynomial's degree over `|F| - |H|` (with `H` the largest wire
domain, which the random point avoids), or has a nonzero circuit output. The
hidden, independent gadget and output weights then make the whole sum vanish with probability
at most `1/|F|`. The weights must be drawn per gadget: with a shared weight, a prover could
move an error from one gadget polynomial to another so that the two differences cancel. For an
honest proof the combined answer is determined by the wire answers, so folding the gadget
identities into it leaves zero knowledge unchanged.

There is no Fiat–Shamir transform: soundness rests on the verifier's query randomness being
chosen independently of the proof and kept from the prover.

Zero knowledge holds when each vector is answered against a single `Query`. Every wire
polynomial carries one random seed, so the wire answers of two queries at different points
can be combined to cancel it and reveal a linear function of the input.

## API flow

```rust
use rust_flpcp::{Flp, Range};

let flp = Flp::<Fr, _>::new(Range { len: 4, bounds: 10..=20 })?;

let vector = flp.prove(&[10u64, 12, 15, 20].map(Fr::from), &mut rng)?;

let query = flp.query(&mut rng);
assert!(flp.verify(&query, &vector));

let answers: Vec<Fr> = query
    .linear_queries()
    .iter()
    .map(|linear| ipfe_inner_product(&linear.coeffs) + linear.constant)
    .collect();
assert!(flp.decide(&query, &answers));
```

`verify` is the plaintext shortcut for computing the answers and calling `decide`.
`Flp::new` validates the circuit once and returns an `FlpError` describing what is wrong;
`prove` rejects an input of the wrong length.

## Circuits

- **`Range { len, bounds }`**: every coordinate lies in `bounds`, checked by one gadget whose
  degree is the number of values in `bounds`. `Range::binary(len)` is the `0..=1` case. The
  proof grows with the width of `bounds`, so prefer `BitRange` for wide ranges.
- **`BitRange { len, bounds }`**: every coordinate lies in `bounds`, checked through bit
  decompositions of its distance to both ends. When `bounds` holds a power-of-two number of
  values (more than one), the distance to the lower end alone pins the value, so the upper
  decomposition is omitted. Build the input with `BitRange::encode`.
- **`OneHot { len }`**: every coordinate is a bit and exactly one of them is set, as in a
  one-hot vote or histogram contribution.
- **`Membership { len, values }`**: every coordinate is one of `values`, checked by one
  `SetProduct` gadget whose degree is the number of values; `values` may not be empty.
- **`NormBound { len, input_bits, norm_bits, square_chunk }`**: every coordinate is below
  `2^input_bits`, and the circuit accepts exactly when the sum of squares is below
  `2^norm_bits`. Build the input with `NormBound::encode`. With `square_chunk: None` the
  squares reuse the bit gadget (`x^2 = x(x - 1) + x`); `Some(c)` adds a `ParallelSum` of `c`
  `Square` gadgets instead, which changes the proof length and adds `c` queries; `c` may
  not exceed `len`.
- **`SignedNormBound { len, input_bits, norm_bits, square_chunk }`**: the signed counterpart
  of `NormBound`. Every coordinate lies in `[-2^(input_bits - 1), 2^(input_bits - 1) - 1]`, and
  the circuit accepts exactly when the sum of squares is below `2^norm_bits`, as when clipping
  client updates in federated learning. Build the input with `SignedNormBound::encode` from
  `i64` values; callers scale fixed-point values to integers first.
- **`MeanVariance { len, input_bits }`**: the input holds each value `x` below `2^input_bits`
  together with a claimed square `y`, and the circuit checks `y = x^2` through the bit gadget.
  Summing accepted inputs across clients yields `sum(x)` and `sum(x^2)`, from which the
  aggregator computes the mean and variance. Build the input with `MeanVariance::encode`.

Circuits whose bit decompositions, bounds, or allowed values could wrap around the field
modulus are rejected with `FlpError::ExceedsField`; this only matters for fields smaller than
64 bits. That check covers one client's input: when an aggregator adds many
clients' accepted inputs, such as the squares of `NormBound` or `MeanVariance`, keeping those
sums below the field modulus is the caller's responsibility. For concrete input, vector,
proof, and query counts of each circuit, run `cargo run --example costs`.

## Writing a circuit

Implement `Circuit<F>`:

- `input_len` is the length of the input vector.
- `gadgets` lists the non-affine building blocks; a gadget's position is its `GadgetId`.
- `eval` maps the input wires to output wires that must all be zero for a valid input. It is
  generic over `W: Wire<F>`, which supports addition, subtraction, scaling by a field element,
  and constants. The only non-affine operation is `calls.call(GadgetId(i), inputs)`;
  multiplying two wires does not compile. The same `eval` runs symbolically to derive the
  linear queries and on field values to build the proof.
- `check_params` optionally rejects parameters before anything else runs.

The gadgets `RangeProduct`, `SetProduct`, `Square`, and `ParallelSum` are public, and any type
implementing `Gadget<F>` (`arity`, `degree`, `eval`) can be used; a gadget must also be
`Debug + Send + Sync`, so an `Flp` can be shared across verifier threads. `Flp::new` requires
every gadget to be called at least once, with exactly its arity in inputs.

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

### `ParallelSum` and cancellation

A `ParallelSum` call proves only the sum of its inner gadget over the chunks. Use it where the
circuit needs that sum, such as a sum of squares compared to a claimed value; never use it to
batch zero checks. Terms can cancel over the field: with `x = (1 + sqrt(-23)) / 2`,
`3 * (3 - 1) + x * (x - 1) = 0` although neither value is a bit. Give each zero check its own
output: the verifier weights every output with independent randomness, and that is what makes
such a cancelling pair fail.

## Padding

Each gadget's wire polynomials are interpolated over a power-of-two domain with room for the
gadget's calls plus one random seed. A gadget called `2^k - 1` times fills its domain exactly;
one more call doubles it. Compare the 63/64, 255/256, and 1023/1024 rows of
`cargo run --example costs`.

## Randomness

`prove` and `query` take `rng: &mut impl ark_std::rand::Rng`. Pass a cryptographically secure
generator. `ark_std::test_rng()` is deterministic and meant only for tests and benchmarks.

## Features

`parallel` (off by default) enables the rayon-backed parallelism of `ark-ff`, `ark-poly`, and
`ark-std`.

## Benchmarks and costs

```sh
cargo bench --bench flp
cargo run --example costs
```

The benchmark measures `prove`, `query`, and `verify` for each shipped circuit, either across
input lengths or at length 1024 (the `len_1024` group), plus `NormBound` across `square_chunk`
settings.

## Related work

[libprio-rs](https://github.com/divviup/libprio-rs) implements FLPs for the Prio3 VDAF. There
the verifiers evaluate the queries directly on secret shares of the proof and exchange the
results, so the query vectors are never materialized. This crate exports explicit coefficient
vectors instead, so they can serve as IPFE function keys.
