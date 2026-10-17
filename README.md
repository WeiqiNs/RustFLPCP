# Fully Linear PCPs in Rust (RustFLPCP)

[![CI](https://github.com/WeiqiNs/RustFLPCP/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/WeiqiNs/RustFLPCP/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/WeiqiNs/RustFLPCP/graph/badge.svg)](https://codecov.io/gh/WeiqiNs/RustFLPCP)

RustFLPCP lets a client prove its input is valid, such as a one-hot vote or a vector with a
bounded norm, to a verifier that only computes linear functions of the input and the proof. The
check therefore works on data the verifier cannot read: inner-product functional encryption
(IPFE) ciphertexts or secret shares. It implements Theorem 4.3 of
[Boneh et al.](https://eprint.iacr.org/2019/188) as a generic engine for
[PEAR](https://eprint.iacr.org/2025/420): write a validity circuit once and get a prover,
explicit linear queries, and a decision procedure over any [arkworks](https://arkworks.rs)
FFT field.

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

Under IPFE each `coeffs` is a function key and the verifier sees only the answers;
`flp.verify(&query, &vector)` is the plaintext shortcut. This is `examples/quickstart.rs`.

## Circuits

| Circuit | Accepts |
|---|---|
| `Range { len, bounds }` | every value in `bounds` (`Range::binary(len)` for bits) |
| `BitRange { len, bounds }` | every value in `bounds`, via bit decompositions (for wide ranges) |
| `OneHot { len }` | exactly one value is 1, the rest 0 |
| `Membership { len, values }` | every value is one of `values` |
| `NormBound { len, input_bits, norm_bits, square_chunk }` | values below `2^input_bits`, sum of squares below `2^norm_bits` |
| `SignedNormBound { .. }` | the same for signed values, as in federated-learning clipping |
| `MeanVariance { len, input_bits }` | each claimed `y` equals `x^2`, for private mean and variance |

Circuits with auxiliary bits build their input with `encode`. Run `cargo run --example costs`
for each circuit's vector length and query count.

## Writing a circuit

Implement `Circuit<F>`: `input_len`, `gadgets`, and an `eval` generic over `W: Wire<F>` that
returns outputs which must all be zero for a valid input. Wires only support affine operations;
every multiplication goes through `calls.call(GadgetId(i), inputs)` on a gadget (`RangeProduct`,
`SetProduct`, `Square`, `ParallelSum`, or your own `Gadget<F>`). The same `eval` builds both the
proof and the queries. `ParallelSum` proves only a sum, so never use it to batch zero checks.

## Security notes

- Pass a cryptographically secure generator (`rand` 0.8, such as `OsRng`); `test_rng()` is for
  tests only.
- The `Query` holds the verifier's secret randomness: keep it from the prover, and answer each
  vector against a single query, or zero knowledge is lost.
- A cheating proof passes with probability about its polynomial degree over the field size.

## Using it

```toml
[dependencies]
rust_flpcp = { git = "https://github.com/WeiqiNs/RustFLPCP" }
rand = "0.8"
```

The `parallel` feature enables arkworks' multithreaded FFTs.

## Build and test

```sh
cargo test --all
cargo clippy --all --all-targets -- -D warnings
cargo llvm-cov --all --fail-uncovered-lines 0
cargo bench --bench flp
```
