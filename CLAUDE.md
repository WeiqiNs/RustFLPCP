# RustFLPCP for agents

A generic fully linear PCP engine (Boneh et al., CRYPTO 2019) in Rust over arkworks, generic over
`ark_ff::FftField` and tested on BLS12-381 and Goldilocks. It is the proof layer of PEAR (PoPETs 2025):
a client proves its input satisfies a validity circuit, and the verifier only ever needs inner products
`<coeffs, input || proof> + constant`, which PEAR evaluates under inner-product functional encryption.
`README.md` is the user-facing overview.

## Commands

```bash
cargo fmt --all -- --check
cargo clippy --all --all-targets -- -D warnings
cargo clippy --all --all-targets --features parallel -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo llvm-cov --all --fail-uncovered-lines 0
cargo bench --bench flp -- --test
cargo run --example quickstart && cargo run --example costs
cargo deny check && cargo machete
```

CI (`.github/workflows/ci.yml`) runs these on pushes, pull requests and a weekly schedule, in three jobs on the latest
stable Rust: lint, the test suite with `--features parallel`, and docs; tests under coverage, the benchmark smoke run
and both examples, then an OIDC upload to Codecov; and the dependency audit.
Clippy pedantic is on through `[lints]` in `Cargo.toml`. The coverage gate is `--fail-uncovered-lines 0` because
`--fail-under-lines 100` scores each monomorphization of a generic function separately and cannot reach 100 even
when every line runs. `cargo run --example costs` prints each circuit's vector length and query count.

## Layout

- `src/circuit.rs`: the public traits. `Circuit<F>` (`input_len`, `gadgets`, `eval`, `check_params`), `Wire<F>`
  (affine operations only, implemented by `F` and `LinearForm<F>`), `Gadget<F>` (`arity`, `degree`, `eval`,
  `check_params`), `GadgetCalls<W>`, `GadgetId`, and the crate-wide `check_field_bits`.
- `src/linear_form.rs`: `LinearForm`, a sparse combination of `Variable::Input` and `Variable::GadgetOutput`.
- `src/flp/mod.rs`: `Flp`, `GadgetPlan` (per-gadget domains and vector offsets), `decide`, `verify`.
- `src/flp/prove.rs`: `prove` and `gadget_poly`. `src/flp/query.rs`: `query`, `Query`, `LinearQuery`, and the
  in-crate reference test of query construction.
- `src/gadgets.rs`: `RangeProduct`, `SetProduct`, `Square`, `ParallelSum`.
- `src/circuits/`: one file per shipped circuit; `mod.rs` holds the shared helpers (`BIT`, `bit_gadget`,
  `call_each`, `square_via_bit`, `binary_sum`, `low_bits`, `u64_bits`). `norm_bound.rs` holds both
  `NormBound` and `SignedNormBound`, which share `eval_shifted` and `encode_shifted`.
- `src/error.rs`: `FlpError`. `tests/`: one file per circuit plus `flp.rs` (engine) and `goldilocks.rs` (a small
  field); `tests/common` has `accepts` and `assert_tampering_rejected`.
- `benches/flp.rs`, `examples/costs.rs`, `examples/quickstart.rs` (the README's opening example).

## How it works

- `Flp::new` runs `Circuit::check_params`, then the circuit's `eval` once with `LinearForm` wires (`SymbolicCalls`),
  recording every gadget call's inputs and every output as linear forms, then builds a `GadgetPlan` per gadget, which
  runs `Gadget::check_params` and the arity, usage and domain checks. All validation happens here.
- Each gadget gets a radix-2 wire domain of size `(calls + 1).next_power_of_two()`: node 0 holds a random seed per
  input slot (zero knowledge), call k sits at node k + 1, and padding nodes are zero. Its proof polynomial has
  `degree * (N - 1) + 1` coefficients. The vector is `input || per gadget: seeds || polynomial coefficients`.
- `prove` runs `eval` on field values (`ProverCalls`) to fill the wire nodes; `gadget_poly` interpolates each wire
  (IFFT), evaluates on a larger domain (FFT), applies the gadget pointwise and interpolates back.
- `query` samples a point outside the largest wire domain plus one weight per gadget and per output
  (`QueryCoins`). There is one `LinearQuery` per gadget input slot (Lagrange coefficients at the point) and one
  combined query: the weighted gadget polynomials at the point plus the weighted outputs. `QueryAccumulator::finish`
  turns gadget-output terms into coefficients on the proof polynomial with one FFT per gadget.
- `decide` accepts when the weighted sum of each gadget evaluated on its wire answers equals the combined answer.
  `verify` computes the answers in the clear and calls `decide`.

## Adding a circuit

Add a struct with named fields in `src/circuits/`, implement `Circuit<F>`, give it an `encode` if its input carries
auxiliary bits, and reject parameters in `check_params` (bit widths through `check_field_bits`). Re-export it from
`src/circuits/mod.rs` and `src/lib.rs`, add `tests/<circuit>.rs` using `tests/common`, a config in
`query_answers_match_reference_evaluation`, and rows in `examples/costs.rs` and `benches/flp.rs`.

## Design facts

- Gadget weights are drawn per gadget: with one shared weight a prover can shift an error between two gadget
  polynomials so they cancel (`rejects_cross_gadget_cancellation` in `tests/flp.rs`).
- Every circuit output gets its own random weight, which is what rejects cancelling non-bits such as 3 and
  (1 + sqrt(-23)) / 2 (`binary_rejects_non_bits_including_cancelling_pair` in `tests/range.rs`). A `ParallelSum`
  call proves only a sum, so it serves sums of squares, not batched zero checks.
- Parameters whose decompositions or constants could wrap the field modulus fail with `FlpError::ExceedsField`
  via `check_field_bits`: circuits check their bit widths in `Circuit::check_params`, and `RangeProduct`,
  `SetProduct` and `ParallelSum` check their constants in `Gadget::check_params`, which also covers custom
  circuits. Only fields under 64 bits can hit this (`tests/goldilocks.rs`).
- `BitRange` drops the upper-gap decomposition when the bounds hold a power-of-two number of values, decided only
  in `BitRange::upper_gap_bits`; a single-value range stays unaligned (`degenerate_single_value_range`).
- The `Query` carries the verifier's private gadget weights, so `decide` takes it alongside the answers.
- `Gadget<F>: Debug + Send + Sync` keeps `Flp` shareable across verifier threads (`engine_types_are_thread_safe`).
- `prove` and `query` take arkworks' `ark_std::rand::Rng`, which is `rand` 0.8, so the `rand` dev-dependency used
  by examples and tests stays on 0.8 until arkworks moves.
