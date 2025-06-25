# RustFLPCP

[![CI](https://github.com/WeiqiNs/RustFLPCP/actions/workflows/ci.yml/badge.svg)](https://github.com/WeiqiNs/RustFLPCP/actions/workflows/ci.yml)

This Rust library implements the non-interactive zero-knowledge proof protocol introduced by
[Boneh et al.](https://eprint.iacr.org/2019/188.pdf) constructed from the fully-linear probabilistically checkable
proofs.
It provides several simple validity checks:

- `BinaryValidation`: verifies that the input vector represents a valid binary vector.
- `RangeValidation`: verifies each element in the input vector falls within a specified range.
- `NormBoundValidation`: verifies whether the L2 norm of the input vector is below a given threshold.

The script relies on [arkworks](https://arkworks.rs)  for prime field arithmetic.
Additionally, it incorporates the verification-time optimizations described
in [this paper](https://eprint.iacr.org/2025/420) to improve performance.