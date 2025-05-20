# RustFLPCP
[![CI](https://github.com/WeiqiNs/RustFLPCP/actions/workflows/ci.yml/badge.svg)](https://github.com/WeiqiNs/RustFLPCP/actions/workflows/ci.yml)

This Rust script implements the zero-knowledge proof (ZKP) protocol introduced by [Boneh et al.](https://eprint.iacr.org/2019/188.pdf).
It features a simple circuit where the `G Gate` is defined as `x * (x - 1)`.
The circuit can be used to verify whether an input is formatted as a valid binary number. 
The script utilizes [arkworks](https://arkworks.rs) for computations in the prime fields.
Additionally, we apply the optimization described in [this paper](https://eprint.iacr.org/2025/420) to speed up the verification.
