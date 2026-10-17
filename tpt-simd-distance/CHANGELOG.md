# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- AVX2 lane kernels for `inner_product_f32`, `l2_squared_f32` and `cosine_distance_f32` (and so `top_k_l2_f32`) on x86_64 builds with `avx2` enabled. Bit-identical to the portable path: same lane mapping, separate multiply and add, no FMA. Measured on one host, 1024 `f32`: inner product 116 ns to 44 ns, squared L2 145 ns to 48 ns.
- `portable` module with the reference lane kernels.
- `scalar-only` feature forces the portable path. The umbrella crate forwards it.

### Changed

- The crate root is `forbid(unsafe_code)` unless the AVX2 path is compiled. `unsafe` is confined to the `x86` module, which is the only place that allows it.

## [0.1.1] - 2026-10-17

- No code changes; version bumped with the workspace.

## [0.1.0] - 2026-10-12

### Added

- Initial release.
- `inner_product_f32`, `l2_squared_f32`, `cosine_distance_f32`: `f32` kernels with eight partial sums in a fixed order, no FMA, identical on every target.
- `inner_product_i8`, `l2_squared_i8`, `inner_product_u8`, `l2_squared_u8`: exact integer kernels with `i64` accumulation.
- `hamming_distance_u64`: exact Hamming distance over `u64` bit-vectors.
- `top_k_l2_f32`: top-k nearest neighbours by squared L2 distance.
- Paired slices must have equal lengths; mismatches panic. Empty input returns zero.
- `#![no_std]`; no runtime dependencies.
