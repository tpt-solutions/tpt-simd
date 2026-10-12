# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release.
- `inner_product_f32`, `l2_squared_f32`, `cosine_distance_f32`: `f32` kernels with eight partial sums in a fixed order, no FMA, identical on every target.
- `inner_product_i8`, `l2_squared_i8`, `inner_product_u8`, `l2_squared_u8`: exact integer kernels with `i64` accumulation.
- `hamming_distance_u64`: exact Hamming distance over `u64` bit-vectors.
- `top_k_l2_f32`: top-k nearest neighbours by squared L2 distance.
- Paired slices must have equal lengths; mismatches panic. Empty input returns zero.
- `#![no_std]`; no runtime dependencies.
