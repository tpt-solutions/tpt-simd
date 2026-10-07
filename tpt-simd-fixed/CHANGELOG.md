# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Fixed<T, INT_BITS, FRAC_BITS, N>` SIMD fixed-point vector over `i32` or `i16` storage (sealed `FixedRepr` trait), with compile-time format validation and support for narrow formats stored in wider lanes.
- Construction and conversion: `from_raw`, `raw`, `splat_raw`, `splat_f32`, `from_f32` (ties to even, saturating, NaN to 0), `to_f32`, `min_value`, `max_value`, `one`, `epsilon`, `clamp_to_format`, and `convert::<I2, F2>` between Q formats.
- Wrapping and saturating arithmetic: `wrapping_add/sub/mul/neg`, `saturating_add/sub/mul/neg/abs/div`, `abs`, `mul_trunc`, `div`, `checked_div`, `min`, `max`, `clamp`.
- Operators `+ - * / -x` and assign forms (wrapping at storage width).
- AVX2 backend for `i32` multiplies (`_mm256_mul_epi32` on even/odd lanes), bit-identical to the portable path; disabled by `scalar-only`.
- Features: `std`, `nightly`, `scalar-only` (forwarded to `tpt-simd-core`).
- `#![no_std]`; `unsafe` is only permitted when the AVX2 backend is compiled in.
- Property tests against an `i128` reference (multiply, divide, add/sub, negate/abs, `f32` round trip, format conversion), edge-case tests, doc tests, and Criterion benchmarks (`cargo bench -p tpt-simd-fixed`).

### Notes

- `mul`, `div` and `convert` round to nearest with ties away from zero; `mul_trunc` floors; `from_f32` rounds ties to even.
- `div` and `/` panic on a zero divisor lane; `checked_div` and `saturating_div` do not.
- Operators wrap on overflow (ADR 0002); use the `saturating_*` methods to clamp.
