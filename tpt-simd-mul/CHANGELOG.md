# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Integer multiply variants: `mul_hi_i16`, `mul_lo_i32`, `mul_hi_i32`.
- Widening multiplies: `mul_widen_i16` (`i16 -> i32`) and `mul_widen_i32` (`i32 -> i64`).
- `mul_add_sub_f32`: fused `a*b + c` on even lanes and `a*b - c` on odd lanes.
- `complex_mul_f32` with an AVX2+FMA intrinsic path (x86_64, compile-time selected), the portable reference `complex_mul_f32_portable`, and the `COMPLEX_MUL_F32_USES_INTRINSICS` constant; `ComplexSimd` re-exported from `tpt-simd-core`.
- Features: `std`, `nightly`, `scalar-only` (forwarded to `tpt-simd-core`; `scalar-only` also disables the intrinsic path).
- `#![no_std]`, `#![deny(missing_docs)]`.
- Property and special-value tests (NaN, infinity, signed zero) comparing every function with a scalar model, plus doc tests.

### Notes

- `complex_mul_f32` uses a fixed fused rounding rule, so the intrinsic and portable paths are bit-identical on every target; targets without hardware FMA use a slower software fused multiply-add.
- NEON is not implemented; aarch64 uses the portable path.
- The only `unsafe` code is the AVX2 intrinsic block.
