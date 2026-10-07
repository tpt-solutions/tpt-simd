# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `round_f32` (ties away from zero), `floor_f32`, `ceil_f32`, `trunc_f32` and `round_ties_even_f32` on `f32x8`.
- `round_to_nearest_even_i32` and `round_with_bias_i32`: rounding plus conversion to `i32x8`, saturating with NaN mapped to 0.
- `ROUNDING_USES_INTRINSICS` constant reporting whether the `vroundps` path is compiled in.
- `vroundps` path for floor, ceil, trunc and ties-to-even when `avx` is enabled at compile time; branch-free integer bit-manipulation fallback (no `libm`) elsewhere.
- Cargo features `std`, `nightly` and `scalar-only`, forwarded to `tpt-simd-core`; `scalar-only` also disables the `vroundps` path. `default` is empty. The crate is `#![no_std]`.
- Doc-tests on every public function, and property-based tests against a `libm` reference (a dev-dependency only).
- Criterion benchmark `benches/rounding.rs`.

### Notes

- Float-returning functions preserve NaN, infinities and the sign of zero.
- Results are exact roundings; there is no tolerance.
