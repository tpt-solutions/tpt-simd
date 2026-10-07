# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release.
- `f32` elementary functions `exp`, `ln`, `sin`, `cos`, `tanh` and `erf`, each with four entry points: `*_scalar`, `*_f32x8`, `*_f32` (slice to slice) and `*_f32_inplace`.
- Branch-free kernels (range reduction plus polynomial, conditionals as selects) using only `+ - * /`; no `libm` calls and no FMA contraction, so results are bit-identical across targets, feature sets and entry points (tails included).
- Documented ULP bounds asserted by tests: `exp` 1.1 (1.0 in the subnormal range), `ln` 1.0, `sin`/`cos` 3.0 up to `abs(x) <= 8192` and 3.5 up to 100000, `tanh` 1.5, `erf` 3.0.
- Special-value handling: NaN passthrough, signed zeros, infinities, overflow/underflow of `exp`, subnormal inputs; `sin`/`cos` return NaN beyond `abs(x) > 100000`.
- Re-export of `Simd` from `tpt-simd-core`.
- Feature flags `std`, `nightly`, `scalar-only`, forwarded to `tpt-simd-core`.
- `#![no_std]` and `#![forbid(unsafe_code)]`.
- Dense-sweep and random-bit-pattern tests (`src/tests.rs`) and Criterion benchmarks against `libm` and `std` (`benches/math.rs`).

### Notes

- Accuracy is an ULP bound, not bit equality with `libm` or `std` (ADR 0003, Tier 2).
