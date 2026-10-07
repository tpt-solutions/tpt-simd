# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release.
- `dot_product_i16`: wrapping `i32` accumulation, with an explicit AVX2 `vpmaddwd` kernel (two accumulators, 32 elements per iteration) on x86_64 builds with `avx2` enabled, bit-identical to the portable path.
- `dot_product_saturating_i16`: saturating `i32` accumulation over 8 lane accumulators combined by a halving tree (fixed order, portable).
- `dot_product_f32`: 32 independent partial sums, no FMA, fixed combination order (portable; bit-identical on every target).
- `dot_product_complex_f32`: lane-wise (non-conjugated) complex dot product over `ComplexSimd<f32, 8>` slices.
- `portable` module with the reference implementations.
- Uniform length policy: equal-length slices required (panic naming the function and both lengths), empty input returns zero, any length accepted.
- Feature flags `std`, `nightly`, `scalar-only` (the last also disables the AVX2 kernel).
- `#![no_std]`; `unsafe` confined to the AVX2 module and forbidden otherwise.
- Unit and property tests (`src/tests.rs`), a FLAC-style LPC integration test (`tests/flac_lpc.rs`) and Criterion benchmarks (`benches/dot.rs`).

### Notes

- Float results use a fixed multi-accumulator order without FMA; they can differ from a naive left-to-right loop by normal rounding.
- Saturating accumulation clamps per accumulator, which can differ from clamping the exact sum once.
