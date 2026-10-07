# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `SplitMix64` seeding generator.
- Scalar `Xoshiro256pp` reference generator with `from_state`, `from_seed`, `state`, `jump` (2^128 steps) and `long_jump` (2^192 steps).
- `Xoshiro256ppX8`: eight independent xoshiro256++ streams in structure-of-arrays layout, with `from_seed`, `from_seed_block`, `from_lanes`, `lane`, `jump_blocks`, `step` and `next_simd`.
- `Philox4x32X8`: counter-based Philox4x32-10 generator computing eight blocks per step, with `(key, stream)` independence and `seek` random access; scalar reference `philox4x32_10`.
- `Rng8` trait with vectorised slice fills: `fill_u64`, `fill_u32`, `fill_f32`, `fill_f64` (uniform in `[0, 1)`), `fill_normal_f32`, `fill_normal_f64` (Box-Muller).
- `u32_to_unit_f32` and `u64_to_unit_f64` conversions.
- Public `math` module with branch-free 8-lane `ln`, `sincos_turns`, `sqrt` (f32 and f64) and `box_muller` (f32 and f64).
- Features: `std`, `nightly`, `scalar-only` (all forwarded to `tpt-simd-core`).
- `#![no_std]`, `#![forbid(unsafe_code)]`.
- Unit and property tests (known-answer vectors, lane/scalar equivalence, jump distance, statistical checks, math accuracy against libm) and Criterion benchmarks (`cargo bench -p tpt-simd-rng`).

### Notes

- Not cryptographically secure.
- Output is bit-identical on every target and feature set (no FMA, no libm).
- Normal variates are truncated at `|z| <= 5.77` (f32) and `<= 8.5` (f64); `f64` uniforms carry 52 random bits.
