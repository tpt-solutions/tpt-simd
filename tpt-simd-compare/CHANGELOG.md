# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Lane-wise comparisons `cmp_gt_*`, `cmp_lt_*`, `cmp_eq_*`, `cmp_ne_*`, `cmp_ge_*`, `cmp_le_*` for `i32x8`, `f32x8`, `i16x16` and `i8x32` (24 functions, signed integer semantics), returning `SimdMask`.
- `cmp_unord_f32`: unordered (NaN) lane detection.
- Mask helpers `mask_any`, `mask_all`, `mask_count`, `mask_to_bitmask`, `mask_from_bitmask`.
- Fused slice counters `count_gt_i32`, `count_lt_i32`, `count_eq_i32`, `count_gt_f32`, `count_lt_f32`, `count_eq_f32` that never materialise a mask.
- Explicit AVX2 paths (`vpcmpgt*`, `vpcmpeq*`, `vcmpps`, `vmovmskps`/`vpmovmskb`, `popcnt`) selected at compile time; portable fallback elsewhere with identical results.
- Re-export of `SimdMask` from `tpt-simd-core`.
- Cargo features `std`, `nightly` and `scalar-only`, all forwarded to `tpt-simd-core`; `default` is empty. The crate is `#![no_std]`.
- Doc-tests on every public function, and property-based tests (`proptest`) against a scalar reference covering NaN handling and mask helpers.
- Criterion benchmark `benches/compare.rs`.

### Notes

- `f32` comparisons use IEEE 754 ordered semantics identical to Rust's scalar operators: `gt`, `lt`, `ge`, `le`, `eq` are false when either lane is NaN, `ne` is true; `-0.0 == +0.0`.
- The AVX2 `cfg` does not test the `scalar-only` feature.
