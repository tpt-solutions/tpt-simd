# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release.
- Fixed halving-tree reduction order (lane `i` combined with lane `i + ceil(N/2)`), identical on every target; float results are bit-for-bit reproducible across targets.
- Generic reductions `reduce_sum`, `reduce_product`, `reduce_min`, `reduce_max` over any `SimdElement` and any non-zero lane count (panic on zero lanes).
- Fixed-width sums: `horizontal_sum_i32` (wrapping), `horizontal_sum_f32`, and widening sums `horizontal_sum_i16` (to `i32`), `horizontal_sum_u8` (to `u32`), `horizontal_sum_i8` (to `i32`).
- Fixed-width extrema: `horizontal_min_i16`/`horizontal_max_i16`, `horizontal_min_i32`/`horizontal_max_i32`, `horizontal_min_f32`/`horizontal_max_f32`.
- `horizontal_product_f32`.
- Feature flags `std`, `nightly` and `scalar-only`, all forwarded to `tpt-simd-core`.
- `#![no_std]` and `#![forbid(unsafe_code)]`.
- Unit and property tests (`src/tests.rs`) and Criterion benchmarks (`benches/horizontal.rs`).

### Notes

- Float min/max ignore NaN lanes (all-NaN gives NaN); float sum and product propagate NaN.
- Float sums and products are reassociated relative to a left-to-right loop.
