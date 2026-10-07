# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release.
- Float reductions for `f32` and `f64` (`_f32` / `_f64` suffixes): `sum`, `pairwise_sum`, `sum_compensated`, `mean`, `variance`, `sample_variance`, `variance_welford`, `covariance`, `sample_covariance`, `min`, `max`, `argmin`, `argmax`, `sum_squares`, `norm`.
- Integer reductions: `sum_i32` (wrapping), `sum_i32_wide` (exact `i64`), `min_i32`, `max_i32`, `argmin_i32`, `argmax_i32`.
- Multi-accumulator kernels (four accumulators of 8 `f32` / 4 `f64` lanes) with a fixed accumulation order, finished by the `tpt-simd-horizontal` halving tree; no FMA, so results are reproducible across targets and feature sets.
- Vectorised Neumaier/TwoSum compensated sum, falling back to the plain sum for non-finite results.
- Two-pass variance and covariance (robust when the mean is large relative to the spread) and a serial Welford variance.
- NaN-ignoring `min`/`max` and first-index `argmin`/`argmax`; `Option` returns for empty input.
- Feature flags `std`, `nightly`, `scalar-only`, forwarded to `tpt-simd-core` and `tpt-simd-horizontal`.
- `#![no_std]` and `#![forbid(unsafe_code)]`.
- Unit and property tests (`src/tests.rs`) and Criterion benchmarks (`benches/reduce.rs`).

### Notes

- Float sums, means, variances and norms are reassociated relative to a left-to-right loop and differ by a few ulps of the sum of absolute values (ADR 0003, Tier 2). Min/max/arg* and integer functions are exact.
- `norm_*` does no overflow/underflow scaling.
- A slice-length mismatch in `covariance_*` / `sample_covariance_*` panics.
