# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release of `tpt-simd-window`.
- Allocation-free generators `hamming_window_into_f32`,
  `hanning_window_into_f32`, `blackman_window_into_f32` and
  `kaiser_window_into_f32` (symmetric windows, denominator `N - 1`).
- `Vec`-returning generators `hamming_window_f32`, `hanning_window_f32`,
  `blackman_window_f32` and `kaiser_window_f32` behind the `alloc` feature.
- Window application: `apply_window_f32`, `apply_window_simd_f32` and the scalar
  reference `apply_window_scalar_f32`.
- Math helpers `cos_approx_f32`, `cos_approx_simd_f32`, `bessel_i0_f64` and
  `bessel_i0_f32`.
- Feature flags `alloc`, `std` (implies `alloc`), `nightly`, `scalar-only`;
  `no_std` by default.
- Unit and property tests, doc examples, and a Criterion benchmark
  (`cargo bench -p tpt-simd-window`).

### Notes

- Cosine-sum windows are accurate to about `3e-6` versus `f64`; the polynomial
  cosine has a maximum absolute error of `2e-6` for `|x| <= 1000`. Kaiser is
  evaluated in `f64`.
