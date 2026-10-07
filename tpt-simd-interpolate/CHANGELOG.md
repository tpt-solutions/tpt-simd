# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release of `tpt-simd-interpolate`.
- Linear: `interpolate_linear_f32` and `interpolate_linear_simd_f32`.
- Catmull-Rom cubic: `interpolate_cubic_scalar_f32` and `interpolate_cubic_f32`.
- Lanczos-`a`: `interpolate_lanczos_scalar_f32` and `interpolate_lanczos_f32`
  (normalised weights, `2a` samples).
- Normalised sinc: `sinc_f32` (`libm`) and `sinc_simd_f32` (polynomial).
- Feature flags `std`, `nightly`, `scalar-only` (forwarded to `tpt-simd-core`);
  `no_std` by default.
- Unit and property tests, doc examples, and a Criterion benchmark
  (`cargo bench -p tpt-simd-interpolate`).

### Notes

- Linear and cubic are bit-identical between SIMD and scalar versions.
- `sinc_simd_f32` has a maximum absolute error of `2e-6` for `|x| <= 64`;
  Lanczos SIMD results agree with the scalar reference to about `1e-5`.
