# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release of `tpt-simd-convolve`.
- `convolve_1d_f32` (full 1-D convolution, zero extension) and
  `convolve_1d_output_len`.
- `convolve_2d_separable_f32`: "same"-size separable 2-D convolution of a
  row-major image with zero padding and caller-provided scratch.
- `fir_filter_i16`: causal Q15 FIR filter with 64-bit accumulation,
  round-half-up and `i16` saturation.
- Scalar references `convolve_1d_scalar_f32`, `convolve_2d_separable_scalar_f32`
  and `fir_filter_scalar_i16`, bit-identical to the SIMD versions.
- Feature flags `std`, `nightly`, `scalar-only` (forwarded to `tpt-simd-core`);
  `no_std` and allocation-free by default.
- Unit and property tests, doc examples, and a Criterion benchmark
  (`cargo bench -p tpt-simd-convolve`) comparing SIMD and scalar paths.

### Notes

- Fixed accumulation order (`fma` over taps in ascending order) keeps SIMD and
  scalar results bit-identical.
- Small private `unsafe` helpers call x86 FMA intrinsics only when `avx`/`fma`
  are statically enabled; other targets use the portable fused path.
