# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release of `tpt-simd-complex`: complex-number helpers for FFT and MDCT
  code built on `ComplexSimd` from `tpt-simd-core`.
- `ComplexSimdF32Ext` for every `ComplexSimd<f32, N>`: `fmadd`, `fmsub`,
  `twiddle_mul_fma`, `mul_conj`, `norm_sq`, `from_polar`, `i_mul`, `neg_i_mul`,
  `scale_scalar`, `fft_butterfly`, `from_interleaved`, `to_interleaved`.
- `ComplexSimdF64Ext` for every `ComplexSimd<f64, N>`: `twiddle_mul_fma`,
  `mul_conj`, `norm_sq`, `i_mul`, `neg_i_mul`.
- Slice helpers `interleaved_to_split` and `split_to_interleaved`.
- Re-exports of `ComplexSimd`, `F32x8`, `Simd` and `tpt_simd_mul::complex_mul_f32`.
- Feature flags `std`, `nightly`, `scalar-only` (forwarded to `tpt-simd-core`
  and `tpt-simd-mul`); `no_std` by default.
- Unit and property tests, doc examples, and a Criterion benchmark
  (`cargo bench -p tpt-simd-complex`).

### Notes

- `#![forbid(unsafe_code)]`; compile-time dispatch only.
- All fused methods use FMA rounding on every target (hardware FMA or correctly
  rounded software `fmaf`), so results are bit-identical across targets.
- `from_polar` uses `libm::sincosf`.
