# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release of `tpt-simd-butterfly`: in-place radix-2 butterfly
  `(a, b) <- (a + b, a - b)` on `tpt-simd-core` vectors.
- `butterfly_f32` (8 x `f32`), `butterfly_i16` (16 x `i16`, wrapping) and
  `butterfly_i16_saturating` (16 x `i16`, saturating).
- `butterfly_generic` (any lane type and width) and
  `butterfly_generic_saturating` (any integer lane type and width).
- `butterfly_complex_f32` for split-layout `ComplexSimd<f32, 8>`.
- Twiddle forms: `butterfly_with_twiddle_f32` (real twiddle, unfused multiply)
  and `butterfly_with_twiddle_complex_f32` (decimation-in-time,
  `(a + b*w, a - b*w)`), replacing the ambiguous spec signature.
- Feature flags `std`, `nightly`, `scalar-only` (all forwarded to
  `tpt-simd-core`); `no_std` by default.
- Unit and property tests, doc examples, and a Criterion benchmark
  (`cargo bench -p tpt-simd-butterfly`) comparing scalar and SIMD forms.

### Notes

- `#![forbid(unsafe_code)]`; compile-time dispatch only (LLVM vectorisation).
- Integer butterflies wrap or saturate as named; floats follow IEEE-754.
