# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- AVX2 `peak` and `fir` on x86_64 builds with `avx2` enabled. `peak` is an exact maximum of absolute values, with the NaN rule of `f32::max`. `fir` keeps each output's sum in tap order, eight outputs per vector, and falls back to the scalar loop for the first `taps - 1` outputs and the tail. Both are bit-identical to the scalar code (`tests/simd_matches_scalar.rs`). Criterion (`fir_32_taps`): 1.10 ms naive vs 103 µs in the AVX2 build; the default build is scalar and unchanged. `peak` has no criterion bench; a harness measured 6.3 → 2.0 µs on 16K samples.
- `scalar-only` feature forces the scalar code. The umbrella crate forwards it.

### Changed

- The crate root is `deny(unsafe_code)`; `unsafe` is confined to the `x86` module.
- `rms` is unchanged. Its f64 sum is strictly ordered, so a vector sum would change the rounding.

## [0.1.1] - 2026-10-17

### Added
- Channel-batched biquad over interleaved frames.

### Changed
- Tests: external oracles for the biquad and resampler, proptest reference tests; Criterion benches.

## [0.1.0] - 2026-10-12

### Added

- Initial release.
- `mix_into`: `dst += gain * src`.
- `interleave` and `deinterleave`: planar <-> interleaved conversion.
- `peak` and `rms`: level of a buffer.
- `fir`: causal direct-form FIR filter with zero history.
- `Biquad`: RBJ low-pass and high-pass filters, direct form II transposed, with `f64` state.
- `resample_sinc`: windowed-sinc resampling with a Blackman window.
- Length rules checked up front; mismatched lengths panic.
- `#![no_std]`; no runtime dependencies beyond `libm`.
