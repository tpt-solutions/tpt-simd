# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
