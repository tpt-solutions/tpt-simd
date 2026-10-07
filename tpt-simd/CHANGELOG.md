# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Umbrella crate re-exporting `tpt-simd-core` at the root and every other published workspace crate as a module: `aligned`, `blas`, `blend`, `butterfly`, `compare`, `complex`, `convolve`, `dot`, `fixed`, `gather`, `horizontal`, `interpolate`, `math`, `matrix`, `mul`, `permute`, `reduce`, `rng`, `rounding`, `saturate`, `scatter`, `select`, `shift`, `sparse`, `window`.
- Features `std`, `nightly`, `scalar-only` (forwarded to `tpt-simd-core`) and `runtime-dispatch` (implies `std`, enables `tpt-simd-blas/runtime-dispatch`; see ADR 0003).
- `#![no_std]`, `#![forbid(unsafe_code)]`; default features are empty.
