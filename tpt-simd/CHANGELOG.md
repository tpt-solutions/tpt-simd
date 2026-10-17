# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `scalar-only` forwards to `tpt-simd-audio`, `tpt-simd-bits`, `tpt-simd-dct`, `tpt-simd-distance`, `tpt-simd-half`, `tpt-simd-ntt` and `tpt-simd-pixel`, which gained AVX2 or F16C paths. `runtime-dispatch` and `std` forward to `tpt-simd-half` (F16C detection).

## [0.1.1] - 2026-10-17

### Fixed

- `scalar-only` and `std` now forward to every member crate that defines them, not only `tpt-simd-core`. Before, `tpt-simd/scalar-only` did not force the portable path in the kernel crates.
- Re-exports the nine published crates the description already promised: `audio`, `bits`, `dct`, `distance`, `geom`, `half`, `ntt`, `pixel` and `transpose`.
- README and CHANGELOG now match the manifest: `runtime-dispatch` is in the default feature set, so the default is not empty. `no_std` builds use `default-features = false`.


## [0.1.0] - 2026-10-12

### Added

- Umbrella crate re-exporting `tpt-simd-core` at the root and every other published workspace crate as a module: `aligned`, `blas`, `blend`, `butterfly`, `compare`, `complex`, `convolve`, `dot`, `fixed`, `gather`, `horizontal`, `interpolate`, `math`, `matrix`, `mul`, `permute`, `reduce`, `rng`, `rounding`, `saturate`, `scatter`, `select`, `shift`, `sparse`, `window`.
- Features `std`, `nightly`, `scalar-only` (forwarded to `tpt-simd-core`) and `runtime-dispatch` (implies `std`, enables `tpt-simd-blas/runtime-dispatch`; see ADR 0003).
- `#![no_std]`, `#![forbid(unsafe_code)]`; default features are empty.
