# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `select_i32` and `select_lanes_f32`: index-based lane selection (`out[i] = v[indices[i] & 7]`), never panicking.
- `select_from_slice_i32`: selection from a slice by index via `tpt-simd-gather` (panics on a bad index).
- `try_select_from_slice_i32`: non-panicking form returning `Option`.
- Re-export of `select_f32` (sign-bit condition blend) from `tpt-simd-blend`.
- Cargo features `std`, `nightly` and `scalar-only`, forwarded to `tpt-simd-core`; `default` is empty. The crate is `#![no_std]`.
- Doc-tests on every public function, and property-based and unit tests covering lane selection, slice selection, panics and negative indices.

### Notes

- `select_lanes_f32` is the index-based function renamed from the spec's `select_f32` to avoid a name clash with the blend-based `select_f32`.
- The features are not forwarded to `tpt-simd-blend` or `tpt-simd-gather`.
- No benchmarks are included.
