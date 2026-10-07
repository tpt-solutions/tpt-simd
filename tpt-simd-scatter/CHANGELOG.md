# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `scatter_i32`, `scatter_f32`: `unsafe` eight-lane scatters by signed `i32` element index (AVX-512 scatter when `avx512f` and `avx512vl` are enabled).
- `scatter_i32_portable`, `scatter_f32_portable`: `unsafe` scalar references storing in lane order.
- `scatter_checked_i32`, `scatter_checked_f32`: safe slice scatters that panic, before writing anything, on a bad index.
- `try_scatter_i32`, `try_scatter_f32`: safe slice scatters returning `Result<(), OutOfBounds>`, leaving the slice untouched on error.
- `OutOfBounds { lane, index }` error type with `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq` and `Display`.
- AVX-512 path compiled in only for `x86_64` with `avx512f` and `avx512vl` enabled and `scalar-only` off; scalar loop otherwise (there is no AVX2 scatter).
- Cargo features `std`, `nightly` and `scalar-only`, forwarded to `tpt-simd-core`; `scalar-only` also disables the AVX-512 scatter. `default` is empty. The crate is `#![no_std]`.
- Doc-tests on every public function, and property-based and unit tests covering duplicate indices and out-of-bounds handling.
- Criterion benchmark `benches/scatter.rs`.

### Notes

- Duplicate indices: the highest-numbered lane wins on every path.
- The `unsafe` functions require every addressed element to be valid for writes; the checked and `try_` variants are safe.
