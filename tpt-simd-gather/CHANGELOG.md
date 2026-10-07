# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `gather_i32`, `gather_f32`: `unsafe` eight-lane gathers by signed `i32` element index (`vpgatherdd`/`vgatherdps` on AVX2).
- `gather_i32_portable`, `gather_f32_portable`: `unsafe` scalar reference implementations.
- `gather_checked_i32`, `gather_checked_f32`: safe slice gathers that panic on an out-of-bounds or negative index.
- `try_gather_i32`, `try_gather_f32`: safe slice gathers returning `Option`, reading nothing unless all eight indices are valid.
- AVX2 path compiled in only for `x86_64` with `avx2` enabled and `scalar-only` off; portable loop otherwise.
- Cargo features `std`, `nightly` and `scalar-only`, forwarded to `tpt-simd-core`; `scalar-only` also disables the AVX2 gather. `default` is empty. The crate is `#![no_std]`.
- Doc-tests on every public function, and property-based and unit tests covering duplicates, negative indices, out-of-bounds rejection and bit-exact `f32` NaN payloads.
- Criterion benchmark `benches/gather.rs`.

### Notes

- Hardware gather is not necessarily faster than eight scalar loads (see the crate docs).
- The `unsafe` functions require every addressed element to be valid for reads; the checked and `try_` variants are safe.
