# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Aligned16`, `Aligned32`, `Aligned64`: `#[repr(C, align(N))]` wrappers with `ALIGN`, `new`, `into_inner`, `Deref`/`DerefMut` and `From<T>`.
- `is_aligned` and `vector_align::<T, N>()`.
- `load_aligned` / `store_aligned` (panicking) and `try_load_aligned` / `try_store_aligned` (non-panicking) alignment-checked `Simd` slice loads and stores.
- `AlignedBuf<T, ALIGN>` densely packed aligned heap buffer (`filled`, `from_slice`, `zeroed`, `len`, `is_empty`, `BASE_ALIGN`; derefs to `[T]`) and `aligned_vec_f32`, `aligned_vec_i16`, `aligned_vec_i32` helpers.
- Features `alloc` (default; enables `AlignedBuf`), `std`, `nightly` and `scalar-only` (forwarded to `tpt-simd-core`).
- `#![no_std]`; unit and property tests (`src/tests.rs`) and doc tests.

### Notes

- `unsafe` is confined to `AlignedBuf` allocation and slice construction, with `// SAFETY:` comments.
- Panic policy follows ADR 0002: plain functions panic on short or misaligned slices, `try_*` variants return `None`/`false`.
