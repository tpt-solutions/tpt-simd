# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Saturating add and subtract for `i16x16`, `i8x32`, `u8x32` and `u16x16` (`saturating_add_*`, `saturating_sub_*`).
- `saturating_mul_i16`: exact 32-bit product clamped to the `i16` range.
- Saturating narrowing packs `pack_i16_to_i8`, `pack_i16_to_u8` and `pack_i32_to_i16`, preserving natural lane order.
- Slice forms `pack_i16_to_i8_slice`, `pack_i16_to_u8_slice` and `pack_i32_to_i16_slice` (panic on length mismatch; tails handled identically to the vector body).
- Implemented as auto-vectorised lane loops; the crate is `#![forbid(unsafe_code)]` and `#![no_std]`.
- Cargo features `std`, `nightly` and `scalar-only`, forwarded to `tpt-simd-core`; `default` is empty.
- Doc-tests on every public function, and unit and property-based tests in `src/tests.rs` covering edge values, lane order and slice tails.
- Criterion benchmark `benches/saturate.rs`.

### Notes

- All operations clamp rather than wrap; there is no tolerance and no unsafe code.
- `scalar-only` has no further effect in this crate (no intrinsic paths).
