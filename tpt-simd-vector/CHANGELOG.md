# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Simd<T, N>`: array-backed vector type (`pub struct Simd<T, const N: usize>(pub [T; N])`) with construction (`splat`, `zero`, `from_array`, `from_fn`, `from_slice`, `from_slice_or`), access (`to_array`, `as_array`, `as_mut_array`, indexing, `copy_to_slice`, `store_partial`), `map`, `zip_with` and `cast`.
- Arithmetic (`+ - * /`, unary `-`) and, for integers, bitwise (`& | ^ !`) and shift (`<<`, `>>`) operators with `*Assign` forms.
- `min`, `max`, `clamp`, `abs`, `reverse`, `swizzle`; comparisons `simd_eq/ne/lt/le/gt/ge` and `is_nan`.
- Integer operations: `sat_add`, `sat_sub`, `sat_mul`, `shl_scalar`, `shr_scalar`, `shr_logical`, `rotl`.
- Float operations: `sqrt`, `floor`, `ceil`, `round`, `round_ties_even`, `trunc`, `mul_add`, `atan2`.
- `SimdMask<T, N>` with `from_array`/`to_array`, `from_raw`/`to_raw`, `splat`, `test`, `set`, `any`, `all`, `count`, `to_bitmask`/`from_bitmask`, `select` and `!`.
- Lane traits `SimdElement`, `SimdInt`, `SimdFloat`, `LaneCast`, `MaskLane`, implemented for `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32` and `f64`.
- Features `std` (use `std` float intrinsics instead of `libm`), `nightly` and `scalar-only` (both currently no-ops).
- `#![no_std]`, `#![forbid(unsafe_code)]`; builds on stable Rust 1.95, edition 2024.
- Unit and property tests (`src/tests.rs`).

### Notes

- Integer arithmetic wraps; shifts by `>= BITS` give 0 (sign fill for arithmetic right shift); float `min`/`max` ignore NaN; comparisons with NaN are false except `!=`.
- Float math is exactly specified and target-independent (`libm`, or equivalent `std` intrinsics with `std`).
