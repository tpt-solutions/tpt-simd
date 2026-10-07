# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Uniform-amount `shift_left_i32`, `shift_right_logical_i32`, `shift_right_arithmetic_i32`, `rotate_left_i32` and `shift_with_rounding_i32` (round-half-up, 64-bit intermediate) on `i32x8`.
- Per-lane variants `shift_left_var_i32`, `shift_right_logical_var_i32`, `shift_right_arithmetic_var_i32`, `rotate_left_var_i32` and `shift_with_rounding_var_i32`, with `_portable` references for the three shifts.
- AVX2 `vpsllvd`/`vpsrlvd`/`vpsravd` paths for the variable shifts when `avx2` is enabled at compile time; the `SHIFT_VAR_USES_INTRINSICS` constant reports this.
- Defined out-of-range policy for amounts `>= 32`: left and logical shifts give 0, arithmetic shift sign-fills, rounding shift gives 0, rotate is modulo 32 (ADR 0002).
- Cargo features `std`, `nightly` and `scalar-only`, forwarded to `tpt-simd-core`; `scalar-only` also disables the AVX2 variable shifts. `default` is empty. The crate is `#![no_std]`.
- Doc-tests on every public function, and property-based tests covering out-of-range amounts.
- Criterion benchmark `benches/shift.rs`.

### Notes

- No function panics for any shift amount.
