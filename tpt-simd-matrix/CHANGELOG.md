# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `mat4x4_mul_f32`: 4x4 `f32` matrix product using a fused multiply-add per lane, with the unfused scalar reference `mat4x4_mul_scalar_f32`.
- `mat8x8_mul_i16`: 8x8 `i16` matrix product with `i64` accumulation and saturation to `i16`, with the scalar reference `mat8x8_mul_scalar_i16`.
- `mat4x4_transpose_f32` and `mat8x8_transpose_i16`: in-place transposes delegating to `tpt-simd-permute`.
- `mat3x3_inverse_f32`: adjugate-based 3x3 inverse returning `None` for non-finite or nearly singular input, and the `SINGULAR_EPSILON` threshold constant.
- Features: `std`, `nightly`, `scalar-only` (all forwarded to `tpt-simd-core`).
- `#![no_std]`.
- Property tests (multiply vs scalar reference, transpose involution, inverse round trip) and doc tests.

### Notes

- `mat4x4_mul_f32` can differ from the unfused scalar reference by rounding error; `mat8x8_mul_i16` is bit-exact against its reference.
- The singularity test is `|det| <= SINGULAR_EPSILON * max|m|^3` or a non-finite determinant.
