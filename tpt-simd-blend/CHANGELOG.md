# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `blend_i32`, `blend_f32`, `blend_i8`: per-lane `mask ? a : b` with a `SimdMask` on `i32x8`, `f32x8` and `i8x32`; lane bits (NaN payloads, signed zeros) are copied exactly.
- `select_f32`: `_mm256_blendv_ps`-style selection driven by the sign bit of a float condition vector.
- `blend_imm_i32`, `blend_imm_f32`, `blend_imm_i8`: compile-time-mask blends (`MASK` const generic; a set bit selects `a`).
- `blend_gt_f32`, `blend_gt_i32`: fused `if x > y { a } else { b }` without materialising a mask.
- Explicit AVX2 paths (`vblendvps`, `vpblendvb`, `vcmpps`/`vpcmpgtd`) selected at compile time via `target_feature = "avx2"`; portable fallback on all other targets with identical results.
- Re-export of `SimdMask` from `tpt-simd-core`.
- Cargo features `std`, `nightly` and `scalar-only`, all forwarded to `tpt-simd-core`; `default` is empty. The crate is `#![no_std]`.
- Doc-tests on every public function, and property-based tests (`proptest`) against a scalar reference including NaN-bit, sign-bit and raw-mask cases.
- Criterion benchmark `benches/blend.rs`.

### Notes

- `blend_imm_*` use the portable per-lane implementation on all targets; no explicit immediate-blend intrinsic is used.
- The AVX2 `cfg` does not test the `scalar-only` feature.
