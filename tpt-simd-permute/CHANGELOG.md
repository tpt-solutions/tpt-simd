# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `transpose_8x8_i16` and `transpose_4x4_f32` in-place block transposes (SSE2 unpack networks on x86_64), with `transpose_8x8_i16_portable` and `transpose_4x4_f32_portable` references.
- `interleave_stereo_i16` and `deinterleave_stereo_i16` for planar/packed stereo audio.
- `interleave_yuv420`: planar I420 to semi-planar NV12.
- `unpack_i8_to_i16`: sign-extending widen of a byte slice.
- `permute_f32` and `permute_i32` lane permutations (`vpermps`/`vpermd` on AVX2) with `permute_f32_portable` and `permute_i32_portable` references, and the generic `permute` over `Simd<T, N>` (indices wrap modulo `N`).
- Re-exports of `pack_i16_to_i8` and `pack_i16_to_i8_slice` from `tpt-simd-saturate`.
- Slice functions assert exact length relationships and panic with descriptive messages (ADR 0002).
- Cargo features `std`, `nightly` and `scalar-only`, forwarded to `tpt-simd-core` and `tpt-simd-saturate`; `scalar-only` also disables the SSE2 transposes and AVX2 permutes. `default` is empty. The crate is `#![no_std]`.
- Doc-tests on every public function, and unit and property-based tests in `src/tests.rs`.
- Criterion benchmark `benches/permute.rs`.

### Notes

- Results are identical between the SIMD and portable paths.
