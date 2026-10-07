# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Re-exports of the `tpt-simd-vector` backend: `Simd`, `SimdMask`, `SimdElement`, `SimdInt`, `SimdFloat`, `LaneCast`, `MaskLane`.
- Type aliases `F32x4/8/16`, `F64x2/4/8`, `I8x16/32/64`, `I16x8/16/32`, `I32x4/8/16`, `I64x2/4/8` and the unsigned `U*` equivalents, plus `Mask8<T>`, `Mask16<T>`, `Mask32<T>`.
- `SimdVector` and `SimdOps` traits and the `DefaultBackend` marker.
- `width` module: `NATIVE_VECTOR_BITS`, `NATIVE_VECTOR_BYTES`, `native_lanes::<T>()`, `NATIVE_F32_LANES`, `NATIVE_I32_LANES`, `NATIVE_I16_LANES`, `NATIVE_I8_LANES`.
- `detect` module: `Features` (SSE2, SSE4.1, AVX, AVX2, FMA, AVX-512F, NEON, SVE, RVV) with `compile_time()` and `runtime()` (real CPU queries on x86/x86_64 and aarch64 with `std`).
- `ComplexSimd<T, N>`: split real/imaginary vectors with `new`, `splat`, `zero`, `add`, `sub`, `neg`, `conj`, `mul`, `scale`, `mag_sq`, `div`, `twiddle_mul`, `butterfly`, `mag`, `phase` and the `+ - * /` / unary `-` operators.
- Features `std` (runtime detection, forwards to `tpt-simd-vector/std`), `nightly` and `scalar-only` (forwarded; currently no-ops).
- `#![no_std]`, `#![forbid(unsafe_code)]`; unit tests in `src/tests.rs` and doc tests.

### Notes

- `ComplexSimd` uses unfused arithmetic, so results are identical on every target. `div` follows IEEE semantics for zero divisors.
- Width constants come from `cfg(target_feature)`; they are hints and every width works on every target.
