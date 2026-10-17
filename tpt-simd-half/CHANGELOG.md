# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- F16C slice conversions for `f32_to_f16_slice` and `f16_to_f32_slice`, eight values per instruction. Used when F16C and AVX are enabled at compile time, or detected at run time with the new `runtime-dispatch` feature (which needs `std`). Results match `half` bit for bit, NaN payloads included, over every f16 pattern and a large f32 sample.
- Criterion (`f32_to_f16`, 65536 values, one host): 67 µs in the default build, 2.6 µs with `runtime-dispatch` (and in the AVX2 build). A separate harness on 1M values gave 760 → 100 µs (f32→f16) and 691 → 119 µs (f16→f32).
- `scalar-only` feature forces the software conversions. The umbrella crate forwards `std`, `scalar-only` and `runtime-dispatch`.

## [0.1.1] - 2026-10-17

### Added
- Element-wise `f16` and `bf16` arithmetic.

## [0.1.0] - 2026-10-12

### Added

- Initial release.
- `f16` and `bf16` types, re-exported from the `half` crate.
- `f32_to_f16_slice`, `f16_to_f32_slice`, `f32_to_bf16_slice`, `bf16_to_f32_slice`: length-checked slice conversion.
- `quantize_i8` and `dequantize_i8`: symmetric per-tensor int8 with a caller-supplied scale.
- `BlockQ8_0`, `quantize_q8_0`, `dequantize_q8_0`: 32-value blocks with an `f16` scale, ggml `Q8_0` layout.
- `BlockQ4_0`, `quantize_q4_0`, `dequantize_q4_0`: 32-value 4-bit blocks with an `f16` scale, ggml `Q4_0` layout.
- `BLOCK` constant (32).
- Paired slices must match in length, block counts must match value counts, and `scale` must be positive and finite; violations panic.
- `#![no_std]`.
