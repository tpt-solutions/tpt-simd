# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
