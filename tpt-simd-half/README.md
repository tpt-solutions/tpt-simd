# tpt-simd-half

Half-precision (`f16`, `bf16`) slice conversions and low-bit quantisation: int8 per tensor, and the ggml-compatible Q8_0 and Q4_0 block formats.

## Overview

Use this crate to convert `f32` buffers to and from 16-bit float formats, and to quantise weights or activations to 8-bit or 4-bit integers for model storage and inference.

| Item | Source | Notes |
|---|---|---|
| `f16`, `bf16` | re-exported from `half` | Types only; conversions are in this crate |
| `f32_to_f16_slice`, `f16_to_f32_slice` | this crate | Length-checked slice conversion |
| `f32_to_bf16_slice`, `bf16_to_f32_slice` | this crate | Length-checked slice conversion |
| `quantize_i8`, `dequantize_i8` | this crate | Symmetric per-tensor int8 with a caller-supplied scale |
| `BlockQ8_0`, `quantize_q8_0`, `dequantize_q8_0` | this crate | 32-value blocks, one `f16` scale |
| `BlockQ4_0`, `quantize_q4_0`, `dequantize_q4_0` | this crate | 32-value blocks, 4-bit values, one `f16` scale |

The block layouts follow the ggml `Q8_0` and `Q4_0` conventions, so blocks written here can be read by tools that use the same layout. The crate is `#![no_std]`.

## Installation

```toml
[dependencies]
tpt-simd-half = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports this crate.

## Quick start

```rust
use tpt_simd_half::{
    dequantize_q8_0, f16, f32_to_f16_slice, quantize_q8_0, BlockQ8_0,
};

// Slice conversion to f16.
let src = [1.0f32, 0.5, -2.0];
let mut half = [f16::ZERO; 3];
f32_to_f16_slice(&src, &mut half);
assert_eq!(half[2].to_f32(), -2.0);

// One Q8_0 block: 32 values, so the input length must be a multiple of 32.
let values = [1.0f32; 32];
let mut blocks = [BlockQ8_0 { scale: f16::ZERO, q: [0; 32] }; 1];
quantize_q8_0(&values, &mut blocks);
let mut restored = [0.0f32; 32];
dequantize_q8_0(&blocks, &mut restored);
assert!((restored[0] - 1.0).abs() < 0.01);
```

## API overview

| Item | Description |
|---|---|
| `f32_to_f16_slice(&[f32], &mut [f16])` | Convert, rounding to nearest |
| `f16_to_f32_slice(&[f16], &mut [f32])` | Widen; exact |
| `f32_to_bf16_slice(&[f32], &mut [bf16])` | Convert to bfloat16 |
| `bf16_to_f32_slice(&[bf16], &mut [f32])` | Widen; exact |
| `quantize_i8(&[f32], scale, &mut [i8])` | `round(x / scale)`, clamped to `-127..=127` |
| `dequantize_i8(&[i8], scale, &mut [f32])` | `q * scale` |
| `quantize_q8_0(&[f32], &mut [BlockQ8_0])` | `src.len()` must equal `32 * dst.len()` |
| `dequantize_q8_0(&[BlockQ8_0], &mut [f32])` | Inverse of `quantize_q8_0` |
| `quantize_q4_0(&[f32], &mut [BlockQ4_0])` | `src.len()` must equal `32 * dst.len()` |
| `dequantize_q4_0(&[BlockQ4_0], &mut [f32])` | Inverse of `quantize_q4_0` |
| `BLOCK` | Values per block (32) |

### Block layouts

- `BlockQ8_0 { scale: f16, q: [i8; 32] }`: `x = q * scale`, where `scale` is `max|x| / 127`.
- `BlockQ4_0 { scale: f16, qs: [u8; 16] }`: `x = (q - 8) * scale`. Element `i` is in the low nibble of `qs[i]` for `i < 16`, and element `i + 16` is in the high nibble of `qs[i]`.

## Feature flags

This crate has no optional features.

## Platform support and semantics

- Pure Rust on every target; no architecture-specific code.
- Conversions to `f16` and `bf16` round to nearest. Widening conversions are exact.
- Quantisation uses `libm::roundf`, so rounding is the same on every target.
- Length rules are checked up front. Mismatched slices, a block count that does not match the value count, or a non-positive or non-finite `scale` cause a **panic**.

## Example

Runnable example: [examples/half.rs](examples/half.rs). Run it with `cargo run -p tpt-simd-half --example half`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
