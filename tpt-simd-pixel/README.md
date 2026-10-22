# tpt-simd-pixel

Pixel kernels: channel layout changes, colour conversion, premultiplied alpha, unit-range float conversion and histograms.

## Overview

| Function | What it does |
|---|---|
| `rgb8_to_rgba8` / `rgba8_to_rgb8` | add an alpha channel with a fixed value, or drop it |
| `planar_to_interleaved_rgb8` / `interleaved_to_planar_rgb8` | three 8-bit planes and packed RGB |
| `planar_to_interleaved_rgba8` / `interleaved_to_planar_rgba8` | four 8-bit planes and packed RGBA |
| `u8_to_unit_f32` / `unit_f32_to_u8` | `[0, 255]` to `[0, 1]` and back, with rounding |
| `premultiply_rgba8` | colour times alpha, in place, rounded |
| `rgb_to_hsv` / `hsv_to_rgb` | `f32`; hue in degrees `[0, 360)` |
| `histogram_u8` | 256-bin count of a byte buffer |

Image resizing, YUV conversion and blur are not included; those are separate concerns.

The crate is `#![no_std]` and forbids `unsafe` code.

## Installation

```toml
[dependencies]
tpt-simd-pixel = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports this crate.

## Quick start

```rust
use tpt_simd_pixel::{histogram_u8, hsv_to_rgb, rgb8_to_rgba8, rgb_to_hsv, u8_to_unit_f32};

// Add an opaque alpha channel to RGB pixels.
let rgb = [10u8, 20, 30, 40, 50, 60];
let mut rgba = [0u8; 8];
rgb8_to_rgba8(&rgb, &mut rgba, 255);
assert_eq!(rgba, [10, 20, 30, 255, 40, 50, 60, 255]);

// Normalise bytes to [0, 1] for float pipelines.
let mut unit = [0.0f32; 2];
u8_to_unit_f32(&[0, 255], &mut unit);
assert_eq!(unit, [0.0, 1.0]);

// Pure red is hue 0, full saturation, full value.
assert_eq!(rgb_to_hsv(1.0, 0.0, 0.0), (0.0, 1.0, 1.0));
assert_eq!(hsv_to_rgb(120.0, 1.0, 1.0), (0.0, 1.0, 0.0));

// Histogram of a tiny buffer.
let mut bins = [0u32; 256];
histogram_u8(&[0, 0, 255, 7], &mut bins);
assert_eq!((bins[0], bins[7], bins[255]), (2, 1, 1));
```

## Platform support and semantics

- Length rules are checked up front. Mismatched buffers cause a **panic**.
- All integer conversions are exact or rounded as documented; results are identical on every target.

## Feature flags

This crate has no optional features.

## Example

Runnable example: [examples/pixel.rs](examples/pixel.rs). Run it with `cargo run -p tpt-simd-pixel --example pixel`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
