# tpt-simd-audio

Audio kernels on `f32` samples: mixing, interleaving, level meters, FIR and biquad filtering, and windowed-sinc resampling.

## Overview

| Function / type | What it does |
|---|---|
| `mix_into` | `dst += gain * src` |
| `interleave` / `deinterleave` | planar channels to interleaved samples, and back |
| `peak` / `rms` | level of a buffer |
| `fir` | causal direct-form FIR filter, zero history before the first sample |
| `Biquad` | RBJ low-pass and high-pass sections, direct form II transposed, `f64` state |
| `resample_sinc` | windowed-sinc (Blackman) sample-rate conversion |

The crate is `#![no_std]` and forbids `unsafe` code. The only dependency is `libm`, for trigonometry.

## Installation

```toml
[dependencies]
tpt-simd-audio = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports this crate.

## Quick start

```rust
use tpt_simd_audio::{Biquad, deinterleave, interleave, mix_into, peak};

// Mix a source into a bus at half gain.
let mut bus = [0.0f32; 4];
mix_into(&mut bus, &[1.0, 1.0, 1.0, 1.0], 0.5);
assert_eq!(bus, [0.5; 4]);

// Stereo: planar <-> interleaved.
let mut packed = [0.0f32; 4];
interleave(&[&[1.0, 2.0], &[-1.0, -2.0]], &mut packed);
assert_eq!(packed, [1.0, -1.0, 2.0, -2.0]);

// Low-pass biquad at 1 kHz for a 48 kHz stream, applied in place.
let mut x = [1.0f32, 0.0, 0.0, 0.0];
let mut lp = Biquad::lowpass(48_000.0, 1_000.0, core::f64::consts::FRAC_1_SQRT_2);
lp.process(&mut x);
assert!(peak(&x) > 0.0);
```

## API overview

| Item | Signature (abbreviated) |
|---|---|
| `mix_into` | `(dst: &mut [f32], src: &[f32], gain: f32)` |
| `interleave` | `(channels: &[&[f32]], out: &mut [f32])` |
| `deinterleave` | `(input: &[f32], channels: &mut [&mut [f32]])` |
| `peak`, `rms` | `(x: &[f32]) -> f32` |
| `fir` | `(taps: &[f32], x: &[f32], y: &mut [f32])` |
| `Biquad::lowpass`, `Biquad::highpass` | `(sample_rate: f64, f0: f64, q: f64) -> Biquad` |
| `Biquad::process`, `tick`, `reset` | filter a block, one sample, or clear the state |
| `resample_sinc` | `(input, in_rate, out_rate, half_width, out) -> usize` |

## Platform support and semantics

- Pure Rust on every target; no architecture-specific code.
- Length rules are checked up front. Mismatched slice lengths cause a **panic**.
- `resample_sinc` treats samples outside the input as zero. When downsampling, the cutoff drops to the new Nyquist frequency to limit aliasing. `half_width` trades sharpness for cost (typically 16 to 64).
- `Biquad` keeps its state in `f64`, which keeps long low-frequency filters stable.

## Feature flags

This crate has no optional features.

## Example

Runnable example: [examples/audio.rs](examples/audio.rs). Run it with `cargo run -p tpt-simd-audio --example audio`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
