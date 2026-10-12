# tpt-simd-dct

Orthonormal DCT-II / DCT-III and MDCT / IMDCT for codec-sized blocks, with a precomputed cosine table.

## Overview

| Type | Transform | Input length | Output length |
|---|---|---|---|
| `Dct` (`forward`) | orthonormal DCT-II | N | N |
| `Dct` (`inverse`) | orthonormal DCT-III, the inverse | N | N |
| `Mdct` (`forward`) | MDCT | 2N | N |
| `Mdct` (`inverse`) | IMDCT | N | 2N |

`sine_window` provides the window used for analysis and synthesis.

Each transform is a matrix-vector product, so it costs O(N²) per block. The inner loops run over contiguous table rows and vectorise well, but this is not an FFT. Use it for block sizes up to a few thousand. For larger sizes, use an FFT-based crate.

The crate is `#![no_std]` (it uses `alloc` for the tables) and forbids `unsafe` code.

## Installation

```toml
[dependencies]
tpt-simd-dct = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports this crate.

## Quick start

```rust
use tpt_simd_dct::Dct;

// Orthonormal DCT-II of a constant block: all energy lands in coefficient 0.
let dct = Dct::new(4);
let mut coeffs = [0.0f32; 4];
dct.forward(&[1.0; 4], &mut coeffs);
assert!((coeffs[0] - 2.0).abs() < 1e-6);
assert!(coeffs[1..].iter().all(|c| c.abs() < 1e-6));

// The inverse recovers the input.
let mut back = [0.0f32; 4];
dct.inverse(&coeffs, &mut back);
assert!(back.iter().all(|x| (x - 1.0).abs() < 1e-5));
```

## Perfect reconstruction (MDCT)

Apply the same window (for example `sine_window`) at analysis and at synthesis, hop by N samples, and overlap-add the windowed IMDCT outputs. Samples covered by two frames are then reconstructed up to rounding. The example below shows the full loop.

## Platform support and semantics

- Buffer lengths must match the transform size; a mismatch causes a **panic**.
- The output is deterministic for a given build. Results can differ from a different summation order by normal rounding.

## Feature flags

This crate has no optional features.

## Example

Runnable example: [examples/dct.rs](examples/dct.rs), which includes the MDCT overlap-add loop. Run it with `cargo run -p tpt-simd-dct --example dct`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
