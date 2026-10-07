# tpt-simd-butterfly

In-place radix-2 butterfly operations on SIMD vectors, for FFT and DCT kernels.

## Overview

A radix-2 butterfly maps `(a, b)` to `(a + b, a - b)`. This crate provides that
operation for `f32` (8 lanes), `i16` (16 lanes, wrapping or saturating), any
lane type and width, and split-layout complex `f32` vectors, plus the
twiddle-factor forms used by decimation-in-time FFTs.

Scalar vs SIMD: the functions are written as plain lane-wise vector `+`/`-`
(and `*` for twiddles) on `tpt_simd_core::Simd`, which LLVM lowers to one vector
add and one vector sub per register. There are no hand-written intrinsics and
no `unsafe` code (`#![forbid(unsafe_code)]`). Dispatch is compile-time only: what
you get depends on the target features you compile with (for example
`-C target-cpu=native`); there is no runtime CPU detection here. See
[ADR 0003](../docs/adr/0003-runtime-dispatch-and-float-tolerance.md).

The crate is `#![no_std]` and needs no allocator.

Deviation from the original spec: the spec's `butterfly_with_twiddle_f32` took
real vectors with a complex twiddle, which is ambiguous. The crate offers two
unambiguous forms instead: `butterfly_with_twiddle_f32` (real twiddle) and
`butterfly_with_twiddle_complex_f32` (complex twiddle).

## Installation

```toml
[dependencies]
tpt-simd-butterfly = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports the individual crates if you prefer a
single dependency.

## Quick start

```rust
use tpt_simd_core::F32x8;
use tpt_simd_butterfly::butterfly_f32;

let mut a = F32x8::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
let mut b = F32x8::from_array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
butterfly_f32(&mut a, &mut b);
assert_eq!(a.to_array(), [9.0; 8]);
assert_eq!(b.to_array(), [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]);
```

Complex butterfly with a twiddle factor (`(a + b*w, a - b*w)`):

```rust
use tpt_simd_core::{ComplexSimd, F32x8};
use tpt_simd_butterfly::butterfly_with_twiddle_complex_f32;

let mut a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(0.0));
let mut b = ComplexSimd::new(F32x8::splat(0.0), F32x8::splat(1.0)); // i
let w = ComplexSimd::new(F32x8::splat(0.0), F32x8::splat(1.0));     // i
butterfly_with_twiddle_complex_f32(&mut a, &mut b, w);              // b*w = -1
assert_eq!(a.real, F32x8::splat(0.0));
assert_eq!(b.real, F32x8::splat(2.0));
```

## API overview

All functions work in place on `&mut` vectors and never panic.

| Function | Description |
|----------|-------------|
| `butterfly_f32` | `(a, b) <- (a + b, a - b)` on `Simd<f32, 8>` |
| `butterfly_i16` | Same on `Simd<i16, 16>` with wrapping arithmetic |
| `butterfly_i16_saturating` | Same on `Simd<i16, 16>` with saturating arithmetic |
| `butterfly_generic<T: SimdElement, N>` | Same for any lane type and width (integers wrap, floats follow IEEE) |
| `butterfly_generic_saturating<T: SimdInt, N>` | Saturating version for integer lanes of any width |
| `butterfly_complex_f32` | Plain butterfly on `ComplexSimd<f32, 8>` (split real/imag layout) |
| `butterfly_with_twiddle_f32` | Real twiddle: `b' = b * w; (a + b', a - b')` on `Simd<f32, 8>` (unfused multiply) |
| `butterfly_with_twiddle_complex_f32` | Complex twiddle (decimation-in-time): `t = b * w; (a + t, a - t)` on `ComplexSimd<f32, 8>` |

The complex product in the twiddle form uses the unfused `ComplexSimd::mul` from
`tpt-simd-core`, so results are identical on every target. For a fused
(FMA-rounded) FFT butterfly see `tpt-simd-complex`.

## Feature flags

- `default = []`: no features are enabled by default; the crate is `#![no_std]`.
- `std`: forwards to `tpt-simd-core/std`.
- `nightly`: forwards to `tpt-simd-core/nightly`.
- `scalar-only`: forwards to `tpt-simd-core/scalar-only` (core uses scalar implementations only, no SIMD).

`no_std` behaviour: the crate is always `no_std`; the `std` feature only links
`std` for the crate's own tests and forwards the feature to core.

## Platform and accuracy notes

- Targets: portable code. Any target supported by `tpt-simd-core` works
  (native vector instructions come from LLVM when the corresponding target
  features are enabled at compile time; otherwise scalar code).
- No `unsafe`.
- Integer butterflies wrap (`butterfly_i16`, `butterfly_generic`) or saturate
  (`*_saturating`); floating-point follows IEEE-754 (NaN and infinity propagate).
- `butterfly_with_twiddle_f32` deliberately does not use FMA so results do not
  depend on hardware FMA availability.

## Performance

Benchmarks live in `benches/` (scalar reference vs SIMD for `f32`, `i16` and the
complex twiddle butterfly):

```sh
cargo bench -p tpt-simd-butterfly
```

See also [docs/benchmarks.md](../docs/benchmarks.md).

## Related crates

- [`tpt-simd-core`](../tpt-simd-core): vector types (`Simd`, `ComplexSimd`) and traits; the only dependency.
- [`tpt-simd-complex`](../tpt-simd-complex): fused complex helpers and an FMA-rounded `fft_butterfly`.
- [`tpt-simd-saturate`](../tpt-simd-saturate), [`tpt-simd-fixed`](../tpt-simd-fixed): sibling crates for saturating and fixed-point arithmetic.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
