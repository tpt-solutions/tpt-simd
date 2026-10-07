# tpt-simd-window

Window function generation (Hamming, Hanning, Blackman, Kaiser) and SIMD window application.

## Overview

- Generators: each window has an allocation-free `*_window_into_f32(&mut [f32])`
  form, and (with the `alloc` or `std` feature) a `*_window_f32(size)` form
  returning a `Vec<f32>`.
- Application: `apply_window_f32` (slices), `apply_window_simd_f32` (slices of
  8-lane vectors) and the scalar reference `apply_window_scalar_f32`.
- Math helpers: a polynomial cosine (`cos_approx_f32`, `cos_approx_simd_f32`)
  and the modified Bessel function `I0` (`bessel_i0_f64`, `bessel_i0_f32`).

Window convention: all windows are symmetric (denominator `N - 1`, the form
used for FIR design and analysis), so `w[0] == w[N-1]`. Size `0` gives an empty
window and size `1` gives `[1.0]`.

Scalar vs SIMD: the cosine-sum windows evaluate 8 lanes at a time with a
plain multiply/add polynomial (no fused operations), which LLVM vectorises.
Dispatch is compile-time only (target features such as `-C target-cpu=native`);
there is no runtime CPU detection. See
[ADR 0003](../docs/adr/0003-runtime-dispatch-and-float-tolerance.md).

The crate is `#![no_std]`; the allocating generators need the `alloc` feature.

## Installation

```toml
[dependencies]
tpt-simd-window = "0.1.0"
# for the Vec-returning generators in a no_std crate:
# tpt-simd-window = { version = "0.1.0", features = ["alloc"] }
```

The umbrella crate `tpt-simd` re-exports the individual crates if you prefer a
single dependency.

## Quick start

```rust
use tpt_simd_window::{apply_window_f32, hanning_window_into_f32};

let mut w = [0.0f32; 8];
hanning_window_into_f32(&mut w);
assert!(w[0].abs() < 1e-5 && w[7].abs() < 1e-5); // symmetric, zero endpoints

let mut data = [1.0f32; 8];
apply_window_f32(&mut data, &w);
```

With the `alloc` (or `std`) feature:

```rust
let w: Vec<f32> = tpt_simd_window::kaiser_window_f32(64, 8.0);
assert_eq!(w.len(), 64);
```

## API overview

| Function | Description |
|----------|-------------|
| `hamming_window_into_f32(out)` | `0.54 - 0.46 cos(2 pi n/(N-1))` |
| `hanning_window_into_f32(out)` | `0.5 - 0.5 cos(2 pi n/(N-1))` |
| `blackman_window_into_f32(out)` | Classic Blackman (a = 0.16): `0.42 - 0.5 cos(2 pi n/(N-1)) + 0.08 cos(4 pi n/(N-1))` |
| `kaiser_window_into_f32(out, beta)` | Kaiser with shape `beta` (`0` is rectangular); panics if `beta` is negative, NaN or infinite |
| `hamming_window_f32(size)`, `hanning_window_f32(size)`, `blackman_window_f32(size)`, `kaiser_window_f32(size, beta)` | `Vec<f32>` versions (require `alloc`) |
| `apply_window_f32(data, window)` | `data[i] *= window[i]`, 8 lanes at a time with a scalar tail; bit-identical to the scalar reference |
| `apply_window_simd_f32(data, window)` | Same for slices of `Simd<f32, 8>` |
| `apply_window_scalar_f32(data, window)` | Scalar reference |
| `cos_approx_simd_f32(x)` / `cos_approx_f32(x)` | Polynomial cosine, max abs error `2e-6` for `abs(x) <= 1000` |
| `bessel_i0_f64(x)` / `bessel_i0_f32(x)` | Modified Bessel `I0` via power series in `f64` |

The apply functions panic if the slice lengths differ.

## Feature flags

- `default = []`: no features are enabled by default; the crate is `#![no_std]`.
- `alloc`: enables the `Vec<f32>`-returning generators (`*_window_f32(size)`).
- `std`: implies `alloc` and forwards to `tpt-simd-core/std`.
- `nightly`: forwards to `tpt-simd-core/nightly`.
- `scalar-only`: forwards to `tpt-simd-core/scalar-only`.

`no_std` behaviour: always `no_std`; only the `alloc`-gated functions need an
allocator.

## Platform and accuracy notes

- Targets: portable code for every target supported by `tpt-simd-core`; native
  vector instructions come from LLVM when target features are enabled at
  compile time. No `unsafe`.
- The cosine-sum windows use `cos_approx_simd_f32`, so they are accurate to
  about `3e-6` absolute versus an `f64` evaluation. Kaiser is evaluated in
  `f64` and rounded to `f32` once.
- `bessel_i0_f64` has a relative error of a few `f64` ulps for `abs(x) <= 700`;
  larger arguments overflow to `inf`. `cos_approx_*` error grows roughly
  linearly with `abs(x)` beyond 1000; it is not intended for huge arguments.

## Performance

Benchmarks live in `benches/` (generators vs a `libm` scalar baseline,
`cos_approx`, Bessel I0 and window application):

```sh
cargo bench -p tpt-simd-window
```

See also [docs/benchmarks.md](../docs/benchmarks.md).

## Related crates

- [`tpt-simd-core`](../tpt-simd-core): the `Simd` vector type; the only workspace dependency.
- [`tpt-simd-convolve`](../tpt-simd-convolve), [`tpt-simd-interpolate`](../tpt-simd-interpolate): sibling signal-processing crates.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
