# tpt-simd-interpolate

SIMD interpolation kernels: linear, Catmull-Rom cubic, Lanczos, and sinc.

## Overview

Eight-lane (`Simd<f32, 8>`) interpolation primitives with scalar references:

- Linear and cubic (Catmull-Rom) use FMA in a fixed order and are bit-identical
  between the SIMD and scalar versions.
- Lanczos uses the polynomial sinc approximation, so the SIMD result matches
  the scalar (`libm`) reference to a small tolerance (about `1e-5` relative to
  the sample magnitude) rather than bit-for-bit.

Dispatch is compile-time only: the kernels are portable lane code that LLVM
vectorises for the target features you compile with (for example
`-C target-cpu=native`); there is no runtime CPU detection. See
[ADR 0003](../docs/adr/0003-runtime-dispatch-and-float-tolerance.md).

The crate is `#![no_std]`, allocation-free and contains no `unsafe` code of its
own.

## Installation

```toml
[dependencies]
tpt-simd-interpolate = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports the individual crates if you prefer a
single dependency.

## Quick start

```rust
use tpt_simd_core::Simd;
use tpt_simd_interpolate::{interpolate_cubic_f32, interpolate_linear_f32, interpolate_linear_simd_f32};

assert_eq!(interpolate_linear_f32(0.0, 10.0, 0.25), 2.5);

let a = Simd::<f32, 8>::splat(0.0);
let b = Simd::<f32, 8>::splat(10.0);
let t = Simd::<f32, 8>::splat(0.5);
assert_eq!(interpolate_linear_simd_f32(a, b, t).to_array(), [5.0; 8]);

// Catmull-Rom on collinear points 0, 1, 2, 3 interpolates linearly between
// the middle two.
let p = |v: f32| Simd::<f32, 8>::splat(v);
let r = interpolate_cubic_f32(p(0.0), p(1.0), p(2.0), p(3.0), p(0.5));
assert_eq!(r.to_array(), [1.5; 8]);
```

## API overview

| Function | Description |
|----------|-------------|
| `interpolate_linear_f32(a, b, t)` | Scalar `a + t * (b - a)` as one fused multiply-add; `t` outside `[0, 1]` extrapolates |
| `interpolate_linear_simd_f32(a, b, t)` | 8-lane version, bit-identical per lane |
| `interpolate_cubic_scalar_f32(a, b, c, d, t)` | Scalar Catmull-Rom cubic between `b` (`t = 0`) and `c` (`t = 1`), with `a` and `d` as outer neighbours |
| `interpolate_cubic_f32(a, b, c, d, t)` | 8-lane Catmull-Rom, bit-identical to the scalar version |
| `sinc_f32(x)` | Normalised sinc `sin(pi x) / (pi x)` using `libm::sinf`; `1` at `x == 0` |
| `sinc_simd_f32(x)` | 8-lane sinc with polynomial approximation; max abs error `<= 2e-6` for `abs(x) <= 64` |
| `interpolate_lanczos_scalar_f32(samples, t, a)` | Scalar Lanczos-`a` interpolation over `2a` samples; weights normalised to sum to one |
| `interpolate_lanczos_f32(samples, t, a)` | 8-lane Lanczos-`a` over `2a` vectors; `t = 0` returns `samples[a - 1]` |

Lanczos sample layout: `samples[i]` sits at position `i - (a - 1)` relative to
the sample just left of the interpolation point; `t` is the fractional offset
past that sample. Both Lanczos functions panic if `a == 0` or
`samples.len() != 2 * a`.

## Feature flags

- `default = []`: no features are enabled by default; the crate is `#![no_std]`.
- `std`: forwards to `tpt-simd-core/std`.
- `nightly`: forwards to `tpt-simd-core/nightly`.
- `scalar-only`: forwards to `tpt-simd-core/scalar-only` (core uses scalar implementations only, no SIMD).

`no_std` behaviour: the crate is always `no_std`; the `std` feature only links
`std` for the crate's own tests and forwards the feature to core.

## Platform and accuracy notes

- Targets: portable code for every target supported by `tpt-simd-core`; native
  vector instructions come from LLVM when target features are enabled at
  compile time. Scalar paths use `libm`.
- No `unsafe`.
- `sinc_simd_f32`: range reduction by magic-number rounding plus an odd
  degree-11 polynomial; the error grows slowly for `abs(x) > 64`. Lanes with
  `abs(x) < 1e-6` return `1`.
- NaN and infinity inputs propagate through the arithmetic; no special
  handling is provided.

## Performance

Benchmarks live in `benches/` (SIMD vs eight scalar calls for linear, cubic and
Lanczos-3):

```sh
cargo bench -p tpt-simd-interpolate
```

See also [docs/benchmarks.md](../docs/benchmarks.md).

## Related crates

- [`tpt-simd-core`](../tpt-simd-core): the `Simd` vector type; the only workspace dependency.
- [`tpt-simd-window`](../tpt-simd-window), [`tpt-simd-convolve`](../tpt-simd-convolve): sibling signal-processing crates.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
