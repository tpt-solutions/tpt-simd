# tpt-simd-math

Vectorised `f32` `exp`, `ln`, `sin`, `cos`, `tanh` and `erf`: branch-free polynomial approximations with documented ULP error.

## Overview

Each function is a branch-free scalar kernel (range reduction plus polynomial, every conditional a value select) applied across 8 lanes. Only plain `+ - * /` are used: no per-lane `libm` calls and no `mul_add`. LLVM turns the 8-lane loops into straight vector code for whatever target features the build enables, so there is no hand-written intrinsic path and no `unsafe` (`#![forbid(unsafe_code)]`). Dispatch is compile-time (ADR 0001); build with `-C target-cpu=native` for AVX2 speed. The crate is `#![no_std]` and needs no allocator or `libm`.

Results are **bit-identical on every target and feature set** (no FMA contraction) and identical between the scalar, `Simd<f32, 8>` and slice entry points, including slice tails. Accuracy is a documented ULP bound rather than bit equality with `libm` or `std` (Tier 2 in ADR 0003).

## Installation

```toml
[dependencies]
tpt-simd-math = "0.1.0"
```

The `Simd` type is re-exported as `tpt_simd_math::Simd`. The umbrella crate `tpt-simd` also re-exports this crate.

## Quick start

```rust
use tpt_simd_math::{exp_f32, exp_f32x8, Simd};

let x = [0.0f32, 1.0, -1.0, 10.0, 0.5, 2.0, -20.0, 88.0, 3.0];
let mut y = [0.0f32; 9];
exp_f32(&x, &mut y);
assert_eq!(y[0], 1.0);
assert!((y[1] - core::f32::consts::E).abs() < 1e-6);

let v = exp_f32x8(Simd::splat(0.0));
assert_eq!(v.to_array(), [1.0; 8]);
```

## API overview

For each function `f` in `exp`, `ln`, `sin`, `cos`, `tanh`, `erf` there are four entry points:

| Entry point | Signature | Notes |
|---|---|---|
| `f_scalar` | `fn(f32) -> f32` | The shared single-lane kernel |
| `f_f32x8` | `fn(Simd<f32, 8>) -> Simd<f32, 8>` | Bit-identical to the scalar kernel per lane |
| `f_f32` | `fn(&[f32], &mut [f32])` | Panics if the slice lengths differ; tails handled |
| `f_f32_inplace` | `fn(&mut [f32])` | In-place variant |

For example `exp_scalar`, `exp_f32x8`, `exp_f32` and `exp_f32_inplace`, and likewise `ln_*`, `sin_*`, `cos_*`, `tanh_*` and `erf_*`.

## Feature flags

| Feature | Default | Effect |
|---|---|---|
| `std` | off | Enables `tpt-simd-core/std`. The crate itself stays `no_std`. |
| `nightly` | off | Forwards `tpt-simd-core/nightly` (reserved, currently a no-op in core). |
| `scalar-only` | off | Forwards `tpt-simd-core/scalar-only`. The kernels have a single portable implementation, so results and code are unchanged here. |

## Platform support and accuracy

Targets: any target supported by `tpt-simd-core`; the portable code auto-vectorises (SSE2 baseline, AVX2 with `-C target-cpu=native`, NEON on aarch64). There is no AVX-512 or NEON-specific code, and no `unsafe`.

Maximum ULP error measured, and the bound asserted by the test suite:

| Function | Range | Max ULP measured | Bound asserted |
|---|---|---|---|
| `exp` | `[-87.3, 88.72]` (normal results) | 0.97 | 1.1 |
| `exp` | `[-103.97, -87.3]` (subnormal results) | 0.75 | 1.0 |
| `ln` | all positive finite, incl. subnormals | 0.79 | 1.0 |
| `sin`, `cos` | `abs(x) <= 8192` | 2.43 | 3.0 |
| `sin`, `cos` | `8192 < abs(x) <= 100000` | 3.42 | 3.5 |
| `tanh` | all finite | 1.28 | 1.5 |
| `erf` | all finite | 2.64 | 3.0 |

Special values:

- NaN input is returned unchanged (payload and sign preserved).
- `exp`: `+inf` for `x > 88.7228`, gradual underflow to `+0` below about `-103.97`; `exp(-inf) = 0`.
- `ln`: `ln(+-0) = -inf`, NaN for every `x < 0` (including `-inf`), `ln(1) = 0` exactly, `ln(+inf) = +inf`.
- `sin`/`cos`: NaN for `abs(x) > 100000` (no Payne-Hanek reduction) and for infinities.
- `tanh` and `erf` are odd functions; the sign of zero follows the input, and both saturate to `+-1` for infinite input.

`tanh` and `erf` evaluate several polynomials and an `exp` and select among them, so their cost is the sum of all branches.

## Performance

Benchmarks live in `benches/math.rs` and compare against scalar `libm` and `std` loops; run them with `cargo bench -p tpt-simd-math`. See [docs/benchmarks.md](../docs/benchmarks.md) for recorded results.

## Related crates

- [`tpt-simd-core`](../tpt-simd-core): provides `Simd`.
- [`tpt-simd-rng`](../tpt-simd-rng): random number generation whose normal/uniform transforms also use documented approximations.
- [`tpt-simd-blas`](../tpt-simd-blas), [`tpt-simd-reduce`](../tpt-simd-reduce): other Tier 2 numeric kernel crates.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
