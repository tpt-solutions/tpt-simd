# tpt-simd-horizontal

Horizontal reductions (sum, product, min, max) that collapse the lanes of one SIMD vector into a scalar, with a fixed, target-independent reduction order.

## Overview

A horizontal reduction combines the lanes of a single vector, as opposed to a vertical (lane-wise) operation between two vectors. This crate provides:

- Fixed-width functions for the common shapes (`horizontal_sum_f32` on `F32x8`, `horizontal_max_i16` on `I16x16`, ...).
- Generic `reduce_sum`, `reduce_product`, `reduce_min` and `reduce_max` that work on any element type and any lane count.

All reductions use one **halving tree**: lane `i` is combined with lane `i + ceil(N/2)`, leaving `ceil(N/2)` partial results, and the step repeats until one value remains. For 8 lanes the order is `((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))`. Because the tree is part of the contract, float results are bit-for-bit identical on every target and feature set (Tier 1 in ADR 0003). Integer sums wrap, and integer min/max are order independent.

Scalar-vs-SIMD: there is a single portable implementation written with `tpt_simd_core::Simd`; LLVM lowers the fixed tree to shuffles and vector ops for the target. There is no hand-written intrinsic path and no `unsafe` (`#![forbid(unsafe_code)]`). Dispatch is compile-time (ADR 0001); there is nothing to dispatch at run time. The crate is `#![no_std]`.

## Installation

```toml
[dependencies]
tpt-simd-horizontal = "0.1.0"
tpt-simd-core = "0.1.0"   # for Simd and the F32x8 / I16x16 / ... aliases
```

The umbrella crate `tpt-simd` re-exports this crate if you prefer a single dependency.

## Quick start

```rust
use tpt_simd_core::{F32x8, I32x8, Simd};
use tpt_simd_horizontal::{
    horizontal_min_f32, horizontal_sum_f32, horizontal_sum_i32, reduce_sum,
};

assert_eq!(horizontal_sum_i32(I32x8::from_array([1, 2, 3, 4, 5, 6, 7, 8])), 36);
assert_eq!(horizontal_sum_i32(I32x8::splat(i32::MAX)), -8); // integer sums wrap
assert_eq!(horizontal_sum_f32(F32x8::from_array([1.0; 8])), 8.0);

// NaN lanes are ignored by min/max.
let v = F32x8::from_array([f32::NAN, 3.0, 2.0, f32::NAN, 5.0, 6.0, 7.0, 8.0]);
assert_eq!(horizontal_min_f32(v), 2.0);

// Generic over element type and lane count.
assert_eq!(reduce_sum(Simd::<i32, 4>::from_array([1, 2, 3, 4])), 10);
```

## API overview

Fixed-width functions:

| Function | Input | Result |
|---|---|---|
| `horizontal_sum_i32` | `Simd<i32, 8>` | wrapping `i32` sum |
| `horizontal_sum_f32` | `Simd<f32, 8>` | `f32` sum in halving-tree order |
| `horizontal_sum_i16` | `Simd<i16, 16>` | widening sum into `i32` (cannot overflow) |
| `horizontal_sum_u8` | `Simd<u8, 32>` | widening sum into `u32` |
| `horizontal_sum_i8` | `Simd<i8, 32>` | widening sum into `i32` |
| `horizontal_max_i16` / `horizontal_min_i16` | `Simd<i16, 16>` | `i16` |
| `horizontal_max_i32` / `horizontal_min_i32` | `Simd<i32, 8>` | `i32` |
| `horizontal_max_f32` / `horizontal_min_f32` | `Simd<f32, 8>` | `f32`, NaN lanes ignored |
| `horizontal_product_f32` | `Simd<f32, 8>` | `f32` product in halving-tree order |

Generic functions (any `T: SimdElement`, any `N > 0`; they panic if `N == 0`):

| Function | Result |
|---|---|
| `reduce_sum` | sum of all lanes (integers wrap) |
| `reduce_product` | product of all lanes (integers wrap) |
| `reduce_min` / `reduce_max` | extremum of all lanes (float NaN lanes ignored) |

## Feature flags

| Feature | Default | Effect |
|---|---|---|
| `std` | off | Enables `tpt-simd-core/std`. The crate itself stays `no_std`. |
| `nightly` | off | Forwards `tpt-simd-core/nightly` (reserved for a `core::simd` backend; currently a no-op in core). |
| `scalar-only` | off | Forwards `tpt-simd-core/scalar-only`. This crate has a single portable implementation, so it changes nothing here. |

## Platform support and semantics

- Targets: any target supported by `tpt-simd-core` (x86/x86_64, aarch64 and others); the code is portable and LLVM picks the instructions. There is no AVX-512 or NEON-specific code in this crate.
- NaN policy: float min/max ignore NaN lanes; an all-NaN vector gives NaN. The sign of a zero result is unspecified (but deterministic) if both `+0.0` and `-0.0` are present. `horizontal_sum_f32` and `horizontal_product_f32` propagate NaN and infinities as IEEE arithmetic does; product overflow gives `inf`.
- Float sums and products are reassociated relative to a left-to-right loop, so they can differ from a naive scalar loop by normal rounding, but never between targets.
- Safety: no `unsafe` code.

## Performance

Benchmarks live in `benches/horizontal.rs`; run them with `cargo bench -p tpt-simd-horizontal`. See [docs/benchmarks.md](../docs/benchmarks.md) for recorded results and methodology. For best code generation build with `-C target-cpu=native` (compile-time dispatch).

## Related crates

- [`tpt-simd-core`](../tpt-simd-core): provides `Simd`, `SimdElement` and the vector aliases used in every signature.
- [`tpt-simd-reduce`](../tpt-simd-reduce): slice-level reductions built on the halving tree of this crate.
- [`tpt-simd-dot`](../tpt-simd-dot): dot products that finish with `horizontal_sum_f32`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
