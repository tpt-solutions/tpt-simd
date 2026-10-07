# tpt-simd-compare

SIMD lane-wise comparisons that produce masks, plus mask helpers and fused compare-and-count slice functions.

## Overview

`tpt-simd-compare` provides the six standard comparisons (`gt`, `lt`, `eq`, `ne`, `ge`, `le`) for the 256-bit vector shapes used across the workspace, each returning a `SimdMask` with one boolean per lane. A set of mask helpers (`mask_any`, `mask_all`, `mask_count`, `mask_to_bitmask`, `mask_from_bitmask`) and fused slice counters (`count_gt_i32`, ...) complete the crate.

Supported vector shapes: `i32x8`, `f32x8`, `i16x16`, `i8x32`. Integer comparisons are signed.

Scalar vs SIMD: there is a portable implementation (the `simd_*` methods of `tpt-simd-core`) and an explicit AVX2 implementation. They return identical results. Dispatch is compile-time (see `docs/adr`): the AVX2 path is compiled in only when `avx2` is enabled for the target, for example with `RUSTFLAGS="-C target-cpu=native"`. There is no runtime detection. The crate is `#![no_std]`.

To count how many elements of a slice satisfy a predicate, prefer the `count_*` functions: they never materialise a mask, whereas a per-vector `cmp_*` plus `mask_count` loop is much slower than a plain auto-vectorised scalar count.

## Installation

```toml
[dependencies]
tpt-simd-compare = "0.1.0"
```

The functions are also re-exported by the umbrella crate `tpt-simd`.

## Quick start

```rust
use tpt_simd_compare::*;
use tpt_simd_core::F32x8;

let a = F32x8::from_array([1.0, f32::NAN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
let b = F32x8::splat(3.0);
let m = cmp_gt_f32(a, b);
assert_eq!(mask_to_bitmask(m), 0b1111_1000);
assert!(mask_any(m) && !mask_all(m));
assert_eq!(mask_count(cmp_ne_f32(a, a)), 1); // only the NaN lane
```

Counting in a slice:

```rust
use tpt_simd_compare::count_gt_i32;

let data = [1, 5, 9, 2, 7];
assert_eq!(count_gt_i32(&data, 4), 3);
```

## API overview

Comparisons, each taking two vectors and returning a `SimdMask<T, N>`:

| Element type | Vector | Functions |
|---|---|---|
| `i32` | `I32x8` | `cmp_gt_i32`, `cmp_lt_i32`, `cmp_eq_i32`, `cmp_ne_i32`, `cmp_ge_i32`, `cmp_le_i32` |
| `f32` | `F32x8` | `cmp_gt_f32`, `cmp_lt_f32`, `cmp_eq_f32`, `cmp_ne_f32`, `cmp_ge_f32`, `cmp_le_f32` |
| `i16` | `I16x16` | `cmp_gt_i16`, `cmp_lt_i16`, `cmp_eq_i16`, `cmp_ne_i16`, `cmp_ge_i16`, `cmp_le_i16` |
| `i8` | `I8x32` | `cmp_gt_i8`, `cmp_lt_i8`, `cmp_eq_i8`, `cmp_ne_i8`, `cmp_ge_i8`, `cmp_le_i8` |

Other comparison:

| Function | Description |
|---|---|
| `cmp_unord_f32(a, b)` | True in lanes where either operand is NaN. |

Mask helpers (generic over element type `T` and lane count `N`):

| Function | Description |
|---|---|
| `mask_any(m)` | True if any lane is set. |
| `mask_all(m)` | True if every lane is set. |
| `mask_count(m)` | Number of set lanes. |
| `mask_to_bitmask(m)` | Bit `i` of the returned `u64` is lane `i`. |
| `mask_from_bitmask::<T, N>(bits)` | Inverse of `mask_to_bitmask`. |

Fused slice counters (`fn(data: &[T], rhs: T) -> usize`):

| Function | Description |
|---|---|
| `count_gt_i32`, `count_lt_i32`, `count_eq_i32` | Number of `i32` elements `x` with `x > rhs`, `x < rhs`, `x == rhs`. |
| `count_gt_f32`, `count_lt_f32`, `count_eq_f32` | Same for `f32`; a NaN element never matches. |

The crate also re-exports `SimdMask` from `tpt-simd-core`.

## Feature flags

The crate is `#![no_std]` in every configuration. The features only forward to `tpt-simd-core` and from there to `tpt-simd-vector`.

| Feature | Default | Effect |
|---|---|---|
| `std` | off | Forwards to `tpt-simd-core/std`; the vector backend then uses `std` float routines instead of `libm`. |
| `nightly` | off | Forwards to `tpt-simd-core/nightly` (reserved; no code in this workspace currently branches on it). |
| `scalar-only` | off | Forwards to `tpt-simd-core/scalar-only`. This crate's own AVX2 `cfg` does not test this feature, so the AVX2 comparisons are still used when `avx2` is enabled at compile time. |

## Platform and target support

- x86 / x86_64 with `avx2` enabled at compile time: `vpcmpgt*`/`vpcmpeq*` for integers and `vcmpps` for floats, with the result register stored straight into the mask's raw lanes (no packing). `mask_any`/`mask_all`/`mask_count`/`mask_to_bitmask` use `vmovmskps`/`vpmovmskb` and `popcnt` on the raw lanes. The `count_*` functions use a vector compare with per-lane counter accumulation.
- Every other target (including aarch64): portable implementation; no hand-written NEON code.
- Unsafe code is limited to loads and stores inside the AVX2 module, each with a `SAFETY` comment. All public functions are safe and never panic.

NaN semantics for `f32` follow IEEE 754 ordered comparisons and match Rust's scalar operators:

- `gt`, `lt`, `ge`, `le`, `eq` are false if either lane is NaN.
- `ne` is true if either lane is NaN (it is exactly `!eq`).
- `-0.0 == +0.0` is true.

Consequently `cmp_ge_f32(a, b)` is not the negation of `cmp_lt_f32(a, b)` when NaNs are present. Use `cmp_unord_f32` to find NaN lanes.

## Performance

Benchmarks live in `benches/compare.rs`; run `cargo bench -p tpt-simd-compare`, preferably with `RUSTFLAGS="-C target-cpu=native"`. Results for the workspace are collected in [../docs/benchmarks.md](../docs/benchmarks.md).

## Related crates

- `tpt-simd-core`: the `Simd` and `SimdMask` types (dependency).
- `tpt-simd-blend`: consumes masks to select between vectors, and offers fused `blend_gt_*` helpers.
- `tpt-simd`: umbrella crate re-exporting the workspace.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
