# tpt-simd-vector

Stable, array-backed `Simd<T, N>` and `SimdMask<T, N>` types: the backend every other tpt-simd crate builds on.

## Overview

`tpt-simd-vector` provides a portable fixed-width vector type without nightly features. `Simd<T, N>` is a newtype over `[T; N]` (`pub struct Simd<T, const N: usize>(pub [T; N])`); every operation is a fixed-length lane loop that LLVM can auto-vectorise to SSE/AVX/AVX-512/NEON when the build enables those features. The crate is `#![no_std]` and `#![forbid(unsafe_code)]`.

Scalar versus SIMD: there is no separate scalar path in this crate. The scalar lane operations (`SimdElement`, `SimdInt`, `SimdFloat`) are the reference semantics, and the vector type applies them lane by lane. Dispatch is compile-time only (`cfg(target_feature)` in the crates that use intrinsics; see `docs/adr/0001-backend-and-dispatch.md` and `docs/adr/0003-runtime-dispatch-and-float-tolerance.md`). Build with `RUSTFLAGS="-C target-cpu=native"` to let LLVM use wider registers.

Semantics shared by every crate in the workspace:

- Integer `+ - *`, `/` and negation wrap; use `sat_add` / `sat_sub` / `sat_mul` for saturation.
- Shifts by `>= BITS` yield `0` (or sign fill for the arithmetic right shift).
- Float `min` / `max` ignore NaN operands; comparisons with NaN are false (except `simd_ne`).
- Float math (`sqrt`, rounding, `mul_add`, `atan2`) is exactly specified, so results are target-independent. It uses `libm`, or the equivalent `std` intrinsics when the `std` feature is on; the two give identical results.

Lane types: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`.

## Installation

```toml
[dependencies]
tpt-simd-vector = "0.1.0"
```

Most users want the umbrella crate `tpt-simd` (which re-exports `Simd` and `SimdMask` through `tpt-simd-core`) rather than this crate directly.

## Quick start

```rust
use tpt_simd_vector::Simd;

let a = Simd::<f32, 4>::from_array([1.0, 2.0, 3.0, 4.0]);
let c = a * Simd::splat(2.0) + Simd::splat(1.0);
assert_eq!(c.to_array(), [3.0, 5.0, 7.0, 9.0]);

// Comparisons produce a SimdMask; select picks per lane.
let m = c.simd_gt(Simd::splat(4.0));
assert_eq!(m.count(), 3);
let s = m.select(c, Simd::zero());
assert_eq!(s.to_array(), [0.0, 5.0, 7.0, 9.0]);

// Integer arithmetic wraps; sat_* saturates.
let x = Simd::<i16, 8>::splat(i16::MAX);
assert_eq!((x + Simd::splat(1))[0], i16::MIN);
assert_eq!(x.sat_add(Simd::splat(1))[0], i16::MAX);
```

## API overview

### `Simd<T, N>`

| Group | Items |
| --- | --- |
| Construction | `splat`, `zero`, `from_array`, `from_fn`, `from_slice`, `from_slice_or`, `Default`, `From<[T; N]>` |
| Access | `to_array`, `as_array`, `as_mut_array`, `Index` / `IndexMut`, `copy_to_slice`, `store_partial`, `LANES` |
| Lane-wise closures | `map`, `zip_with`, `cast::<U>` (`as`-style, via `LaneCast`) |
| Arithmetic operators | `+ - * /` and unary `-` (all lane types), with `*Assign` forms |
| Bitwise operators (integers) | `&`, `\|`, `^`, `!`, `<< u32`, `>> u32`, with `*Assign` forms for the binary ones |
| Min / max | `min`, `max`, `clamp`, `abs` |
| Shuffles | `reverse`, `swizzle` |
| Comparisons (return `SimdMask`) | `simd_eq`, `simd_ne`, `simd_lt`, `simd_le`, `simd_gt`, `simd_ge`, `is_nan` (floats) |
| Integer-only | `sat_add`, `sat_sub`, `sat_mul`, `shl_scalar`, `shr_scalar`, `shr_logical`, `rotl` |
| Float-only | `sqrt`, `floor`, `ceil`, `round`, `round_ties_even`, `trunc`, `mul_add`, `atan2` |

### `SimdMask<T, N>`

`from_array` / `to_array` (`[bool; N]`), `from_raw` / `to_raw`, `splat`, `test`, `set`, `any`, `all`, `count`, `to_bitmask` / `from_bitmask` (at most 64 lanes; panics otherwise), `select`, and the `!` operator.

### Lane traits

| Trait | Purpose |
| --- | --- |
| `SimdElement` | Scalar lane type: constants (`ZERO`, `ONE`, `MIN`, `MAX`, `BITS`, `IS_FLOAT`, `IS_SIGNED`) and `lane_add/sub/mul/div/neg/min/max/abs` |
| `SimdInt` | Integer lanes: bitwise ops, saturating arithmetic, shifts, rotate |
| `SimdFloat` | Float lanes: `lane_sqrt`, rounding, `lane_mul_add`, `lane_is_nan`, `lane_atan2` |
| `LaneCast<U>` | `as`-style numeric conversion between any two lane types |
| `MaskLane` | The signed integer type used for each mask lane (all bits set = true) |

## Feature flags

| Feature | Default | Effect |
| --- | --- | --- |
| `std` | no | Float lane operations call `std` intrinsics instead of `libm` (results are identical). The crate stays `no_std` without it. |
| `nightly` | no | Reserved for a `core::simd` backend; currently has no effect. |
| `scalar-only` | no | Reserved; currently has no effect (the vector type is already plain lane loops). |

## Platform and safety notes

- Works on every target Rust supports; speed comes from LLVM lowering the lane loops, so x86 SSE2/AVX2/AVX-512 and aarch64 NEON are used when the build allows. No intrinsics and no `unsafe` appear in this crate.
- Panics: `from_slice` and `copy_to_slice` need a slice of at least `N` elements; `swizzle` and `test` / `set` indices must be in range; `to_bitmask` / `from_bitmask` require `N <= 64`. See the `# Panics` sections in the rustdoc.
- NaN, overflow and shift semantics are listed in the overview above.

## Related crates

- `tpt-simd-core`: re-exports this crate's types and adds aliases (`F32x8`, ...), `ComplexSimd` and feature detection.
- `tpt-simd`: umbrella crate re-exporting every tpt-simd crate.
- Every kernel crate in the workspace (`tpt-simd-dot`, `tpt-simd-blas`, ...) builds on this backend, via `tpt-simd-core`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
