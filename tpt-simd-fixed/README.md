# tpt-simd-fixed

Type-safe SIMD fixed-point arithmetic: a `Fixed<T, INT_BITS, FRAC_BITS, N>` vector type with explicit wrapping and saturating operations and well-defined rounding.

## Overview

`Fixed<T, INT_BITS, FRAC_BITS, N>` wraps a `Simd<T, N>` of raw two's-complement integers interpreted as `raw / 2^FRAC_BITS`. Use it for DSP, audio, graphics and embedded-style code where fixed-point arithmetic is wanted across a whole vector, with the Q-format checked at compile time.

- `T` is `i32` or `i16` (sealed trait `FixedRepr`).
- `INT_BITS` includes the sign bit: `Fixed<i32, 16, 16, N>` is the classic 16.16 format with range `[-32768, 32768)`.
- `INT_BITS >= 1` and `INT_BITS + FRAC_BITS <= T::BITS` are checked at compile time (a bad format is a compile error once any constructor or operation is instantiated). If the sum is less than `T::BITS` the format is a narrow format stored in a wider lane: saturating operations clamp to the format range, wrapping operations wrap at the storage width.
- Operators `+ - * / -x` (and their assign forms) wrap at the storage width ([ADR 0002](../docs/adr/0002-api-surface-and-policies.md)); explicit `wrapping_*` and `saturating_*` variants exist for every operation.
- Rounding: `mul`, `div` and `convert` round to nearest with ties away from zero; `mul_trunc` truncates (floors); `from_f32` rounds to nearest with ties to even.

Scalar versus SIMD: a portable implementation (lane loops with 64-bit intermediates) is the reference. When compiled with `target_feature = "avx2"` on x86_64 (and without `scalar-only`), `i32` multiplies use `_mm256_mul_epi32` on even/odd lanes; results are bit-identical to the portable path. Dispatch is compile-time (see [ADR 0001](../docs/adr/0001-backend-and-dispatch.md)); build with `-C target-cpu=native` for the AVX2 path.

`no_std`: the crate is `#![no_std]`. `unsafe` is forbidden unless the AVX2 backend is compiled in, where it is limited to the intrinsic multiply.

## Installation

```toml
[dependencies]
tpt-simd-fixed = "0.1.0"
```

The same functionality is available through the umbrella crate `tpt-simd`.

## Quick start

```rust
use tpt_simd_core::F32x8;
use tpt_simd_fixed::Fixed;

let a: Fixed<i32, 16, 16, 8> = Fixed::from_f32(F32x8::splat(1.5));
let b: Fixed<i32, 16, 16, 8> = Fixed::from_f32(F32x8::splat(2.5));
let result = a * b; // 1.5 * 2.5 = 3.75 with correct rounding
assert_eq!(result.to_f32(), F32x8::splat(3.75));
```

Saturating and checked variants:

```rust
use tpt_simd_fixed::Fixed;

type Q = Fixed<i32, 16, 16, 8>;
let big = Q::from_raw(tpt_simd_core::I32x8::splat(1 << 30)); // 16384.0
assert_eq!(big.saturating_mul(big), Q::max_value());
assert!(Q::one().checked_div(Q::splat_raw(0)).is_none());
```

## API overview

Type and trait

| Item | Description |
|---|---|
| `Fixed<T, INT_BITS, FRAC_BITS, N>` | Fixed-point vector wrapping `Simd<T, N>`; `Clone`, `Copy`, `PartialEq`, `Debug` |
| `FixedRepr` | Sealed trait for the raw storage type (`i32`, `i16`) |

Construction and conversion

| Method | Description |
|---|---|
| `from_raw`, `raw` | Wrap / unwrap the raw integer vector (no range check) |
| `splat_raw`, `splat_f32` | Broadcast a raw value / an `f32` |
| `from_f32`, `to_f32` | Convert from `f32` lanes (ties to even, saturating, NaN becomes 0) / to `f32` lanes |
| `min_value`, `max_value`, `one`, `epsilon` | Format constants |
| `clamp_to_format` | Clamp raw lanes into the format range (no-op for full-width formats) |
| `convert::<I2, F2>` | Convert to another Q format of the same storage type (saturating, ties away from zero) |

Arithmetic

| Method | Description |
|---|---|
| `wrapping_add`, `wrapping_sub`, `wrapping_mul`, `wrapping_neg` | Wrapping at storage width (what the operators use) |
| `saturating_add`, `saturating_sub`, `saturating_mul`, `saturating_neg`, `saturating_abs`, `saturating_div` | Clamp to the format range |
| `mul_trunc` | Truncating multiply (floor), wrapping |
| `div`, `checked_div` | Division, rounded to nearest ties away from zero; `div` (and `/`) panics on a zero divisor lane, `checked_div` returns `None` |
| `abs` | Wrapping absolute value |
| `min`, `max`, `clamp` | Lane-wise ordering operations |

Operators: `Add`, `Sub`, `Mul`, `Div`, `Neg` and the corresponding `*Assign` traits (all wrapping).

## Feature flags

| Feature | Effect |
|---|---|
| `default` | empty |
| `std` | Forwards to `tpt-simd-core/std` |
| `nightly` | Forwards to `tpt-simd-core/nightly` |
| `scalar-only` | Forwards to `tpt-simd-core/scalar-only`; also disables the AVX2 backend |

## Platform support and accuracy

- All targets use the portable implementation; x86_64 with AVX2 enabled at compile time additionally uses an intrinsic `i32` multiply. aarch64 NEON and AVX-512 have no dedicated path here (LLVM auto-vectorises the portable code).
- Results are bit-identical between backends and targets: all arithmetic is integer, with 64-bit intermediates for multiply, divide and convert.
- Overflow: operators wrap; use the `saturating_*` methods to clamp. `from_f32` saturates and maps NaN to 0. Division by zero panics in `div` and `/`; `saturating_div` maps `x / 0` to `max_value()`, `min_value()` or 0.
- `to_f32` is exact when the raw value fits in 24 bits, otherwise rounded by the `f32` conversion.

## Performance

Benchmarks against manual scalar fixed-point loops: `cargo bench -p tpt-simd-fixed`. See also [../docs/benchmarks.md](../docs/benchmarks.md).

## Related crates

Depends on `tpt-simd-core`. Siblings: `tpt-simd-mul`, `tpt-simd-saturate`. Umbrella: `tpt-simd`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
