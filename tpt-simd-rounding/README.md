# tpt-simd-rounding

SIMD rounding operations (round, floor, ceil, trunc, ties-to-even, and float-to-integer conversions) on eight `f32` lanes.

## Overview

`tpt-simd-rounding` rounds an `F32x8` with a clearly specified tie rule, either staying in floating point or converting to `i32x8`. It needs no `libm` at runtime (`libm` is only a dev-dependency used as the test reference).

Tie-breaking summary:

| Function | Tie rule | `0.5` | `1.5` | `2.5` | `-0.5` | `-2.5` |
|---|---|---|---|---|---|---|
| `round_f32` | away from zero | 1 | 2 | 3 | -1 | -3 |
| `round_ties_even_f32` | to even | 0 | 2 | 2 | -0 | -2 |
| `round_to_nearest_even_i32` | to even | 0 | 2 | 2 | 0 | -2 |
| `round_with_bias_i32` (bias 0.5) | toward +infinity | 1 | 2 | 3 | 0 | -2 |

Special values: the float-returning functions preserve NaN, infinities and the sign of zero (`ceil(-0.3)` is `-0.0`). The integer-returning functions saturate out-of-range values to `i32::MIN`/`i32::MAX` and map NaN to 0 (Rust `as` semantics).

Scalar vs SIMD: `floor_f32`, `ceil_f32`, `trunc_f32` and `round_ties_even_f32` use `vroundps` on x86_64 when AVX is enabled at compile time. Everywhere else (including plain SSE2, which has no `roundps`; that arrived with SSE4.1) a branch-free bit-manipulation fallback is used, which LLVM vectorises with SSE2 instructions only. `round_f32` (ties away from zero) always uses the bit-manipulation code. NEON `vrnd*` is not hand-written; the fallback is left to LLVM. Results are identical between paths.

Dispatch is compile-time (see `docs/adr`); there is no runtime detection. Build with `RUSTFLAGS="-C target-cpu=native"` or `-C target-feature=+avx2` to get `vroundps`. The crate is `#![no_std]`.

## Installation

```toml
[dependencies]
tpt-simd-rounding = "0.1.0"
```

The functions are also re-exported by the umbrella crate `tpt-simd`.

## Quick start

```rust
use tpt_simd_core::F32x8;
use tpt_simd_rounding::{round_f32, round_ties_even_f32, round_to_nearest_even_i32};

let x = F32x8::from_array([0.5, 1.5, 2.5, -0.5, -2.5, 0.49999997, 2.4, -2.6]);
assert_eq!(
    round_f32(x).to_array(),
    [1.0, 2.0, 3.0, -1.0, -3.0, 0.0, 2.0, -3.0]
);

let y = F32x8::from_array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.6, -2.4]);
assert_eq!(
    round_ties_even_f32(y).to_array(),
    [0.0, 2.0, 2.0, -0.0, -2.0, -2.0, 3.0, -2.0]
);

let z = F32x8::from_array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 1e20, f32::NAN]);
assert_eq!(round_to_nearest_even_i32(z).to_array(), [0, 2, 2, 0, -2, -2, i32::MAX, 0]);
```

## API overview

| Function | Result | Description |
|---|---|---|
| `round_f32(a)` | `F32x8` | Round to nearest, ties away from zero (like C `roundf`). |
| `floor_f32(a)` | `F32x8` | Round toward negative infinity. |
| `ceil_f32(a)` | `F32x8` | Round toward positive infinity. |
| `trunc_f32(a)` | `F32x8` | Round toward zero. |
| `round_ties_even_f32(a)` | `F32x8` | Round to nearest, ties to even (like `rintf` in the default mode, and `roundps`/`vrndn`). |
| `round_to_nearest_even_i32(a)` | `I32x8` | Ties-to-even, converted to `i32` (saturating, NaN to 0). |
| `round_with_bias_i32(a, bias)` | `I32x8` | `floor(x + bias)` converted to `i32`. `bias = 0.5` is round-half-up; `0.0` is floor. The addition is a single `f32` operation, so `0.49999997 + 0.5` rounds to `1.0` and yields 1. |
| `ROUNDING_USES_INTRINSICS` | `bool` const | True when the `vroundps` fast path is compiled in. |

## Feature flags

The crate is `#![no_std]` in every configuration.

| Feature | Default | Effect |
|---|---|---|
| `std` | off | Forwards to `tpt-simd-core/std`; the vector backend then uses `std` float routines instead of `libm`. |
| `nightly` | off | Forwards to `tpt-simd-core/nightly` (reserved; no code in this workspace currently branches on it). |
| `scalar-only` | off | Forwards to `tpt-simd-core/scalar-only` and disables this crate's `vroundps` path, so the bit-manipulation fallback is always used (`ROUNDING_USES_INTRINSICS` becomes false). |

## Platform and target support

- x86_64 with `avx` enabled at compile time (and `scalar-only` off): `vroundps` for floor, ceil, trunc and ties-to-even.
- Every other target (SSE2-only x86_64, aarch64, others): branch-free integer bit-manipulation fallback, auto-vectorised by LLVM.
- Accuracy: all results are exact roundings of the input; there is no tolerance. Unsafe code is limited to the AVX load/store wrapper, with a `SAFETY` comment. All public functions are safe and never panic.

## Performance

Benchmarks live in `benches/rounding.rs`; run `cargo bench -p tpt-simd-rounding`, preferably with `RUSTFLAGS="-C target-cpu=native"`. Results for the workspace are collected in [../docs/benchmarks.md](../docs/benchmarks.md).

## Related crates

- `tpt-simd-core`: the `Simd` type (dependency).
- `tpt-simd-saturate`: saturating narrowing conversions.
- `tpt-simd-shift`: `shift_with_rounding_i32` for fixed-point descaling.
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
