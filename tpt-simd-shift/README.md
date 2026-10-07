# tpt-simd-shift

SIMD shift and rotate variants on eight `i32` lanes, with uniform and per-lane amounts and a fully defined out-of-range policy.

## Overview

`tpt-simd-shift` provides left shift, logical and arithmetic right shift, rotate left, and a rounding right shift (the usual fixed-point "descale" step) for `I32x8`. Each operation has a uniform form (one `u32` amount for all lanes) and a per-lane `_var` form (`Simd<u32, 8>` amounts).

Out-of-range policy (amount `>= 32`): amounts are unsigned, so there are no negative amounts. Behaviour matches AVX2 `vpsllvd`/`vpsrlvd`/`vpsravd` (ADR 0002), so the fast path needs no fix-up. No function panics and none has undefined behaviour for any amount.

| Operation | Amount `>= 32` |
|---|---|
| `shift_left` | `0` |
| `shift_right_logical` | `0` |
| `shift_right_arithmetic` | sign fill (`0` or `-1`) |
| `shift_with_rounding` | `0` (the value is below half an ulp) |
| `rotate_left` | amount taken modulo 32 |

Scalar vs SIMD: the uniform forms are lane maps that LLVM auto-vectorises. The per-lane `shift_left_var_i32`, `shift_right_logical_var_i32` and `shift_right_arithmetic_var_i32` use AVX2 `vpsllvd`/`vpsrlvd`/`vpsravd` when AVX2 is enabled at compile time; `rotate_left_var_i32` and `shift_with_rounding_var_i32` use portable per-lane code. Results are identical between paths.

Dispatch is compile-time (see `docs/adr`); there is no runtime detection. Build with `RUSTFLAGS="-C target-cpu=native"` or `-C target-feature=+avx2` for the AVX2 path. The crate is `#![no_std]`.

## Installation

```toml
[dependencies]
tpt-simd-shift = "0.1.0"
```

The functions are also re-exported by the umbrella crate `tpt-simd`.

## Quick start

```rust
use tpt_simd_core::{I32x8, U32x8};
use tpt_simd_shift::{shift_left_i32, shift_with_rounding_i32, rotate_left_var_i32};

assert_eq!(shift_left_i32(I32x8::splat(3), 4), I32x8::splat(48));
assert_eq!(shift_left_i32(I32x8::splat(3), 32), I32x8::splat(0)); // out of range

// Round-half-up descale: (x + (1 << (n-1))) >> n, evaluated in 64 bits.
let r = shift_with_rounding_i32(
    I32x8::from_array([5, 6, 7, -5, -6, -7, i32::MAX, i32::MIN]),
    2,
);
assert_eq!(r.to_array(), [1, 2, 2, -1, -1, -2, 1 << 29, i32::MIN >> 2]);

// Per-lane amounts; rotation is modulo 32.
let r = rotate_left_var_i32(
    I32x8::splat(i32::MIN),
    U32x8::from_array([0, 1, 31, 32, 33, 0, 0, 0]),
);
assert_eq!(r.to_array(), [i32::MIN, 1, 1 << 30, i32::MIN, 1, i32::MIN, i32::MIN, i32::MIN]);
```

## API overview

Uniform amount (`fn(a: I32x8, amount: u32) -> I32x8`):

| Function | Description |
|---|---|
| `shift_left_i32` | Left shift; `>= 32` gives 0. |
| `shift_right_logical_i32` | Zero-filling right shift; `>= 32` gives 0. |
| `shift_right_arithmetic_i32` | Sign-filling right shift; `>= 32` gives sign fill. |
| `rotate_left_i32` | Rotate left by `amount % 32`. |
| `shift_with_rounding_i32` | Arithmetic right shift with round-half-up (ties toward +infinity), 64-bit intermediate so it never overflows; `0` is the identity, `>= 32` gives 0. |

Per-lane amounts (`fn(a: I32x8, amounts: U32x8) -> I32x8`):

| Function | Description |
|---|---|
| `shift_left_var_i32` | Per-lane left shift (`vpsllvd` on AVX2). |
| `shift_right_logical_var_i32` | Per-lane logical right shift (`vpsrlvd` on AVX2). |
| `shift_right_arithmetic_var_i32` | Per-lane arithmetic right shift (`vpsravd` on AVX2). |
| `rotate_left_var_i32` | Per-lane rotate left, amount modulo 32. |
| `shift_with_rounding_var_i32` | Per-lane rounding right shift. |
| `shift_left_var_i32_portable`, `shift_right_logical_var_i32_portable`, `shift_right_arithmetic_var_i32_portable` | Portable references for the three AVX2-accelerated functions. |

Constant:

| Item | Description |
|---|---|
| `SHIFT_VAR_USES_INTRINSICS` | `bool`; true when the AVX2 variable-shift fast path is compiled in. |

## Feature flags

The crate is `#![no_std]` in every configuration.

| Feature | Default | Effect |
|---|---|---|
| `std` | off | Forwards to `tpt-simd-core/std`; the vector backend then uses `std` float routines instead of `libm`. |
| `nightly` | off | Forwards to `tpt-simd-core/nightly` (reserved; no code in this workspace currently branches on it). |
| `scalar-only` | off | Forwards to `tpt-simd-core/scalar-only` and disables this crate's AVX2 variable shifts, so the portable code is always used (`SHIFT_VAR_USES_INTRINSICS` becomes false). |

## Platform and target support

- x86_64 with `avx2` enabled at compile time (and `scalar-only` off): `vpsllvd`, `vpsrlvd` and `vpsravd` for the three variable shifts.
- Every other target, including aarch64: portable code, auto-vectorised by LLVM where possible; there is no hand-written NEON path.
- Semantics are integer-exact; there is no tolerance. Unsafe code is limited to the AVX2 intrinsic wrappers, with `SAFETY` comments. All public functions are safe and never panic.

## Performance

Benchmarks live in `benches/shift.rs`; run `cargo bench -p tpt-simd-shift`, preferably with `RUSTFLAGS="-C target-cpu=native"`. Results for the workspace are collected in [../docs/benchmarks.md](../docs/benchmarks.md).

## Related crates

- `tpt-simd-core`: the `Simd` type (dependency).
- `tpt-simd-rounding`: float rounding and conversion.
- `tpt-simd-saturate`: saturating arithmetic and narrowing.
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
