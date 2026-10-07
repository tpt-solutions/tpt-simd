# tpt-simd-blend

Branch-free per-lane conditional selection for 256-bit SIMD vectors.

## Overview

`tpt-simd-blend` picks, lane by lane, between two vectors `a` and `b`: `mask ? a : b`. It accepts a `SimdMask` (such as the ones produced by `tpt-simd-compare`), a float vector whose sign bits act as the condition, or a compile-time constant bit mask. It also provides fused "compare greater-than then blend" helpers that never materialise a mask.

Use it when you would otherwise write a per-lane `if`/`else` in a hot loop and want the compiler to emit a single blend instruction.

Supported shapes: `i32x8`, `f32x8` and `i8x32` (via `tpt_simd_core::{I32x8, F32x8, I8x32}`).

Scalar vs SIMD: the crate has one portable implementation (`SimdMask::select` from `tpt-simd-core`, or a per-lane sign-bit test for `select_f32`) and one explicit x86 AVX2 implementation (`vblendvps`, `vpblendvb`, `vcmpps`/`vpcmpgtd`). Both produce identical results, including bit-exact copies of NaN payloads and signed zeros.

Dispatch is compile-time (see `docs/adr`): the AVX2 path is compiled in only when `avx2` is enabled for the target, for example with `RUSTFLAGS="-C target-cpu=native"` or `-C target-feature=+avx2`. There is no runtime detection. The crate is `#![no_std]`.

## Installation

```toml
[dependencies]
tpt-simd-blend = "0.1.0"
```

The functions are also re-exported by the umbrella crate `tpt-simd`. `select_f32` is additionally re-exported by `tpt-simd-select`.

## Quick start

```rust
use tpt_simd_blend::blend_i32;
use tpt_simd_core::{I32x8, SimdMask};

// Lanes 0..4 take `a`, lanes 4..8 take `b`.
let m = SimdMask::<i32, 8>::from_bitmask(0b0000_1111);
let r = blend_i32(m, I32x8::splat(1), I32x8::splat(2));
assert_eq!(r.to_array(), [1, 1, 1, 1, 2, 2, 2, 2]);
```

Fused compare and blend:

```rust
use tpt_simd_blend::blend_gt_f32;
use tpt_simd_core::F32x8;

let x = F32x8::from_array([1.0, 3.0, f32::NAN, 0.0, 5.0, 5.0, 9.0, -1.0]);
// `if x > 2.0 { 1.0 } else { 0.0 }`; NaN compares false and selects `b`.
let r = blend_gt_f32(x, F32x8::splat(2.0), F32x8::splat(1.0), F32x8::splat(0.0));
assert_eq!(r.to_array(), [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
```

## API overview

Mask-driven blends (a true lane selects `a`):

| Function | Description |
|---|---|
| `blend_i32(mask, a, b)` | `i32x8` blend with a `SimdMask<i32, 8>`. |
| `blend_f32(mask, a, b)` | `f32x8` blend with a `SimdMask<f32, 8>`; NaN payloads and signed zeros are copied bit-exactly. |
| `blend_i8(mask, a, b)` | `i8x32` blend with a `SimdMask<i8, 32>`. |

Sign-bit condition:

| Function | Description |
|---|---|
| `select_f32(condition, a, b)` | `_mm256_blendv_ps` semantics: a lane takes `a` when bit 31 of the condition lane is set. `-0.0`, negatives and negative NaNs select `a`; `+0.0`, positives and positive NaNs select `b`. |

Compile-time masks (bit `i` of `MASK` set selects lane `i` of `a`):

| Function | Description |
|---|---|
| `blend_imm_i32::<MASK: u8>(a, b)` | `i32x8` immediate blend. |
| `blend_imm_f32::<MASK: u8>(a, b)` | `f32x8` immediate blend. |
| `blend_imm_i8::<MASK: u32>(a, b)` | `i8x32` immediate blend. |

Note that the operand order is the opposite of Intel's immediate-blend intrinsics: here a set bit selects `a`, matching `blend_*` and `SimdMask::select`.

Fused compare and blend:

| Function | Description |
|---|---|
| `blend_gt_f32(x, y, a, b)` | `if x > y { a } else { b }` for `f32x8`; NaN in `x` or `y` selects `b`. One `vcmpps` plus one `vblendvps` on AVX2. |
| `blend_gt_i32(x, y, a, b)` | Signed `if x > y { a } else { b }` for `i32x8`. |

The crate also re-exports `SimdMask` from `tpt-simd-core`.

## Feature flags

The crate is `#![no_std]` in every configuration. The features only forward to `tpt-simd-core` and from there to `tpt-simd-vector`.

| Feature | Default | Effect |
|---|---|---|
| `std` | off | Forwards to `tpt-simd-core/std`; the vector backend then uses `std` float routines instead of `libm`. |
| `nightly` | off | Forwards to `tpt-simd-core/nightly` (reserved; no code in this workspace currently branches on it). |
| `scalar-only` | off | Forwards to `tpt-simd-core/scalar-only`. This crate's own AVX2 `cfg` does not test this feature, so the AVX2 blends are still used when `avx2` is enabled at compile time. |

## Platform and target support

- x86 / x86_64 with `avx2` enabled at compile time: explicit `vblendvps` (`blend_i32`, `blend_f32`, `select_f32`), `vpblendvb` (`blend_i8`), and `vcmpps`/`vpcmpgtd` plus blend for `blend_gt_*`.
- Every other target (including x86 without AVX2 and aarch64): portable array implementation, which LLVM lowers as it sees fit. There is no hand-written NEON code in this crate.
- `blend_imm_*` are implemented with the portable per-lane code on all targets; with a constant mask LLVM is free to fold them to an immediate blend.
- Unsafe code is limited to loads and stores inside the AVX2 module, each with a `SAFETY` comment. All public functions are safe and never panic.

## Performance

Benchmarks live in `benches/blend.rs`; run `cargo bench -p tpt-simd-blend`, preferably with `RUSTFLAGS="-C target-cpu=native"`. Results for the workspace are collected in [../docs/benchmarks.md](../docs/benchmarks.md).

## Related crates

- `tpt-simd-core`: the `Simd` and `SimdMask` types this crate operates on (dependency).
- `tpt-simd-compare`: produces the masks consumed by `blend_*`.
- `tpt-simd-select`: re-exports `select_f32` and adds index-based lane selection.
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
