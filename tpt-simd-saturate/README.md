# tpt-simd-saturate

SIMD saturating arithmetic and saturating narrowing packs that clamp instead of wrapping.

## Overview

`tpt-simd-saturate` provides lane-wise saturating add and subtract for `i8`, `u8`, `i16` and `u16` vectors, a saturating `i16` multiply, and saturating narrow ("pack") conversions in both vector and slice form. Typical uses are pixel and audio processing, where overflow must clamp.

Lane order is always natural: lane `i` of the result comes from lane `i` of the input. Unlike a raw `_mm256_packs_epi16`, there is no interleaving of the two 128-bit halves.

Scalar vs SIMD: the implementations are fixed-length lane loops over `Simd`, which LLVM lowers to `paddsw`, `psubusb`, `packsswb` and similar on x86 and to `sqadd`/`sqxtn` on aarch64. The portable code is the fast path; there are no hand-written intrinsics and the crate is `#![forbid(unsafe_code)]`. Dispatch is therefore whatever the compiler targets at build time (see `docs/adr`); there is no runtime detection. The crate is `#![no_std]`.

## Installation

```toml
[dependencies]
tpt-simd-saturate = "0.1.0"
```

The functions are also re-exported by the umbrella crate `tpt-simd`. `tpt-simd-permute` re-exports `pack_i16_to_i8` and `pack_i16_to_i8_slice`.

## Quick start

```rust
use tpt_simd_core::{I16x16, I8x16};
use tpt_simd_saturate::{pack_i16_to_i8, saturating_add_i16};

let a = I16x16::splat(i16::MAX);
assert_eq!(saturating_add_i16(a, I16x16::splat(1)), a);

let v = I16x16::from_fn(|i| (i as i16 - 8) * 100);
let p: I8x16 = pack_i16_to_i8(v);
assert_eq!(p[0], -128); // -800 clamps
assert_eq!(p[8], 0);
assert_eq!(p[15], 127); // 700 clamps
```

Slice form:

```rust
use tpt_simd_saturate::pack_i16_to_u8_slice;

let src = [-500i16, -1, 0, 1, 500];
let mut dst = [9u8; 5];
pack_i16_to_u8_slice(&src, &mut dst);
assert_eq!(dst, [0, 0, 0, 1, 255]);
```

## API overview

Saturating add and subtract (`fn(a, b) -> Simd`):

| Element | Vector | Functions |
|---|---|---|
| `i16` | `I16x16` | `saturating_add_i16`, `saturating_sub_i16` |
| `i8` | `I8x32` | `saturating_add_i8`, `saturating_sub_i8` |
| `u8` | `U8x32` | `saturating_add_u8`, `saturating_sub_u8` |
| `u16` | `U16x16` | `saturating_add_u16`, `saturating_sub_u16` |

Saturating multiply:

| Function | Description |
|---|---|
| `saturating_mul_i16(a, b)` | Exact 32-bit product of each `i16` lane pair clamped to `i16::MIN..=i16::MAX`. This is a single clamped result per lane, not the high/low half product (`pmulhw`/`pmullw`). |

Saturating narrowing packs (vector form):

| Function | Description |
|---|---|
| `pack_i16_to_i8(v)` | 16 `i16` lanes to 16 `i8`, clamped to `-128..=127`. |
| `pack_i16_to_u8(v)` | 16 `i16` lanes to 16 `u8`, clamped to `0..=255` (negatives become 0). |
| `pack_i32_to_i16(v)` | 8 `i32` lanes to 8 `i16`, clamped to `i16` range. |

Slice forms (`fn(src: &[S], dst: &mut [D])`): `pack_i16_to_i8_slice`, `pack_i16_to_u8_slice`, `pack_i32_to_i16_slice`. They process full vectors with the SIMD routine and the tail with the same clamp in scalar form, so the result is identical for every length including 0. They panic if `src.len() != dst.len()`. `pack_i16_to_i8_slice` is the home of the function that `tpt-simd-permute` re-exports.

## Feature flags

The crate is `#![no_std]` in every configuration (the crate root pulls in `std` only for the `std` feature or in tests).

| Feature | Default | Effect |
|---|---|---|
| `std` | off | Forwards to `tpt-simd-core/std`; the vector backend then uses `std` float routines instead of `libm`. |
| `nightly` | off | Forwards to `tpt-simd-core/nightly` (reserved; no code in this workspace currently branches on it). |
| `scalar-only` | off | Forwards to `tpt-simd-core/scalar-only`. This crate has no intrinsic paths of its own, so it has no further effect here. |

## Platform and target support

- All targets: portable lane loops, auto-vectorised by LLVM (SSE2/AVX2 on x86, NEON on aarch64) with identical results everywhere.
- Semantics: add, subtract and multiply clamp to the element type's range; packs clamp to the destination range. There is no unsafe code and no public function panics, except that the slice packs panic on a length mismatch.

## Performance

Benchmarks live in `benches/saturate.rs`; run `cargo bench -p tpt-simd-saturate`, preferably with `RUSTFLAGS="-C target-cpu=native"`. Results for the workspace are collected in [../docs/benchmarks.md](../docs/benchmarks.md).

## Related crates

- `tpt-simd-core`: the `Simd` type (dependency).
- `tpt-simd-permute`: re-exports `pack_i16_to_i8` and `pack_i16_to_i8_slice`.
- `tpt-simd-rounding`: float rounding and conversion.
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
