# tpt-simd-matrix

Small fixed-size matrix kernels: 4x4 `f32` and 8x8 `i16` multiply, in-place transposes, and a 3x3 `f32` inverse.

## Overview

Use this crate for the tiny dense matrices that show up in graphics, transforms and codecs (4x4 transforms, 8x8 integer blocks, 3x3 inverses), where a general BLAS call would be overkill. Matrices are row-major arrays of rows (`[[T; N]; N]`) passed and returned by value (transposes are in place).

- `mat4x4_mul_f32`: `a * b` using one fused multiply-add per lane over 4-lane vectors.
- `mat8x8_mul_i16`: `a * b` with `i64` accumulation, saturated to `i16`.
- `mat4x4_transpose_f32`, `mat8x8_transpose_i16`: in-place transposes that delegate to `tpt-simd-permute`.
- `mat3x3_inverse_f32`: adjugate inverse returning `None` for (nearly) singular or non-finite input.
- Scalar reference functions `mat4x4_mul_scalar_f32` and `mat8x8_mul_scalar_i16`.

Scalar versus SIMD: the multiplies are written with the portable `Simd<T, N>` type from `tpt-simd-core`, which LLVM lowers to vector instructions for the enabled target features. There are no hand-written intrinsics in this crate. Dispatch is compile-time (see [ADR 0001](../docs/adr/0001-backend-and-dispatch.md)); build with `-C target-cpu=native` (or `+avx2,+fma`) to get hardware FMA, otherwise the per-lane fused multiply-add falls back to a slower software path.

`no_std`: the crate is `#![no_std]` and contains no `unsafe` code of its own.

## Installation

```toml
[dependencies]
tpt-simd-matrix = "0.1.0"
```

The same functionality is available through the umbrella crate `tpt-simd` (as `tpt_simd::matrix`).

## Quick start

```rust
use tpt_simd_matrix::{mat3x3_inverse_f32, mat4x4_mul_f32, mat8x8_mul_i16};

let i = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
let m = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]];
assert_eq!(mat4x4_mul_f32(m, i), m);

let mut a = [[0i16; 8]; 8];
for k in 0..8 { a[k][k] = 1; }
let b = [[300i16; 8]; 8];
assert_eq!(mat8x8_mul_i16(a, b), b);
assert_eq!(mat8x8_mul_i16(b, b)[0][0], i16::MAX); // 8 * 300 * 300 saturates

let m = [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 8.0]];
let inv = mat3x3_inverse_f32(m).unwrap();
assert_eq!(inv, [[0.5, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.125]]);
assert!(mat3x3_inverse_f32([[1.0; 3]; 3]).is_none());
```

## API overview

| Item | Description |
|---|---|
| `mat4x4_mul_f32(a, b)` | 4x4 `f32` product, fused multiply-add per lane |
| `mat4x4_mul_scalar_f32(a, b)` | Unfused scalar reference for the above |
| `mat8x8_mul_i16(a, b)` | 8x8 `i16` product, `i64` accumulation, saturating |
| `mat8x8_mul_scalar_i16(a, b)` | Scalar reference for the above (bit-exact match) |
| `mat4x4_transpose_f32(&mut m)` | In-place 4x4 `f32` transpose |
| `mat8x8_transpose_i16(&mut m)` | In-place 8x8 `i16` transpose |
| `mat3x3_inverse_f32(m)` | 3x3 inverse via adjugate, `Option<[[f32; 3]; 3]>` |
| `SINGULAR_EPSILON` | Relative singularity threshold (`1e-6`) |

## Feature flags

| Feature | Effect |
|---|---|
| `default` | empty |
| `std` | Forwards to `tpt-simd-core/std` |
| `nightly` | Forwards to `tpt-simd-core/nightly` |
| `scalar-only` | Forwards to `tpt-simd-core/scalar-only` |

## Platform support and accuracy

- All targets; code generation for x86 SSE/AVX2/AVX-512 and aarch64 NEON comes from LLVM lowering of the portable `Simd` type. No target-specific code paths exist in this crate.
- `mat4x4_mul_f32` uses a fused multiply-add per lane (single rounding), so it can differ from the unfused `mat4x4_mul_scalar_f32` by rounding error. The `f32` product has no other special handling: NaN and infinity propagate as in IEEE arithmetic.
- `mat8x8_mul_i16` accumulates in `i64` (eight products of up to 2^30 can exceed `i32`) and saturates each result to `i16`; it matches the scalar reference exactly.
- `mat3x3_inverse_f32` returns `None` when the determinant is not finite or `|det| <= SINGULAR_EPSILON * max|m|^3` (so NaN input yields `None`). It uses unfused arithmetic and a single reciprocal of the determinant.

## Performance

No benchmarks are included in this crate.

## Related crates

Depends on `tpt-simd-core` and `tpt-simd-permute` (transposes). Siblings: `tpt-simd-blas` (general matrix routines), `tpt-simd-mul`. Umbrella: `tpt-simd`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
