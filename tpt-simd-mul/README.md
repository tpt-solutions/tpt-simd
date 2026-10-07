# tpt-simd-mul

SIMD multiply variants: high/low halves, widening multiplies, alternating multiply-add/sub and a fused complex multiply.

## Overview

Use this crate when you need integer multiply flavours that the standard operators do not give you (the high half of a product, widening products) or an `f32` complex multiply with a fixed, target-independent rounding rule. All functions operate on the `Simd<T, N>` / `ComplexSimd` types of `tpt-simd-core`.

- `mul_hi_i16`, `mul_hi_i32`, `mul_lo_i32`: high 16 bits of the `i16` product, high and low 32 bits of the `i32` product.
- `mul_widen_i16`, `mul_widen_i32`: full-width products (`i16 -> i32`, `i32 -> i64`).
- `mul_add_sub_f32`: `a*b + c` on even lanes, `a*b - c` on odd lanes, each fused.
- `complex_mul_f32` and `complex_mul_f32_portable`: complex multiply of 8 `f32` lanes in split (real/imaginary) layout.

Scalar versus SIMD: the integer functions are lane loops that LLVM auto-vectorises (for example `vpmulhw` on AVX2). `complex_mul_f32` has a hand-written AVX2+FMA intrinsic path that is selected at compile time (`target_arch = "x86_64"` with `avx2` and `fma` enabled and the `scalar-only` feature off); everywhere else it uses the portable path. Dispatch is compile-time (see [ADR 0001](../docs/adr/0001-backend-and-dispatch.md)); build with `-C target-cpu=native` to enable the fast path.

`no_std`: the crate is `#![no_std]` and `#![deny(missing_docs)]`. The only `unsafe` is the AVX2 intrinsic block in `complex_mul_f32`.

## Installation

```toml
[dependencies]
tpt-simd-mul = "0.1.0"
```

The same functionality is available through the umbrella crate `tpt-simd`.

## Quick start

```rust
use tpt_simd_core::{F32x8, I16x16};
use tpt_simd_mul::{complex_mul_f32, mul_hi_i16, ComplexSimd};

let r = mul_hi_i16(I16x16::splat(i16::MIN), I16x16::splat(i16::MIN));
assert_eq!(r, I16x16::splat(0x4000));

let a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0));
let b = ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0));
let r = complex_mul_f32(a, b); // (1+2i)(3+4i) = -5 + 10i
assert_eq!(r.real.to_array(), [-5.0; 8]);
assert_eq!(r.imag.to_array(), [10.0; 8]);
```

## API overview

| Function | Description |
|---|---|
| `mul_hi_i16(Simd<i16,16>, Simd<i16,16>)` | `(a * b) >> 16` per lane, like `pmulhw` |
| `mul_lo_i32(Simd<i32,8>, Simd<i32,8>)` | Wrapping low 32 bits of the product, like `pmulld` |
| `mul_hi_i32(Simd<i32,8>, Simd<i32,8>)` | `(a * b) >> 32` per lane (signed) |
| `mul_widen_i16(Simd<i16,8>, Simd<i16,8>)` | Full 32-bit products, returns `Simd<i32,8>` |
| `mul_widen_i32(Simd<i32,4>, Simd<i32,4>)` | Full 64-bit products, returns `Simd<i64,4>` |
| `mul_add_sub_f32(a, b, c)` | `a*b + c` on even lanes, `a*b - c` on odd lanes (`Simd<f32,8>`) |
| `complex_mul_f32(a, b)` | Fused complex multiply of `ComplexSimd<f32, 8>` |
| `complex_mul_f32_portable(a, b)` | Portable reference for `complex_mul_f32` |
| `COMPLEX_MUL_F32_USES_INTRINSICS` | `true` when the intrinsic fast path is compiled in |
| `ComplexSimd` | Re-export of `tpt_simd_core::ComplexSimd` |

## Feature flags

| Feature | Effect |
|---|---|
| `default` | empty |
| `std` | Forwards to `tpt-simd-core/std` |
| `nightly` | Forwards to `tpt-simd-core/nightly` |
| `scalar-only` | Forwards to `tpt-simd-core/scalar-only`; also disables the AVX2 intrinsic path of `complex_mul_f32` |

## Platform support and accuracy

- x86_64 with AVX2 and FMA enabled at compile time: `complex_mul_f32` uses intrinsics (`vmulps` plus `vfmsub` / `vfmadd`). All other targets, including aarch64 (NEON is not implemented here), use the portable path. The integer functions rely on LLVM auto-vectorisation on every target.
- `complex_mul_f32` always computes `re = fma(a.re, b.re, -(a.im*b.im))` and `im = fma(a.re, b.im, a.im*b.re)`. The portable path uses `Simd::mul_add` (correctly rounded fused multiply-add), so results are bit-identical on every target. Targets without hardware FMA pay for a software fused multiply-add; if you only need speed there, use the unfused `ComplexSimd::mul` from `tpt-simd-core`, whose results can differ in the last bit. NaN and infinity propagate as in IEEE arithmetic.
- `mul_add_sub_f32` uses the opposite lane convention to Intel's `_mm256_fmaddsub_ps` (it adds on even lanes and subtracts on odd lanes), and each lane is fused with a single rounding.
- None of the functions panic.

## Performance

No benchmarks are included in this crate.

## Related crates

Depends on `tpt-simd-core` (which provides `Simd` and `ComplexSimd`). Siblings: `tpt-simd-fixed`, `tpt-simd-matrix`. Umbrella: `tpt-simd`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
