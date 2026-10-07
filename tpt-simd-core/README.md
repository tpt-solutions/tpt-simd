# tpt-simd-core

Core traits, vector type aliases, native-width constants and CPU feature detection for tpt-simd.

## Overview

`tpt-simd-core` is the shared foundation of the workspace. It re-exports the backend types from `tpt-simd-vector` (`Simd`, `SimdMask`, `SimdElement`, `SimdInt`, `SimdFloat`, `LaneCast`, `MaskLane`) and adds:

- Type aliases for common shapes, such as `F32x8` and `I16x16`, in 128-, 256- and 512-bit families.
- `SimdVector` / `SimdOps` / `DefaultBackend`, traits that abstract over the backend type.
- `width`: constants for the native vector register width of the compile target.
- `detect`: compile-time and (with `std`) runtime CPU feature detection.
- `ComplexSimd`: split (structure-of-arrays) complex vectors shared by the complex, mul, butterfly and dot crates.

Scalar versus SIMD: the backend is array-backed lane loops that LLVM vectorises; there is no separate scalar kernel here. Dispatch in the workspace is compile-time (`cfg(target_feature)`); see `docs/adr/0001-backend-and-dispatch.md` and `docs/adr/0003-runtime-dispatch-and-float-tolerance.md`. The crate is `#![no_std]` and `#![forbid(unsafe_code)]`.

## Installation

```toml
[dependencies]
tpt-simd-core = "0.1.0"
```

Or depend on the umbrella crate `tpt-simd`, which re-exports everything in this crate at its root.

## Quick start

```rust
use tpt_simd_core::detect::Features;
use tpt_simd_core::width::NATIVE_F32_LANES;
use tpt_simd_core::{ComplexSimd, F32x8};

// Complex multiply on 8 lanes at once: (1+2i)(3+4i) = -5 + 10i
let a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0));
let b = ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0));
let r = a * b;
assert_eq!(r.real.to_array(), [-5.0; 8]);
assert_eq!(r.imag.to_array(), [10.0; 8]);

// What may the compiler assume for this build, and how wide is a register?
let f = Features::compile_time();
println!("avx2 = {}, native f32 lanes = {}", f.avx2, NATIVE_F32_LANES);
```

## API overview

| Item | Description |
| --- | --- |
| `F32x4`, `F32x8`, `F32x16`, `F64x2`, `F64x4`, `F64x8` | Float aliases for `Simd<T, N>` |
| `I8x16`..`I8x64`, `I16x8`..`I16x32`, `I32x4`..`I32x16`, `I64x2`..`I64x8` | Signed integer aliases |
| `U8x16`..`U8x64`, `U16x8`..`U16x32`, `U32x4`..`U32x16`, `U64x2`..`U64x8` | Unsigned integer aliases |
| `Mask8<T>`, `Mask16<T>`, `Mask32<T>` | `SimdMask<T, 8/16/32>` |
| `SimdVector<T, N>` | Trait: `splat`, `from_array`, `to_array`, `LANES` |
| `SimdOps<T, N>` | Trait mapping an element type and lane count to a vector type |
| `DefaultBackend` | The backend selected at compile time (`tpt-simd-vector`'s `Simd`) |
| `width::NATIVE_VECTOR_BITS` / `NATIVE_VECTOR_BYTES` | Native register width: 512 with `avx512f`, 256 with `avx`, otherwise 128 |
| `width::native_lanes::<T>()` | Lanes of `T` that fill one native register |
| `width::NATIVE_F32_LANES`, `NATIVE_I32_LANES`, `NATIVE_I16_LANES`, `NATIVE_I8_LANES` | Precomputed lane counts |
| `detect::Features` | Flags: `sse2`, `sse41`, `avx`, `avx2`, `fma`, `avx512f`, `neon`, `sve`, `rvv` |
| `Features::compile_time()` | `const fn`; what the compiler was told it may assume |
| `Features::runtime()` | Queries the running CPU with `std`; otherwise the compile-time set |
| `ComplexSimd<T, N>` | Fields `real`, `imag`; `new`, `splat`, `zero`, `add`, `sub`, `neg`, `conj`, `mul`, `scale`, `mag_sq`, `div`, `twiddle_mul`, `butterfly`, and (floats) `mag`, `phase`; operators `+ - * /` and unary `-` |

`ComplexSimd` uses plain unfused arithmetic so results are identical on every target. The FMA-accelerated `f32` versions live in `tpt-simd-mul` and `tpt-simd-complex`.

## Feature flags

| Feature | Default | Effect |
| --- | --- | --- |
| `std` | no | Enables `Features::runtime()` CPU queries (`is_x86_feature_detected!`, `is_aarch64_feature_detected!`) and forwards to `tpt-simd-vector/std`. Without it the crate is `no_std` and `runtime()` returns the compile-time set. |
| `nightly` | no | Forwards to `tpt-simd-vector/nightly`; reserved for a `core::simd` backend and currently a no-op. |
| `scalar-only` | no | Forwards to `tpt-simd-vector/scalar-only`; currently a no-op. |

## Platform and safety notes

- Width constants are derived from `cfg(target_feature)`, so build with `RUSTFLAGS="-C target-cpu=native"` (or explicit `-C target-feature`) to raise them. They are hints for choosing a lane count; every width works on every target.
- Runtime detection covers x86/x86_64 (SSE2, SSE4.1, AVX, AVX2, FMA, AVX-512F) and aarch64 (NEON, SVE). `rvv` is only set from compile-time `target_feature`. Runtime results are ORed with the compile-time set.
- No `unsafe` code. `ComplexSimd::div` follows IEEE semantics (division by zero gives inf/NaN); it is intended for float lanes.

## Related crates

- `tpt-simd-vector`: the backend this crate re-exports.
- `tpt-simd-testutil` (unpublished): dev-dependency used for tests.
- `tpt-simd-aligned`, `tpt-simd-complex`, `tpt-simd-mul`, `tpt-simd-butterfly`, `tpt-simd-dot` and the rest of the workspace depend on this crate.
- `tpt-simd`: umbrella crate.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
