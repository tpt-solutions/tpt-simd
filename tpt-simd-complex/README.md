# tpt-simd-complex

SIMD complex-number operations (split real/imaginary layout) for FFT and MDCT code.

## Overview

`tpt-simd-core` defines `ComplexSimd<T, N>`: a vector of `N` complex numbers
stored as separate `real` and `imag` vectors. This crate re-exports it and adds
extension traits with fused multiply-add forms, twiddle multiplication,
conjugate multiply, squared magnitude, polar construction, exact multiplication
by `i` / `-i`, a twiddle-factor FFT butterfly, and conversion between
interleaved (AoS, `[re0, im0, re1, im1, ...]`) and split (SoA) data.

Scalar vs SIMD: the 8-lane `f32` multiply goes through `tpt_simd_mul::complex_mul_f32`
(hardware FMA when compiled in); other widths use the same formula via
`Simd::mul_add`. Dispatch is compile-time only (target features such as
`+avx2,+fma`, e.g. via `-C target-cpu=native`); there is no runtime CPU
detection in this crate. See
[ADR 0003](../docs/adr/0003-runtime-dispatch-and-float-tolerance.md).

The crate is `#![no_std]` (no allocator needed), `#![forbid(unsafe_code)]` and
`#![deny(missing_docs)]`.

## Installation

```toml
[dependencies]
tpt-simd-complex = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports the individual crates if you prefer a
single dependency.

## Quick start

```rust
use tpt_simd_complex::ComplexSimd;
use tpt_simd_core::F32x8;

let a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0));
let b = ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0));
let result = a * b; // (1+2i)(3+4i) = -5 + 10i
assert_eq!(result.real.to_array(), [-5.0; 8]);
assert_eq!(result.imag.to_array(), [10.0; 8]);
```

FFT butterfly with a fused twiddle multiply, and AoS to SoA conversion:

```rust
use tpt_simd_complex::{ComplexSimd, ComplexSimdF32Ext};

let mut a = ComplexSimd::<f32, 8>::splat(1.0, 0.0);
let mut b = ComplexSimd::<f32, 8>::splat(0.0, 1.0);
ComplexSimd::fft_butterfly(&mut a, &mut b, ComplexSimd::splat(0.0, 1.0)); // w = i
assert_eq!((a.real[0], a.imag[0]), (0.0, 0.0));
assert_eq!((b.real[0], b.imag[0]), (2.0, 0.0));

let (mut re, mut im) = ([0.0; 3], [0.0; 3]);
tpt_simd_complex::interleaved_to_split(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &mut re, &mut im);
assert_eq!(re, [1.0, 3.0, 5.0]);
assert_eq!(im, [2.0, 4.0, 6.0]);
```

## API overview

Re-exports: `ComplexSimd`, `F32x8`, `Simd` (from `tpt-simd-core`) and
`complex_mul_f32` (from `tpt-simd-mul`).

`ComplexSimdF32Ext<N>` (implemented for every `ComplexSimd<f32, N>`):

| Method | Description |
|--------|-------------|
| `fmadd(b, c)` | `self * b + c`, fused rounding |
| `fmsub(b, c)` | `self * b - c`, fused rounding |
| `twiddle_mul_fma(w)` | Multiply by a twiddle factor (fused; same bits as `complex_mul_f32`). Named `_fma` because the inherent unfused `twiddle_mul` in core would shadow it |
| `mul_conj(b)` | `self * conj(b)`, fused |
| `norm_sq()` | `re^2 + im^2` with one fused operation |
| `from_polar(mag, phase)` | `mag * (cos(phase) + i sin(phase))` per lane via `libm` (scalar per lane, not a hot path) |
| `i_mul()` / `neg_i_mul()` | Multiply by `i` / `-i` (no multiplies, exact) |
| `scale_scalar(s)` | Multiply every lane by the scalar `s` |
| `fft_butterfly(a, b, w)` | `(a, b) <- (a + b*w, a - b*w)` with the fused twiddle multiply |
| `from_interleaved(src)` | Load `N` complex numbers from AoS `[re0, im0, ...]` (panics unless `src.len() == 2 * N`) |
| `to_interleaved(dst)` | Store as AoS (panics unless `dst.len() == 2 * N`) |

`ComplexSimdF64Ext<N>` (implemented for every `ComplexSimd<f64, N>`, portable):
`twiddle_mul_fma`, `mul_conj`, `norm_sq`, `i_mul`, `neg_i_mul`.

Slice helpers (any length, including empty; panic on length mismatch):

| Function | Description |
|----------|-------------|
| `interleaved_to_split(src, re, im)` | Deinterleave `[re0, im0, ...]` into `re` and `im` slices |
| `split_to_interleaved(re, im, dst)` | Interleave `re` and `im` into `dst` |

## Feature flags

- `default = []`: no features are enabled by default; the crate is `#![no_std]`.
- `std`: forwards to `tpt-simd-core/std` and `tpt-simd-mul/std`.
- `nightly`: forwards to `tpt-simd-core/nightly` and `tpt-simd-mul/nightly`.
- `scalar-only`: forwards to `tpt-simd-core/scalar-only` and `tpt-simd-mul/scalar-only`.

`no_std` behaviour: the crate is always `no_std`; the `std` feature only links
`std` and forwards the feature to its dependencies.

## Platform and accuracy notes

- Targets: whatever `tpt-simd-core` and `tpt-simd-mul` support (an AVX2+FMA
  path for the 8-lane multiply when compiled in; otherwise a correctly rounded
  software `fmaf` / portable path). This crate has no `unsafe` of its own.
- Rounding rule: the operators on `ComplexSimd` (`+ - * /`, `mul`, `mag_sq`) are
  unfused and identical everywhere. Every fused method of this crate
  (`twiddle_mul_fma`, `fmadd`, `fmsub`, `mul_conj`, `norm_sq`, `fft_butterfly`)
  uses FMA rounding on all targets, so results are bit-identical across targets.
- `from_polar` uses `libm::sincosf`; accuracy is that of `libm`.
- `from_interleaved`, `to_interleaved` and the slice helpers panic on length
  mismatch.

## Performance

Benchmarks live in `benches/` (complex multiply over 1024 elements: scalar,
portable fused, core unfused, and the twiddle multiply):

```sh
cargo bench -p tpt-simd-complex
```

See also [docs/benchmarks.md](../docs/benchmarks.md).

## Related crates

- [`tpt-simd-core`](../tpt-simd-core): `ComplexSimd` and vector types.
- [`tpt-simd-mul`](../tpt-simd-mul): `complex_mul_f32`, the fused 8-lane complex multiply used here.
- [`tpt-simd-butterfly`](../tpt-simd-butterfly): unfused radix-2 butterflies.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
