# tpt-simd-convolve

SIMD 1-D and separable 2-D convolution for `f32`, and a Q15 FIR filter for `i16`.

## Overview

- `convolve_1d_f32`: full 1-D convolution with zero extension at both edges.
- `convolve_2d_separable_f32`: "same"-size separable 2-D convolution of a
  row-major image with zero padding, using caller-provided scratch space.
- `fir_filter_i16`: causal Q15 fixed-point FIR filter with round-half-up and
  saturation.

Each function has a scalar reference (`*_scalar_*`) that performs the same
floating-point operations in the same order, so SIMD and scalar results are
bit-identical. Every `f32` output sample is `fma(kernel[k], input[n-k], acc)`
accumulated over `k` in ascending order starting from `0.0`; the SIMD paths
vectorise across outputs, which preserves that order.

Dispatch is compile-time only: with `+avx` and `+fma` enabled (for example
`-C target-cpu=native`) on x86_64 the 8-lane fused multiply-add compiles to
`vfmadd`; otherwise a portable path (`Simd::mul_add` / `libm::fmaf`) gives
identical results but is slower. There is no runtime CPU detection. See
[ADR 0003](../docs/adr/0003-runtime-dispatch-and-float-tolerance.md).

The crate is `#![no_std]` and allocation-free (the 2-D routine takes a scratch
slice instead of allocating).

## Installation

```toml
[dependencies]
tpt-simd-convolve = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports the individual crates if you prefer a
single dependency.

## Quick start

```rust
use tpt_simd_convolve::{convolve_1d_f32, convolve_1d_output_len, fir_filter_i16};

let input = [1.0f32, 2.0, 3.0];
let kernel = [1.0f32, 1.0];
let mut out = [0.0f32; 4];
assert_eq!(convolve_1d_output_len(input.len(), kernel.len()), 4);
convolve_1d_f32(&input, &kernel, &mut out);
assert_eq!(out, [1.0, 3.0, 5.0, 3.0]);

// Q15 FIR: 0.5 * x[n] + 0.5 * x[n-1]
let mut y = [0i16; 4];
fir_filter_i16(&[100, 200, 300, 400], &[16384, 16384], &mut y);
assert_eq!(y, [50, 150, 250, 350]);
```

## API overview

| Item | Description |
|------|-------------|
| `convolve_1d_output_len(input_len, kernel_len)` | `const fn`: required output length, `input_len + kernel_len - 1`, or `0` if either is empty |
| `convolve_1d_f32(input, kernel, output)` | Full 1-D convolution; `output.len()` must equal the length above. For "same" size take `output[(kernel.len() - 1) / 2..][..input.len()]` |
| `convolve_1d_scalar_f32` | Scalar reference, bit-identical |
| `convolve_2d_separable_f32(input, width, height, kernel_h, kernel_v, output, scratch)` | "Same"-size separable 2-D convolution, zero padding; kernel centre is index `len / 2`; `scratch.len() >= width * height` |
| `convolve_2d_separable_scalar_f32` | Scalar reference, bit-identical |
| `fir_filter_i16(input, coefficients, output)` | Causal Q15 FIR: `sat_i16((sum coefficients[k] * input[n-k] + 2^14) >> 15)`, 64-bit accumulator; no coefficients gives all zeros |
| `fir_filter_scalar_i16` | Scalar reference, identical results |

Panics: the 1-D routines panic if `output.len()` is wrong; the 2-D routines
panic if `input`/`output` are not `width * height`, `scratch` is too short, or
either kernel is empty (also on `width * height` overflow); the FIR routines
panic unless `output.len() == input.len()`.

## Feature flags

- `default = []`: no features are enabled by default; the crate is `#![no_std]`.
- `std`: forwards to `tpt-simd-core/std`.
- `nightly`: forwards to `tpt-simd-core/nightly`.
- `scalar-only`: forwards to `tpt-simd-core/scalar-only` (core uses scalar implementations only, no SIMD).

`no_std` behaviour: the crate is always `no_std`; the `std` feature only links
`std` for the crate's own tests and forwards the feature to core.

## Platform and accuracy notes

- Targets: x86_64 with static `avx` + `fma` uses `_mm256_fmadd_ps` (and
  `_mm_fmadd_ss` for the scalar tail with static `fma`); all other targets use
  the portable fused path (`libm::fmaf`). Results are bit-identical either way.
- `unsafe`: confined to two small private helpers (`fma_v`, `fma_s`) that call
  x86 FMA intrinsics, compiled only when the required target features are
  statically enabled. The crate does not use `forbid(unsafe_code)`.
- Edge policy: zero padding. `f32` arithmetic follows IEEE-754 (NaN/infinity
  propagate).
- FIR: Q15 coefficients (`32768 == 1.0`), rounding is half up (ties toward
  +infinity), result saturates to `i16::MIN..=i16::MAX`.
- Best for kernels much shorter than the input (one fused multiply-add stream
  per tap).

## Performance

Benchmarks live in `benches/` (SIMD vs scalar for all three operations):

```sh
cargo bench -p tpt-simd-convolve
```

See also [docs/benchmarks.md](../docs/benchmarks.md).

## Related crates

- [`tpt-simd-core`](../tpt-simd-core): vector types; the only workspace dependency.
- [`tpt-simd-window`](../tpt-simd-window): window functions commonly applied before filtering or FFT.
- [`tpt-simd-interpolate`](../tpt-simd-interpolate), [`tpt-simd-fixed`](../tpt-simd-fixed): sibling signal-processing and fixed-point crates.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
