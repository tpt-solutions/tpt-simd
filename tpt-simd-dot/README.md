# tpt-simd-dot

SIMD dot product variants for `i16` (wrapping and saturating), `f32` and lane-wise complex `f32`.

## Overview

Use this crate for the inner-product kernels of DSP code (FLAC-style LPC prediction, FIR filters, correlation). Each function has a documented, fixed accumulation order:

| Function | Element | Accumulator |
|---|---|---|
| `dot_product_i16` | `i16` | `i32`, wrapping (order irrelevant) |
| `dot_product_saturating_i16` | `i16` | `i32`, saturating, 8 lane accumulators |
| `dot_product_f32` | `f32` | 32 independent `f32` partial sums, no FMA |
| `dot_product_complex_f32` | `ComplexSimd<f32, 8>` | one running complex sum per lane, no FMA |

Scalar-vs-SIMD: the `portable` module is the behavioural reference and is written to auto-vectorise. On x86_64 builds with `avx2` enabled (and without `scalar-only`), `dot_product_i16` uses an explicit `vpmaddwd` kernel that is bit-identical to the portable one. The other functions are portable only; an AVX2 `f32` version was benchmarked and was no faster. Dispatch is compile-time (ADR 0001); there is no run-time detection. Results are Tier 1 (bit-exact across targets and build configurations). The crate is `#![no_std]`.

## Installation

```toml
[dependencies]
tpt-simd-dot = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports this crate.

## Quick start

```rust
use tpt_simd_dot::{dot_product_f32, dot_product_i16, dot_product_saturating_i16};

assert_eq!(dot_product_i16(&[1, 2, 3], &[4, 5, 6]), 32);
assert_eq!(dot_product_i16(&[], &[]), 0);
// (-32768)^2 * 2 = 2^31 wraps to i32::MIN
assert_eq!(dot_product_i16(&[i16::MIN; 2], &[i16::MIN; 2]), i32::MIN);

let big = [i16::MIN; 64];
assert_eq!(dot_product_saturating_i16(&big, &big), i32::MAX);

assert_eq!(dot_product_f32(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
```

Complex lane-wise product (requires `tpt-simd-core`):

```rust
use tpt_simd_core::{ComplexSimd, F32x8};
use tpt_simd_dot::dot_product_complex_f32;

let a = [ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0))];
let b = [ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0))];
let r = dot_product_complex_f32(&a, &b); // (1+2i)(3+4i) = -5+10i
assert_eq!(r.real.to_array(), [-5.0; 8]);
assert_eq!(r.imag.to_array(), [10.0; 8]);
```

## API overview

| Item | Description |
|---|---|
| `dot_product_i16(&[i16], &[i16]) -> i32` | Wrapping `i32` accumulation; AVX2 `vpmaddwd` fast path |
| `dot_product_saturating_i16(&[i16], &[i16]) -> i32` | Saturating accumulation into 8 lane accumulators combined by the halving tree |
| `dot_product_f32(&[f32], &[f32]) -> f32` | 32 partial sums combined `(acc0+acc1)+(acc2+acc3)`, then the `horizontal_sum_f32` tree |
| `dot_product_complex_f32(&[ComplexSimd<f32, 8>], &[ComplexSimd<f32, 8>]) -> ComplexSimd<f32, 8>` | Per-lane `sum a[i]*b[i]`, not conjugated; conjugate one input for the Hermitian product |
| `portable` module | Reference implementations of the same four functions (no AVX2) |

All top-level functions accept any length (tails are handled internally), return zero for empty input, and **panic** if the two slices differ in length.

## Feature flags

| Feature | Default | Effect |
|---|---|---|
| `std` | off | Enables `tpt-simd-core/std` and `tpt-simd-horizontal/std`. The crate itself stays `no_std`. |
| `nightly` | off | Forwards `nightly` to core and horizontal (reserved, currently a no-op in core). |
| `scalar-only` | off | Disables the AVX2 `dot_product_i16` kernel (and so re-enables `forbid(unsafe_code)`); the portable path is used everywhere. |

## Platform support and semantics

- x86_64 with `avx2` enabled at compile time (for example `-C target-cpu=native`): AVX2 `dot_product_i16`. All other targets, including aarch64, use the portable code (auto-vectorised by LLVM). There is no AVX-512 or NEON-specific code.
- Safety: `unsafe` exists only in the AVX2 module, which is compiled only when AVX2 is enabled for the whole crate; otherwise `unsafe_code` is forbidden. Slice lengths are checked before the unsafe call.
- Overflow: `dot_product_i16` wraps modulo `2^32`; `dot_product_saturating_i16` saturates per accumulator (no intermediate widening, so it can differ from the exact sum clamped once at the end).
- `f32`: separate multiply and add (never FMA); reassociated relative to a left-to-right loop, so it can differ from a naive loop by normal rounding, but not between targets. NaN and infinities propagate.

## Performance

Benchmarks live in `benches/dot.rs`; run them with `cargo bench -p tpt-simd-dot`. See [docs/benchmarks.md](../docs/benchmarks.md) for recorded results. `tests/flac_lpc.rs` is a realistic FLAC-style LPC residual test built on `dot_product_i16`.

## Related crates

- [`tpt-simd-core`](../tpt-simd-core): `ComplexSimd` and the vector aliases.
- [`tpt-simd-horizontal`](../tpt-simd-horizontal): the lane reduction tree used by the `f32` and saturating kernels.
- [`tpt-simd-reduce`](../tpt-simd-reduce), [`tpt-simd-blas`](../tpt-simd-blas): slice reductions and BLAS-style `f32`/`f64` dot products.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
