# tpt-simd-reduce

SIMD slice reductions: sums (plain, pairwise, compensated), mean, variance, covariance, min/max/argmin/argmax and norms for `f32`/`f64`, plus `i32` sums and extrema.

## Overview

Use this crate when you need to reduce a whole slice to a number quickly and with documented, reproducible numerical behaviour. Kernels are written once as safe portable code on `tpt_simd_core::Simd` (8 lanes for `f32`, 4 lanes for `f64`) and use four independent accumulators to hide add latency. Combination orders are fixed by the code, no FMA is used, and the halving tree of `tpt-simd-horizontal` finishes each reduction, so float results are bit-for-bit reproducible across targets and feature sets. Compared with a left-to-right scalar loop they are reassociated, so they differ by a few ulps of the sum of absolute values (Tier 2 in ADR 0003); `min`/`max`/`arg*` and the integer functions are exact.

Dispatch is compile-time (ADR 0001): there is no run-time CPU detection and no `runtime-dispatch` feature in this crate. Build with `-C target-cpu=native` (or explicit `+avx2`) for AVX2 speed. The crate is `#![no_std]` and `#![forbid(unsafe_code)]`.

## Installation

```toml
[dependencies]
tpt-simd-reduce = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports this crate.

## Quick start

```rust
use tpt_simd_reduce::*;

let xs = [1.0f32, 2.0, 3.0, 4.0];
assert_eq!(sum_f32(&xs), 10.0);
assert_eq!(mean_f32(&xs), Some(2.5));
assert_eq!(variance_f32(&xs), Some(1.25));
assert_eq!(argmax_f32(&[1.0, f32::NAN, 7.0, 3.0]), Some(2));
assert_eq!(min_f32(&[]), None);

// Compensated summation survives catastrophic cancellation:
let big = [1.0e8f32, 1.0, -1.0e8];
assert_eq!(sum_f32(&big), 0.0);
assert_eq!(sum_compensated_f32(&big), 1.0);
```

## API overview

Every float function exists as `<op>_f32` and `<op>_f64`.

| Function | Returns | Description |
|---|---|---|
| `sum_*` | `T` | Multi-accumulator vectorised sum (empty gives `0`) |
| `pairwise_sum_*` | `T` | Recursive blocked sum (leaves of at most 1024 elements), `~log2(n)·eps` error growth |
| `sum_compensated_*` | `T` | Vectorised Neumaier/TwoSum sum, about 1 ulp of the exact sum; falls back to the plain sum if the result is not finite |
| `mean_*` | `Option<T>` | Arithmetic mean (pairwise); `None` if empty |
| `variance_*`, `sample_variance_*` | `Option<T>` | Two-pass population / sample variance; sample needs `len >= 2` |
| `variance_welford_*` | `Option<T>` | Serial single-pass Welford population variance |
| `covariance_*`, `sample_covariance_*` | `Option<T>` | Two-pass population / sample covariance; panic if lengths differ |
| `min_*`, `max_*` | `Option<T>` | Extremum, NaN ignored; `Some(NaN)` if all NaN, `None` if empty |
| `argmin_*`, `argmax_*` | `Option<usize>` | Index of the first extremum; `None` if empty or all NaN |
| `sum_squares_*` | `T` | Sum of squares (unfused) |
| `norm_*` | `T` | Euclidean norm, no overflow scaling |

Integer functions (`i32`): `sum_i32` (wrapping), `sum_i32_wide` (exact, widened to `i64`), `min_i32`, `max_i32`, `argmin_i32`, `argmax_i32`.

## Feature flags

| Feature | Default | Effect |
|---|---|---|
| `std` | off | Enables `tpt-simd-core/std` and `tpt-simd-horizontal/std`. The crate itself stays `no_std`. |
| `nightly` | off | Forwards `nightly` to core and horizontal (reserved, currently a no-op in core). |
| `scalar-only` | off | Forwards `scalar-only` to core and horizontal. This crate has no separate intrinsic path, so it does not change the kernels here. |

## Platform support and semantics

- Targets: any target supported by `tpt-simd-core`; the code is portable and auto-vectorised by LLVM for the enabled target features (SSE2 baseline, AVX2 with `-C target-cpu=native`, NEON on aarch64). There is no hand-written intrinsic code, so no `unsafe`.
- Accumulation order: blocks of `4N` elements go to four accumulators, remaining whole vectors to accumulator 0, accumulators combine as `(a0+a1)+(a2+a3)`, lanes reduce by the halving tree, and the final `< N` tail is added left to right.
- Accuracy: plain sum about `(n/(4N) + log2(4N) + N)·eps` relative to the sum of absolute values; pairwise `~log2(n)·eps`; compensated `~eps + O(n·eps²)`. When terms cancel, compare against a scalar reference with a tolerance relative to the sum of `|x|`, not to the result.
- NaN/infinity: sums, means, variances, covariances and norms propagate them as IEEE arithmetic does. `min`/`max` ignore NaN; the sign of a zero result is unspecified if both zeros are present.
- Empty input: sums are `0`; functions without a sensible value return `None`. Mismatched lengths in covariance panic.
- `norm_*` squares intermediate values, so it overflows for `|x|` above about `sqrt(MAX)`.

## Performance

Benchmarks live in `benches/reduce.rs`; run them with `cargo bench -p tpt-simd-reduce`. See [docs/benchmarks.md](../docs/benchmarks.md) for recorded results.

## Related crates

- [`tpt-simd-core`](../tpt-simd-core): the `Simd` vector type.
- [`tpt-simd-horizontal`](../tpt-simd-horizontal): the halving-tree lane reduction used by every kernel.
- [`tpt-simd-blas`](../tpt-simd-blas), [`tpt-simd-dot`](../tpt-simd-dot): related numeric kernels (dot products, norms).

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
