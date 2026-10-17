# tpt-simd

Umbrella crate re-exporting every tpt-simd crate: portable SIMD vectors and DSP/numeric kernels for stable Rust.

## Overview

`tpt-simd` is a single dependency that gives access to the whole workspace. It is `#![no_std]` and `#![forbid(unsafe_code)]` itself. The core types (`Simd`, `SimdMask`, the `F32x8`-style aliases, `ComplexSimd`, `detect`, `width`) are re-exported at the crate root from `tpt-simd-core`; each other crate is available as a module.

Scalar versus SIMD: the vector backend is stable, array-backed lane loops that LLVM lowers to SSE/AVX/AVX-512/NEON when the build allows. Some kernel crates add `cfg(target_feature)`-gated x86 intrinsics paths, with a portable scalar reference used for tests and as the fallback.

Dispatch: the `tpt-simd` umbrella enables `runtime-dispatch` by default. That feature detects CPU features once at runtime (x86_64) and selects the fastest kernel, so a plain build gets the same kernels as an `-C target-cpu=native` build without any flags. It requires `std`. To keep `no_std` (with the compile-time kernels only), depend on `tpt-simd` with `default-features = false`. The individual crates default to no runtime dispatch; enable their `runtime-dispatch` feature to get it. See `docs/adr/0001-backend-and-dispatch.md` and `docs/adr/0003-runtime-dispatch-and-float-tolerance.md`.

## Installation

```toml
[dependencies]
tpt-simd = "0.1.0"
```

To depend on a single area only, use the individual crate instead (for example `tpt-simd-dot = "0.1.0"`).

## Quick start

```rust
use tpt_simd::{ComplexSimd, F32x8};

let a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0));
let b = ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0));
let r = a * b; // (1+2i)(3+4i) = -5 + 10i
assert_eq!(r.real.to_array(), [-5.0; 8]);
assert_eq!(r.imag.to_array(), [10.0; 8]);
```

Other crates are reached through modules, for example `tpt_simd::dot`, `tpt_simd::blas` or `tpt_simd::window`; see each crate's documentation for its API.

Runnable examples live in [examples/](examples/):

```sh
cargo run -p tpt-simd --example quickstart                      # tour: complex multiply, dot, FIR, reduce
cargo run -p tpt-simd --features std --example cpu_report       # which SIMD paths this build and CPU use
```

## Module map

| Module | Crate | Area |
| --- | --- | --- |
| (root) | `tpt-simd-core` | Types, aliases, detection, `ComplexSimd` |
| `aligned` | `tpt-simd-aligned` | Aligned wrappers, checked loads/stores, aligned buffers |
| `blas` | `tpt-simd-blas` | BLAS-style kernels |
| `blend` | `tpt-simd-blend` | Blending |
| `butterfly` | `tpt-simd-butterfly` | FFT butterflies |
| `compare` | `tpt-simd-compare` | Comparisons |
| `complex` | `tpt-simd-complex` | Complex arithmetic |
| `convolve` | `tpt-simd-convolve` | Convolution |
| `dot` | `tpt-simd-dot` | Dot products |
| `fixed` | `tpt-simd-fixed` | Fixed-point arithmetic |
| `gather` | `tpt-simd-gather` | Gather |
| `horizontal` | `tpt-simd-horizontal` | Horizontal operations |
| `interpolate` | `tpt-simd-interpolate` | Interpolation |
| `math` | `tpt-simd-math` | Vectorised `f32` elementary functions |
| `matrix` | `tpt-simd-matrix` | Small fixed-size matrix kernels |
| `mul` | `tpt-simd-mul` | Multiplies (high/low, widening, complex) |
| `permute` | `tpt-simd-permute` | Permutes and shuffles |
| `reduce` | `tpt-simd-reduce` | Reductions |
| `rng` | `tpt-simd-rng` | Random number generation |
| `rounding` | `tpt-simd-rounding` | Rounding |
| `saturate` | `tpt-simd-saturate` | Saturating arithmetic |
| `scatter` | `tpt-simd-scatter` | Scatter |
| `select` | `tpt-simd-select` | Selection |
| `shift` | `tpt-simd-shift` | Shifts |
| `sparse` | `tpt-simd-sparse` | Sparse kernels |
| `window` | `tpt-simd-window` | Window function generation and application |

`tpt-simd-vector` is reached through the root re-exports of `tpt-simd-core`; `tpt-simd-testutil` is unpublished and not re-exported.

## Feature flags

| Feature | Default | Effect |
| --- | --- | --- |
| `std` | no | Forwards to every member crate that has a `std` feature (runtime CPU feature queries, `std` float intrinsics). |
| `nightly` | no | Forwards to `tpt-simd-core/nightly` (reserved; currently a no-op). |
| `scalar-only` | no | Forwards to every member crate that has a `scalar-only` feature, forcing the portable path. |
| `runtime-dispatch` | **yes** | Implies `std` and enables runtime AVX2+FMA detection in `tpt-simd-blas`, `tpt-simd-hash`, `tpt-simd-text` and `tpt-simd-search` (ADR 0003). |

The default feature set is `runtime-dispatch`, which needs `std`. For a `no_std` build, depend on `tpt-simd` with `default-features = false`; kernels then use only the compile-time `cfg(target_feature)` paths.

## Platform and accuracy notes

- Targets: x86/x86_64 (SSE2, AVX2, AVX-512 where a kernel crate has a path for it), aarch64 (via LLVM lowering of the portable backend) and a portable fallback everywhere else. Build with `RUSTFLAGS="-C target-cpu=native"` (or explicit `-C target-feature`) to use wider vectors, or enable `runtime-dispatch` for crates that support it.
- Accuracy follows the tiers in ADR 0003: integer, saturating, shuffling and similar crates are bit-exact against the scalar reference; `blas`, `reduce`, `math` and `sparse` document tolerances; `rng` is bit-reproducible per seed.

## Performance

The umbrella crate has no benchmarks of its own. Most member crates do; run `cargo bench -p <crate>` (for example `cargo bench -p tpt-simd-dot`). Results and methodology are in [../docs/benchmarks.md](../docs/benchmarks.md).

## Related crates

All `tpt-simd-*` crates in the workspace, listed in the module map above.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
