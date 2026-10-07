# tpt-simd-blas

BLAS-style `f32`/`f64` and complex (`c32`/`c64`) kernels: level-1 routines, column-major `gemv`, and a packed, register- and cache-blocked `gemm`.

## Overview

Use this crate when you need fast dense linear algebra primitives without pulling in a native BLAS. It covers level-1 (`axpy`, `scal`, `dot`, `nrm2`, `asum`), level-2 (`gemv`, `gemv_t`) and level-3 (`gemm`) operations for both `f32` and `f64`. All matrices are **column-major** with an explicit leading dimension (`lda >= rows`), matching BLAS; vectors are contiguous slices.

Scalar-vs-SIMD story:

- Level-1 and `gemv` routines are portable code written to auto-vectorise; they use no intrinsics.
- `gemm` packs panels of `A` and `B` and runs a register-blocked microkernel (16x4 for `f32`, 8x4 for `f64`) inside cache blocks (`kc = 256`). On x86_64 with AVX2+FMA the microkernel is an explicit intrinsics kernel; otherwise a portable lane-array kernel is used (auto-vectorised to the baseline ISA, without FMA).
- Dispatch is compile-time by default (ADR 0001): the AVX2+FMA kernel is selected with `cfg(all(target_feature = "avx2", target_feature = "fma"))`, e.g. `-C target-cpu=native`. The opt-in `runtime-dispatch` feature (ADR 0003) instead detects AVX2+FMA once at run time on x86_64, caches the answer, and calls the `#[target_feature]` kernel only after the check. The `scalar-only` feature forces the portable kernel.
- Numerically this crate is Tier 2 in ADR 0003: reductions are reassociated and, on the AVX2+FMA path, fused, so results differ from a naive scalar loop by a few ulps and can differ slightly between the portable and AVX2+FMA paths (FMA rounding only).

The crate is `#![no_std]`. Allocation is optional (the `alloc` feature, on by default).

## Installation

```toml
[dependencies]
tpt-simd-blas = "0.1.0"
```

Enable run-time AVX2+FMA detection without `-C target-cpu`:

```toml
tpt-simd-blas = { version = "0.1.0", features = ["runtime-dispatch"] }
```

The umbrella crate `tpt-simd` re-exports this crate.

## Quick start

```rust
use tpt_simd_blas::{dot_f32, gemm_f32};

assert_eq!(dot_f32(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);

// C (2x2) = A (2x3) * B (3x2), column-major.
let a = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
let b = [7.0, 9.0, 11.0, 8.0, 10.0, 12.0];
let mut c = [0.0f32; 4];
gemm_f32(2, 2, 3, 1.0, &a, 2, &b, 3, 0.0, &mut c, 2);
assert_eq!(c, [58.0, 139.0, 64.0, 154.0]);
```

`no_std` without an allocator, or reusing a packing buffer across calls:

```rust
use tpt_simd_blas::{gemm_with_workspace_f32, gemm_workspace_len_f32};

let (m, n, k) = (2, 2, 3);
let a = [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
let b = [7.0f32, 9.0, 11.0, 8.0, 10.0, 12.0];
let mut c = [0.0f32; 4];
let mut ws = vec![0.0f32; gemm_workspace_len_f32(m, n, k)];
gemm_with_workspace_f32(m, n, k, 1.0, &a, 2, &b, 3, 0.0, &mut c, 2, &mut ws);
assert_eq!(c, [58.0, 139.0, 64.0, 154.0]);
```

## API overview

Every function exists as `<op>_f32` and `<op>_f64`.

| Function | Computes |
|---|---|
| `axpy_*(alpha, x, y)` | `y += alpha * x` (no-op when `alpha == 0`) |
| `scal_*(alpha, x)` | `x *= alpha` (`alpha == 0` writes zeros) |
| `dot_*(x, y)` | `x . y`, reassociated; empty gives 0 |
| `nrm2_*(x)` | `sqrt(x . x)`, no overflow/underflow scaling |
| `asum_*(x)` | `sum abs(x_i)` |
| `gemv_*(m, n, alpha, a, lda, x, beta, y)` | `y = alpha * A x + beta * y` |
| `gemv_t_*(m, n, alpha, a, lda, x, beta, y)` | `y = alpha * A^T x + beta * y` |
| `gemm_*(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)` | `C = alpha * A B + beta * C`; allocates packing buffers (needs `alloc`) |
| `gemm_with_workspace_*(..., c, ldc, workspace)` | Same as `gemm_*` but packs into a caller-provided buffer |
| `gemm_workspace_len_*(m, n, k)` | Number of elements the workspace needs (0 if any dimension is 0) |

The `reference` module provides naive left-to-right scalar versions (`axpy_*`, `dot_*`, `nrm2_*`, `asum_*`, `gemv_*`, `gemv_t_*`, `gemm_*`, plus the split-plane complex `*_c32`/`*_c64` set) with the same semantics. They are the test oracle and benchmark baseline.

BLAS special cases:

- `beta == 0` **overwrites** the output; existing NaN/inf in `y`/`C` are not propagated. `beta == 1` leaves it untouched.
- `alpha == 0` skips reading the inputs (`gemv`/`gemm` reduce to `beta * y` / `beta * C`), so NaN/inf in `A`, `B`, `x` are not propagated in that case.
- `k == 0` in `gemm` gives `C = beta * C`. Zero dimensions are valid and never read the operands.

## Complex kernels (`*_c32`, `*_c64`)

Complex `axpy`, `scal` (complex `alpha`), `dotu`, `dotc` (first argument conjugated), `nrm2`, `asum` (`sum |re| + |im|`, BLAS `scasum`), `gemv` / `gemv_t` / `gemv_h` (plain, transpose, conjugate transpose) and `gemm`, in `f32` (`_c32`) and `f64` (`_c64`). Scalars and complex results are `[re, im]`.

**Layout: both are supported, and the split-plane one is the fast one.**

| Family | Data | Notes |
|---|---|---|
| `<op>_c32` / `<op>_c64` (split) | separate `re` and `im` slices; a matrix is two column-major planes sharing one `lda` | native layout; `gemm` = four real packed `gemm` calls, so it gets the AVX2+FMA microkernel |
| `<op>_il_c32` / `<op>_il_c64` (interleaved) | `&[[T; 2]]` = `re, im, re, im, ...` (layout of a `#[repr(C)]` complex struct / C `_Complex`) | level 1 and `gemv` run in place, no conversion (portable auto-vectorised); `gemm_il_*` converts to planes and back (needs `alloc`) |

`deinterleave_c32` / `interleave_c32` (and `_c64`) convert explicitly.

```rust
use tpt_simd_blas::{dotc_c32, gemm_c32};

// conj(i) * 1 = -i
assert_eq!(dotc_c32(&[0.0], &[1.0], &[1.0], &[0.0]), [0.0, -1.0]);

// C = A B for 1x1 complex matrices: (1+2i)(3+4i) = -5+10i.
let (mut cr, mut ci) = ([0.0f32], [0.0f32]);
gemm_c32(1, 1, 1, [1.0, 0.0], &[1.0], &[2.0], 1, &[3.0], &[4.0], 1, [0.0, 0.0], &mut cr, &mut ci, 1);
assert_eq!((cr[0], ci[0]), (-5.0, 10.0));
```

**Fit with `tpt-math-linalg-complex`.** Its `Complex<T>` is a plain `struct { re, im }` *without* `#[repr(C)]`, stored as a column-major `Vec<Complex<T>>` (`lda = nrows`). Its layout is therefore unspecified and slices of it cannot soundly be cast to `&[[T; 2]]` (tpt-math also forbids `unsafe`). Callers have to copy once, either into planes or into `[T; 2]` pairs, which is `O(n)` for vectors and `O(mn)` for matrices (cheap next to `O(mnk)` gemm, but comparable to the work of a level-1 call, so for a single `dotc` or `axpy` on freshly converted data the conversion dominates; keep data in split or interleaved form across calls). The interleaved storage order matches column-major `ComplexDMatrix` after that copy.

**`gemm` method.** `C = beta*C`, then `Cr += Ar*Br - Ai*Bi` and `Ci += Ar*Bi + Ai*Br` as four real `gemm` calls (4M). No temporary when `alpha == 1`; otherwise one `2*k*n` scaled copy of `B` (workspace: `gemm_workspace_len_c32/c64`, covers any `alpha`). The 3M (Karatsuba) method is not used: it saves 25% of flops but its imaginary part only has a norm-wise error bound. Special cases follow the real routines with complex `alpha`/`beta` compared against exactly `0`/`1` (`beta == 0` overwrites, `alpha == 0` skips the inputs). `nrm2` does no overflow scaling.

## Feature flags

| Feature | Default | Effect |
|---|---|---|
| `alloc` | on | Provides the `gemm_*` entry points that allocate their own packing buffers. Without it, use `gemm_with_workspace_*`. |
| `std` | off | Implies `alloc`; enables `tpt-simd-core/std`. |
| `runtime-dispatch` | off | Implies `std`. On x86_64, detect AVX2+FMA once at run time (no `-C target-cpu` needed) and use the AVX2+FMA `gemm` microkernel when present; see ADR 0003. |
| `nightly` | off | Forwards `tpt-simd-core/nightly` (reserved, currently a no-op in core). |
| `scalar-only` | off | Forwards `tpt-simd-core/scalar-only` and forces the portable `gemm` kernel regardless of CPU features. |

## Platform support and safety

- x86_64: portable path by default; AVX2+FMA `gemm` microkernel when built with those target features or with `runtime-dispatch`. No AVX-512 kernel.
- aarch64 and other targets: portable code only (auto-vectorised by LLVM, so NEON is used where LLVM chooses it); no hand-written NEON kernel.
- Unsafe: only the AVX2+FMA microkernels and the call into them use `unsafe`; they are compiled only on x86_64 builds with AVX2+FMA enabled (or `runtime-dispatch`) and without `scalar-only`. In every other configuration `unsafe_code` is forbidden. Each kernel documents its `# Safety` requirements.
- Accuracy: reductions use `W` independent partial sums (`W = 32` for `f32`, 16 for `f64`) with a fixed pairwise reduction and the tail added last, so the order depends only on the length, never on the target. `gemm` sums each `k`-block (256) in order in the microkernel, scales by `alpha` and adds to `C`. Compare with the `reference` module using a tolerance relative to `sum abs(a_i * b_i)`. With `runtime-dispatch`, the same binary can give slightly different results on machines with and without FMA.
- `nrm2` uses `libm::sqrt`/`libm::sqrtf` and does not scale, so it overflows for `abs(x)` above about `sqrt(MAX)`.
- Panics: operands too short for their dimensions or leading dimension, `lda < rows`, mismatched vector lengths, or a too-small workspace panic with a descriptive message.

## Performance

Benchmarks live in `benches/blas.rs` (gemm, gemv and dot against the naive `reference` loops, reported as flop/s) and `benches/complex.rs` (complex gemm, gemv and level 1, split and interleaved); run them with `cargo bench -p tpt-simd-blas`, adding `RUSTFLAGS="-C target-cpu=native"` to enable the AVX2+FMA kernel. See [docs/benchmarks.md](../docs/benchmarks.md) for recorded results.

## Related crates

- [`tpt-simd-core`](../tpt-simd-core): core traits and feature detection (the only workspace dependency).
- [`tpt-simd-reduce`](../tpt-simd-reduce), [`tpt-simd-dot`](../tpt-simd-dot): slice reductions and dot products with different accumulation contracts.
- [`tpt-simd-matrix`](../tpt-simd-matrix): small fixed-size matrix operations.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
