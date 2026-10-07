# tpt-simd-sparse

Sparse linear-algebra kernels for iterative solvers: CSR and CSC sparse matrix-vector products (plain and transposed) and the fused dense vector updates used by CG and BiCGSTAB, for `f32` and `f64`.

## Overview

Use this crate as the inner-loop kernel set of a Krylov solver (conjugate gradient, BiCGSTAB) or whenever you need `y = alpha*A*x + beta*y` for a sparse `A`.

- SpMV for CSR and CSC matrices, both `A*x` and `Aᵀ*x`.
- Level-1 vector kernels (`dot`, `sqnorm`, `axpy`, `xpay`) and fused update-plus-reduction kernels (`axpy_dot`, `axpy_sqnorm`, `cg_update`, `bicgstab_p_update`) that make one pass over memory instead of several.
- Borrowed `CsrView` / `CscView` with O(nnz) validation, plus (with `alloc`) owned `CsrMatrix` / `CscMatrix` helpers.
- Everything is generic over the sealed `Real` trait (`f32`, `f64`).
- A naive scalar `reference` module for testing.

Scalar versus SIMD: the kernels are safe-looking portable Rust (multi-accumulator loops and 8-lane array chunks) that LLVM auto-vectorises; there are no hand-written intrinsics. Dispatch is compile-time (see [ADR 0001](../docs/adr/0001-backend-and-dispatch.md) and [ADR 0003](../docs/adr/0003-runtime-dispatch-and-float-tolerance.md)), so build with `-C target-cpu=native` for best results. SpMV is memory-bound; measured gains over a naive loop are modest (roughly 1.0-2x), while the fused vector updates give about 1.5x over separate passes in cache.

`no_std`: the crate is `#![no_std]`. Owned matrices need the default `alloc` feature.

## Installation

```toml
[dependencies]
tpt-simd-sparse = "0.1.0"
```

The same functionality is available through the umbrella crate `tpt-simd` (as `tpt_simd::sparse`).

## Quick start

```rust
use tpt_simd_sparse::{CsrMatrix, spmv_csr};

// [[2, 0], [1, 3]]
let a = CsrMatrix::<f64>::from_triplets(2, 2, &[(0, 0, 2.0), (1, 0, 1.0), (1, 1, 3.0)]).unwrap();
let mut y = [f64::NAN; 2]; // beta = 0 overwrites, NaN does not leak
spmv_csr(1.0, &a.view(), &[1.0, 2.0], 0.0, &mut y);
assert_eq!(y, [2.0, 7.0]);
```

A complete conjugate-gradient solver on the 2D Poisson matrix is in `examples/cg.rs` (`cargo run --release -p tpt-simd-sparse --example cg`).

## API overview

SpMV (`y = alpha*op(A)*x + beta*y`)

| Function | Description |
|---|---|
| `spmv_csr` | `A*x` for CSR (gather style, one sparse dot per row) |
| `spmv_csr_with` | `spmv_csr` with an explicit `RowStrategy` (for experiments and benchmarks) |
| `spmv_csc_t` | `Aᵀ*x` for CSC (gather style) |
| `spmv_csr_t` | `Aᵀ*x` for CSR (scatter style) |
| `spmv_csc` | `A*x` for CSC (scatter style) |
| `RowStrategy`, `DEFAULT_STRATEGY` | `Scalar`, `Lanes4`, `Lanes8`, `Hybrid`; default is `Lanes4` |

Vector kernels

| Function | Description |
|---|---|
| `dot`, `sqnorm` | Dot product and squared 2-norm |
| `axpy`, `xpay` | `y += alpha*x` and `y = x + alpha*y` |
| `axpy_dot`, `axpy_sqnorm` | `axpy` fused with a dot product / squared norm |
| `cg_update` | The two CG updates plus the residual norm in one pass |
| `bicgstab_p_update` | `p = r + beta*(p - omega*v)` |

Types

| Type | Description |
|---|---|
| `CsrView`, `CscView` | Borrowed views: `try_new` (validated), `new_unchecked` (unsafe), `nnz`, `nrows`, `ncols`, `indptr`, `indices`, `data`. Note `CscView::try_new` takes `(ncols, nrows, ...)` (major dimension first) |
| `CsrMatrix`, `CscMatrix` | Owned (feature `alloc`). `CsrMatrix`: `try_new`, `from_triplets`, `poisson_2d`, `view`, `to_csc`, `nrows`, `ncols`. `CscMatrix`: `view`, `nrows`, `ncols` (built with `CsrMatrix::to_csc`) |
| `SparseError` | Validation errors (`IndptrLength`, `IndptrStart`, `IndptrNotMonotone`, `NnzMismatch`, `IndexOutOfRange`, `DimensionTooLarge`) |
| `Real` | Sealed scalar trait for `f32` and `f64` |
| `reference` | Naive scalar versions of the SpMV and vector kernels |

Semantics

- Indices within a row/column may be unsorted; duplicate `(row, col)` entries are summed; explicit zeros are ordinary entries. Column indices are `u32`, `indptr` is `usize`.
- Only `beta == 0` is special: `y` is overwritten without being read (NaN/inf in `y` do not leak). `alpha == 0` is not special-cased, so NaN/inf in `A` or `x` still propagate.
- Kernels panic on mismatched `x`/`y` lengths.

## Feature flags

| Feature | Effect |
|---|---|
| `default` | `alloc` |
| `alloc` | Enables `CsrMatrix` / `CscMatrix` (uses `alloc`; still `no_std`) |
| `std` | Implies `alloc`; forwards to `tpt-simd-core/std`; implements `std::error::Error` for `SparseError` |
| `nightly` | Forwards to `tpt-simd-core/nightly` |
| `scalar-only` | Forwards to `tpt-simd-core/scalar-only` |

## Platform support, accuracy and safety

- All targets; vector code generation (x86 SSE/AVX2/AVX-512, aarch64 NEON) is chosen by LLVM from the enabled target features. There is no runtime dispatch in this crate.
- No FMA is used and the accumulation order depends only on the code, so results are reproducible across targets and feature sets. The order differs from a naive left-to-right loop by a few ulps of the sum of absolute terms; compare against `reference` with a tolerance relative to that sum. The fused functions are bit-identical to the unfused pass sequence.
- `unsafe`: the kernels use unchecked loads of `x[idx]`, which is sound because view fields are private and every view is validated (or created through the `unsafe` `new_unchecked`, whose caller must uphold the invariants).

## Performance

Benchmarks: `cargo bench -p tpt-simd-sparse` (compares row strategies, including a hardware-gather variant that lives only in the bench, against naive references). Measured findings are summarised in the crate documentation and in [../docs/benchmarks.md](../docs/benchmarks.md).

## Related crates

Depends on `tpt-simd-core`; `tpt-simd-gather` is a dev-dependency used by the benchmarks. Siblings: `tpt-simd-blas`, `tpt-simd-reduce`, `tpt-simd-math`. Umbrella: `tpt-simd`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
