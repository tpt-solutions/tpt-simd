//! Sparse linear-algebra kernels for iterative solvers: CSR and CSC sparse
//! matrix-vector products (plain and transposed) and the fused dense vector
//! updates used by CG and BiCGSTAB, for `f32` and `f64`.
//!
//! Everything is generic over the sealed [`Real`] trait (implemented for
//! `f32` and `f64`) instead of `_f32`/`_f64` function pairs. The crate is
//! `no_std`; the owned [`CsrMatrix`]/[`CscMatrix`] helpers need the default
//! `alloc` feature.
//!
//! | Function | Meaning |
//! |---|---|
//! | [`spmv_csr`], [`spmv_csc_t`] | `y = alpha*A*x + beta*y` / `alpha*Aᵀ*x + beta*y`, one sparse dot per line |
//! | [`spmv_csr_t`], [`spmv_csc`] | `alpha*Aᵀ*x + beta*y` / `alpha*A*x + beta*y`, scatter per line |
//! | [`dot`], [`sqnorm`], [`axpy`], [`xpay`] | level-1 vector kernels |
//! | [`axpy_dot`], [`axpy_sqnorm`], [`cg_update`] | fused update + reduction in one pass |
//! | [`bicgstab_p_update`] | `p = r + beta*(p - omega*v)` |
//! | [`mod@reference`] | naive scalar versions of everything above |
//!
//! ```
//! use tpt_simd_sparse::{CsrMatrix, spmv_csr};
//! // [[2, 0], [1, 3]]
//! let a = CsrMatrix::<f64>::from_triplets(2, 2, &[(0, 0, 2.0), (1, 0, 1.0), (1, 1, 3.0)]).unwrap();
//! let mut y = [f64::NAN; 2]; // beta = 0 overwrites, NaN does not leak
//! spmv_csr(1.0, &a.view(), &[1.0, 2.0], 0.0, &mut y);
//! assert_eq!(y, [2.0, 7.0]);
//! ```
//!
//! ## Validation and the unchecked path
//!
//! [`CsrView::try_new`] / [`CscView::try_new`] check, in O(nnz): `indptr`
//! length, `indptr[0] == 0`, monotonicity, `indptr[last] == indices.len() ==
//! data.len()`, and every index `< ncols` (CSR) / `< nrows` (CSC). Invalid
//! input is rejected with a [`SparseError`]; nothing is ever read out of
//! bounds. The view fields are private, so a view always upholds these
//! invariants, which lets the kernels load `x[idx]` without a bounds check.
//! `new_unchecked` (unsafe) skips the scan for data from a trusted builder.
//! Kernels themselves are identical for both constructors; they still
//! **panic** on mismatched `x`/`y` lengths (ADR 0002).
//!
//! ## Index policy
//!
//! * Indices within a row/column may be **unsorted**: not required, not
//!   checked, no effect on correctness.
//! * **Duplicate** `(row, col)` entries are allowed and **summed**, as in
//!   the dense matrix they describe (`A[i][j] = Σ data`).
//! * Explicit zeros are stored entries like any other (`0 * inf = NaN`
//!   propagates, as in IEEE arithmetic).
//! * Column indices are `u32`, so a minor dimension is at most `2^32`;
//!   `indptr` is `usize`.
//!
//! ## `alpha` / `beta` and NaN policy
//!
//! `y = alpha*A*x + beta*y`. Only `beta == 0` is special: `y` is then
//! **overwritten without being read** (BLAS semantics: pre-existing NaN/inf
//! in `y` do not leak). `alpha == 0` is not special-cased: NaN/inf in `A` or
//! `x` still propagate. Everything else is plain IEEE arithmetic.
//!
//! ## Accumulation order
//!
//! No FMA is used and orders depend only on the code, so results are
//! reproducible across targets and feature sets.
//!
//! * Gather-style kernels ([`spmv_csr`], [`spmv_csc_t`]): entry `k` of each
//!   4-chunk of a line goes to accumulator `k % 4`; the accumulators combine
//!   as `(a0 + a2) + (a1 + a3)`, then the `< 4` leftover entries are added
//!   left to right (so lines shorter than 4 are plain left-to-right sums).
//!   The result is then scaled: `alpha * t + beta * y`.
//! * Scatter-style kernels ([`spmv_csr_t`], [`spmv_csc`]): `y` is first
//!   scaled by `beta`, then contributions are added in storage order,
//!   `y[i] += a_ik * (alpha * x_k)`.
//! * Vector reductions ([`dot`] and the fused functions): blocks of 32
//!   elements feed 4 accumulator vectors of 8 lanes (element `32b + 8k + j`
//!   goes to accumulator `k`, lane `j`); these combine as
//!   `(a0 + a1) + (a2 + a3)`, lane-reduce with a halving tree, and the
//!   `< 32` leftover elements are added left to right. The fused functions
//!   use exactly this order for the reduction, so they are bit-identical to
//!   the unfused pass sequence.
//!
//! A naive left-to-right loop differs from these by a few ulps of `Σ|terms|`;
//! compare with a tolerance relative to `Σ|terms|` ([`mod@reference`] is the
//! left-to-right version).
//!
//! ## Measured findings
//!
//! Numbers are from one noisy Windows machine (AVX2, `-C target-cpu=native`,
//! f32, interleaved best-of-150 timings from `examples/strategies.rs`); the
//! absolute values are memory-bound and only the ratios matter.
//!
//! * SpMV is bandwidth/latency bound: **~1.0-1.4x** over the naive safe
//!   loop for CSR, up to ~2x on long rows (128 nnz/row), 0.4-0.6 ns per
//!   stored entry for every strategy.
//! * Rows are independent, so the out-of-order core already overlaps the
//!   add chains of neighbouring rows; extra accumulators help mostly on
//!   long rows. `Lanes4` was fastest or tied everywhere and is the default.
//!   `Lanes8` and the scalar-short-row `Hybrid` gave no gain; the scalar
//!   single accumulator is slowest on Poisson (0.58 vs 0.41 ns/nnz).
//! * Dropping the bounds check on `x[idx]` (sound because views are
//!   validated) made no measurable difference: the one-accumulator kernel
//!   with unchecked loads times the same as the safe-indexing reference
//!   (0.575 vs 0.573 ns/nnz on Poisson).
//! * A hardware-gather row kernel (`vgatherdps` through `tpt-simd-gather`,
//!   bench-only code) came out 5-15% *faster* than `Lanes4` on rows of 12
//!   or more entries when `x` fits in L2, and slower or equal on 5-entry
//!   rows. It is not adopted: f32 only (the gather crate has no `f64`),
//!   AVX2 only, needs `unsafe` and `u32 -> i32` index conversion, and the
//!   gain is within the run-to-run noise of this machine. It is an
//!   optional follow-up, not a regression: the earlier micro-benchmark in
//!   `docs/benchmarks.md` (gather slower than scalar loads) was an isolated
//!   gather loop, where the multiply-accumulate and loop overhead are not
//!   amortised the way they are in SpMV.
//! * Transposed/CSC products (scatter) run at 0.6-1.0 ns/nnz, 1.2-1.6x
//!   slower than the gather-style product because of the read-modify-write
//!   of `y`; `spmv_csr_t` is 1.0-2.1x faster than the naive scatter
//!   reference (long rows gain most).
//! * Fused vector updates matter: `cg_update` (two axpy + norm in one
//!   pass) is ~1.5x faster than three separate passes in cache and ~1.1x
//!   from memory (4 Mi elements), ~3x vs naive scalar passes in cache.
//!   (Do not write the multi-accumulator state as `[[T; 8]; 4]`: LLVM kept
//!   it in memory and the fused loops ran ~10x slower than a flat `[T; 32]`.)
#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "std")]
extern crate std;

mod real;
mod spmv;
mod vec;
mod view;

pub mod reference;

#[cfg(test)]
mod tests;

pub use real::Real;
pub use spmv::{
    DEFAULT_STRATEGY, RowStrategy, spmv_csc, spmv_csc_t, spmv_csr, spmv_csr_t, spmv_csr_with,
};
pub use vec::{axpy, axpy_dot, axpy_sqnorm, bicgstab_p_update, cg_update, dot, sqnorm, xpay};
#[cfg(feature = "alloc")]
pub use view::{CscMatrix, CsrMatrix};
pub use view::{CscView, CsrView, SparseError};
