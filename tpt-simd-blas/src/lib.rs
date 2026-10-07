//! BLAS-style `f32`/`f64` kernels: level-1 (`axpy`, `scal`, `dot`, `nrm2`,
//! `asum`), column-major `gemv` (plain and transposed) and a packed,
//! register- and cache-blocked `gemm`.
//!
//! All matrices are **column-major** with an explicit leading dimension
//! (`lda >= rows`), matching BLAS and the storage used by `tpt-math`. Vectors
//! are contiguous slices.
//!
//! | function | computes |
//! |---|---|
//! | `axpy_*` | `y += alpha * x` |
//! | `scal_*` | `x *= alpha` |
//! | `dot_*` | `x . y` |
//! | `nrm2_*` | `sqrt(x . x)` (no overflow scaling) |
//! | `asum_*` | `sum |x_i|` |
//! | `gemv_*` | `y = alpha * A x + beta * y` |
//! | `gemv_t_*` | `y = alpha * A^T x + beta * y` |
//! | `gemm_*` | `C = alpha * A B + beta * C` |
//! | `getrf_*` | `P A = L U` (blocked, partial pivoting) |
//! | `getrs_*` | solve `A X = B` from the `getrf` factors |
//! | `trsm_*` | `B = alpha * op(A)^-1 B`, `A` triangular (left side) |
//! | `potrf_*` | `A = L L^T` (blocked Cholesky, lower) |
//! | `potrs_*` | solve `A X = B` from the `potrf` factor |
//!
//! ## Accumulation order and tolerance (ADR 0001 relaxation)
//!
//! ADR 0001 asks for bit-identical results across paths. That is **not** met
//! here, by design: reductions (`dot`, `nrm2`, `asum`, `gemv_t`, `gemm`) are
//! reassociated and, on the AVX2+FMA path, fused. Results may differ from the
//! naive left-to-right scalar [`crate::reference`] by a few ulps relative to
//! `sum |a_i||b_i|` (the usual forward error bound `~ n * eps`). Specifics:
//!
//! * `dot`/`nrm2`/`asum`: `W` independent partial sums (`W` = 32 for `f32`,
//!   16 for `f64`), a fixed pairwise reduction, then the tail added last. The
//!   order depends only on the length, never on the target.
//! * `gemv`: four columns are combined per pass; `gemv_t` uses `dot`.
//! * `gemm`: each `k`-block (256) is summed in order by the microkernel
//!   (FMA on AVX2+FMA builds, separate multiply and add otherwise), scaled by
//!   `alpha` and added to `C`. The portable and AVX2 paths therefore differ
//!   by FMA rounding only.
//!
//! ## BLAS special cases
//!
//! * `beta == 0` **overwrites** the output: existing NaN/inf in `y`/`C` are
//!   not propagated. `beta == 1` leaves it untouched.
//! * `alpha == 0` skips reading the inputs (`axpy` is a no-op; `gemv`/`gemm`
//!   reduce to `y = beta * y`), so NaN/inf in `A`, `B`, `x` are not
//!   propagated in that case. `scal` with `alpha == 0` writes zeros.
//! * `k == 0` (`gemm`) likewise gives `C = beta * C`. Zero dimensions are
//!   valid and never read the operands.
//!
//! ## Dispatch
//!
//! Per ADR 0001 the AVX2+FMA `gemm` microkernel (16x4 `f32`, 8x4 `f64`) is
//! chosen at compile time with
//! `cfg(all(target_feature = "avx2", target_feature = "fma"))` (build with
//! `-C target-cpu=native`); otherwise a portable lane-array kernel is used
//! (LLVM auto-vectorises it to the baseline ISA, but without FMA). The
//! `scalar-only` feature forces the portable kernel. The level-1 and `gemv`
//! routines are portable code written to auto-vectorise and use no
//! intrinsics. `nrm2` uses `libm::sqrt`; the `std` feature is forwarded to
//! `tpt-simd-core` only.
//!
//! ## Packing buffers
//!
//! With the default `alloc` feature, `gemm_*` allocates its packing buffers
//! per call. For `no_std` without alloc, or to reuse a buffer across calls,
//! use `gemm_with_workspace_*` and size the buffer with
//! `gemm_workspace_len_*`.
//!
//! ## Factorisations
//!
//! `getrf`/`getrs`/`trsm`/`potrf`/`potrs` follow LAPACK conventions
//! (column-major, leading dimensions, in place):
//!
//! * `getrf(m, n, a, lda, ipiv)` overwrites `A` (`m x n`) with the unit-lower
//!   `L` (below the diagonal) and `U`. `ipiv[i]` (zero-based, `>= i`) is the
//!   row swapped with row `i`; `ipiv` needs `min(m, n)` entries. An exactly
//!   zero pivot gives `Err(SingularError { index })` for the first such
//!   column, but the factorisation is still completed (zero multipliers).
//! * `getrs(n, nrhs, a, lda, ipiv, b, ldb)` solves `A X = B` in place (no
//!   transpose). It does not check for singular `U`: a zero diagonal yields
//!   inf/NaN, as in LAPACK. An inverse is `getrs` on the identity.
//! * `trsm(uplo, trans, diag, m, n, alpha, a, lda, b, ldb)` solves
//!   `op(A) X = alpha B` (left side) for `m x m` triangular `A`. Only the
//!   selected triangle is read (and never the diagonal for [`Diag::Unit`]).
//!   `alpha == 0` writes zeros. A zero diagonal divides by zero (inf/NaN).
//! * `potrf(n, a, lda)` factors a symmetric positive definite matrix from its
//!   **lower** triangle only; the strictly upper triangle is neither read nor
//!   written. A pivot that is `<= 0` or NaN gives
//!   `Err(NotPositiveDefiniteError { index })` and stops (the rest of `A` is
//!   unspecified). `potrs(n, nrhs, a, lda, b, ldb)` solves with that factor.
//!
//! The blocked algorithms (right-looking, panel width 32) update the trailing
//! matrix with the packed `gemm` kernel, so they use the AVX2+FMA
//! microkernel automatically (compile-time or `runtime-dispatch`). `trsm`
//! uses 64-wide diagonal blocks solved with `axpy`/`dot` and updates the rest
//! with `gemm` (`gemv` for a single right-hand side). Panel factorisations
//! are unblocked `axpy` column updates.
//!
//! **Tolerance.** Like `gemm`, the updates are reassociated, so factors differ
//! from the naive [`reference`] versions (plain left-to-right sums) by
//! rounding. The usual backward-error bounds hold: for `getrf` with partial
//! pivoting `|P A - L U| <= c n eps |L||U|` (growth-factor dependent), for
//! `potrf` `|A - L L^T| <= c n eps |L||L^T|`. Pivot choices can differ from
//! the reference only for (near-)ties. Compare solutions by the residual
//! `||A x - b|| / (||A|| ||x||)`, not bitwise.
//!
//! **NaN/inf.** There is no NaN/inf screening. A NaN in a `getrf` pivot
//! column is never chosen as pivot unless it is the first candidate, and it
//! propagates through the updates (no panic); `potrf` reports a NaN pivot as
//! `NotPositiveDefiniteError`. `axpy`-style updates skip zero multipliers, so
//! a NaN/inf multiplied by an exact zero multiplier is not propagated (same
//! rule as `axpy` with `alpha == 0`).
//!
//! **Workspace.** The `_with_workspace_*` variants take a caller buffer sized
//! by `getrf_workspace_len_*`, `potrf_workspace_len_*` or
//! `trsm_workspace_len_*` (which is also what `getrs`/`potrs` need, with
//! `m = n`); the variants without the suffix allocate it (`alloc` feature).
//! The length is 0 for small problems. Zero dimensions are valid and read
//! nothing.
//!
//! ## Panics
//!
//! Operands too short for their dimensions/leading dimension, `lda < rows`,
//! mismatched vector lengths, or a too-small workspace panic with a
//! descriptive message (ADR 0002).
//!
//! ## Example
//!
//! ```
//! use tpt_simd_blas::{dot_f32, gemm_f32};
//!
//! assert_eq!(dot_f32(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
//!
//! // C (2x2) = A (2x3) * B (3x2), column-major.
//! let a = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
//! let b = [7.0, 9.0, 11.0, 8.0, 10.0, 12.0];
//! let mut c = [0.0f32; 4];
//! gemm_f32(2, 2, 3, 1.0, &a, 2, &b, 3, 0.0, &mut c, 2);
//! assert_eq!(c, [58.0, 139.0, 64.0, 154.0]);
//! ```
#![no_std]
#![cfg_attr(
    not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        any(
            all(target_feature = "avx2", target_feature = "fma"),
            feature = "runtime-dispatch"
        )
    )),
    forbid(unsafe_code)
)]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(any(test, feature = "runtime-dispatch"))]
extern crate std;

mod check;
#[macro_use]
mod imp;
#[macro_use]
mod lapack;
#[macro_use]
mod lapack_api;
pub use lapack::{Diag, NotPositiveDefiniteError, SingularError, Trans, Uplo};
#[cfg(all(
    not(feature = "scalar-only"),
    target_arch = "x86_64",
    any(
        all(target_feature = "avx2", target_feature = "fma"),
        feature = "runtime-dispatch"
    )
))]
mod x86;

// The `kernel` path is only expanded (and resolved) on AVX2+FMA builds.
blas_impl!(
    single,
    f32,
    w = 32,
    mr = 16,
    nr = 4,
    mc = 128,
    kc = 256,
    nc = 512,
    sqrt = libm::sqrtf,
    kernel = crate::x86::kernel_f32
);
blas_impl!(
    double,
    f64,
    w = 16,
    mr = 8,
    nr = 4,
    mc = 64,
    kc = 256,
    nc = 512,
    sqrt = libm::sqrt,
    kernel = crate::x86::kernel_f64
);

lapack_impl!(
    fsingle,
    f32,
    blas = single,
    sqrt = libm::sqrtf,
    nb = 32,
    tb = 64
);
lapack_impl!(
    fdouble,
    f64,
    blas = double,
    sqrt = libm::sqrt,
    nb = 32,
    tb = 64
);
lapack_api!(
    fsingle,
    f32,
    getrf_f32,
    getrf_with_workspace_f32,
    getrf_workspace_len_f32,
    getrs_f32,
    getrs_with_workspace_f32,
    trsm_f32,
    trsm_with_workspace_f32,
    trsm_workspace_len_f32,
    potrf_f32,
    potrf_with_workspace_f32,
    potrf_workspace_len_f32,
    potrs_f32,
    potrs_with_workspace_f32
);
lapack_api!(
    fdouble,
    f64,
    getrf_f64,
    getrf_with_workspace_f64,
    getrf_workspace_len_f64,
    getrs_f64,
    getrs_with_workspace_f64,
    trsm_f64,
    trsm_with_workspace_f64,
    trsm_workspace_len_f64,
    potrf_f64,
    potrf_with_workspace_f64,
    potrf_workspace_len_f64,
    potrs_f64,
    potrs_with_workspace_f64
);

macro_rules! public_api {
    (
        $m:ident, $t:ident, $axpy:ident, $scal:ident, $dot:ident, $nrm2:ident,
        $asum:ident, $gemv:ident, $gemv_t:ident, $gemm:ident, $gemm_ws:ident,
        $ws_len:ident
    ) => {
        #[doc = concat!("`y += alpha * x` (`", stringify!($t), "`). No-op when `alpha == 0`.\n\n# Panics\nIf `x.len() != y.len()`.")]
        pub fn $axpy(alpha: $t, x: &[$t], y: &mut [$t]) {
            $m::axpy(alpha, x, y)
        }
        #[doc = concat!("`x *= alpha` (`", stringify!($t), "`). `alpha == 0` writes zeros (NaN not propagated).")]
        pub fn $scal(alpha: $t, x: &mut [$t]) {
            $m::scal(alpha, x)
        }
        #[doc = concat!("Dot product (`", stringify!($t), "`), reassociated; see the crate docs for the accumulation order. Empty input gives 0.\n\n# Panics\nIf the lengths differ.")]
        pub fn $dot(x: &[$t], y: &[$t]) -> $t {
            $m::dot(x, y)
        }
        #[doc = concat!("Euclidean norm (`", stringify!($t), "`) as `sqrt(sum x_i^2)` with no overflow/underflow scaling: overflows for |x| above roughly `sqrt(MAX)`.")]
        pub fn $nrm2(x: &[$t]) -> $t {
            $m::nrm2(x)
        }
        #[doc = concat!("Sum of absolute values (`", stringify!($t), "`).")]
        pub fn $asum(x: &[$t]) -> $t {
            $m::asum(x)
        }
        #[doc = concat!("Column-major `y = alpha * A x + beta * y` (`", stringify!($t), "`); `A` is `m x n` with leading dimension `lda`, `x` has `n` and `y` has `m` elements. `beta == 0` overwrites `y`.\n\n# Panics\nOn inconsistent dimensions or slice lengths.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $gemv(m: usize, n: usize, alpha: $t, a: &[$t], lda: usize, x: &[$t], beta: $t, y: &mut [$t]) {
            $m::gemv(m, n, alpha, a, lda, x, beta, y)
        }
        #[doc = concat!("Column-major `y = alpha * A^T x + beta * y` (`", stringify!($t), "`); `A` is `m x n` with leading dimension `lda`, `x` has `m` and `y` has `n` elements. `beta == 0` overwrites `y`.\n\n# Panics\nOn inconsistent dimensions or slice lengths.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $gemv_t(m: usize, n: usize, alpha: $t, a: &[$t], lda: usize, x: &[$t], beta: $t, y: &mut [$t]) {
            $m::gemv_t(m, n, alpha, a, lda, x, beta, y)
        }
        #[cfg(feature = "alloc")]
        #[doc = concat!("Packed, blocked column-major `C = alpha * A B + beta * C` (`", stringify!($t), "`). `A` is `m x k` (`lda`), `B` is `k x n` (`ldb`), `C` is `m x n` (`ldc`). `beta == 0` overwrites `C`. Allocates its packing buffers; see the `_with_workspace` variant.\n\n# Panics\nOn inconsistent dimensions or slice lengths.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $gemm(m: usize, n: usize, k: usize, alpha: $t, a: &[$t], lda: usize, b: &[$t], ldb: usize, beta: $t, c: &mut [$t], ldc: usize) {
            $m::gemm(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        }
        #[doc = concat!("Like the allocating `gemm` (`", stringify!($t), "`) but packs into the caller-provided `workspace`, which must hold at least the length returned by the matching `gemm_workspace_len` function.\n\n# Panics\nOn inconsistent dimensions or slice lengths, or a too-small workspace.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $gemm_ws(m: usize, n: usize, k: usize, alpha: $t, a: &[$t], lda: usize, b: &[$t], ldb: usize, beta: $t, c: &mut [$t], ldc: usize, workspace: &mut [$t]) {
            $m::gemm_with_workspace(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, workspace)
        }
        #[doc = concat!("Number of `", stringify!($t), "` elements of packing workspace that `gemm_with_workspace` needs for an `m x k` by `k x n` product (0 if any dimension is 0).")]
        pub fn $ws_len(m: usize, n: usize, k: usize) -> usize {
            $m::gemm_workspace_len(m, n, k)
        }
    };
}

public_api!(
    single,
    f32,
    axpy_f32,
    scal_f32,
    dot_f32,
    nrm2_f32,
    asum_f32,
    gemv_f32,
    gemv_t_f32,
    gemm_f32,
    gemm_with_workspace_f32,
    gemm_workspace_len_f32
);
public_api!(
    double,
    f64,
    axpy_f64,
    scal_f64,
    dot_f64,
    nrm2_f64,
    asum_f64,
    gemv_f64,
    gemv_t_f64,
    gemm_f64,
    gemm_with_workspace_f64,
    gemm_workspace_len_f64
);

/// Naive left-to-right scalar implementations used as the test oracle and
/// benchmark baseline (the "naive triple loop" a plain matrix library uses).
///
/// Same semantics as the fast functions (including `beta == 0` overwriting),
/// but with no blocking, no reassociation and no workspace.
pub mod reference {
    macro_rules! wrap {
        ($m:ident, $t:ident, $axpy:ident, $dot:ident, $nrm2:ident, $asum:ident, $gemv:ident, $gemv_t:ident, $gemm:ident) => {
            #[doc = concat!("Naive `y += alpha * x` (`", stringify!($t), "`).")]
            pub fn $axpy(alpha: $t, x: &[$t], y: &mut [$t]) {
                crate::$m::reference::axpy(alpha, x, y)
            }
            #[doc = concat!("Naive sequential dot product (`", stringify!($t), "`).")]
            pub fn $dot(x: &[$t], y: &[$t]) -> $t {
                crate::$m::reference::dot(x, y)
            }
            #[doc = concat!("Naive `sqrt(sum x^2)` (`", stringify!($t), "`).")]
            pub fn $nrm2(x: &[$t]) -> $t {
                crate::$m::reference::nrm2(x)
            }
            #[doc = concat!("Naive sum of absolute values (`", stringify!($t), "`).")]
            pub fn $asum(x: &[$t]) -> $t {
                crate::$m::reference::asum(x)
            }
            #[doc = concat!("Naive row-by-row `y = alpha * A x + beta * y` (`", stringify!($t), "`).")]
            #[allow(clippy::too_many_arguments)]
            pub fn $gemv(m: usize, n: usize, alpha: $t, a: &[$t], lda: usize, x: &[$t], beta: $t, y: &mut [$t]) {
                crate::$m::reference::gemv(m, n, alpha, a, lda, x, beta, y)
            }
            #[doc = concat!("Naive `y = alpha * A^T x + beta * y` (`", stringify!($t), "`).")]
            #[allow(clippy::too_many_arguments)]
            pub fn $gemv_t(m: usize, n: usize, alpha: $t, a: &[$t], lda: usize, x: &[$t], beta: $t, y: &mut [$t]) {
                crate::$m::reference::gemv_t(m, n, alpha, a, lda, x, beta, y)
            }
            #[doc = concat!("Naive triple-loop `C = alpha * A B + beta * C` (`", stringify!($t), "`), column-major.")]
            #[allow(clippy::too_many_arguments)]
            pub fn $gemm(m: usize, n: usize, k: usize, alpha: $t, a: &[$t], lda: usize, b: &[$t], ldb: usize, beta: $t, c: &mut [$t], ldc: usize) {
                crate::$m::reference::gemm(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            }
        };
    }
    macro_rules! lwrap {
        ($f:ident, $t:ident, $getrf:ident, $getrs:ident, $trsm:ident, $potrf:ident, $potrs:ident) => {
            #[doc = concat!("Naive unblocked LU with partial pivoting (`", stringify!($t), "`); same conventions as the fast `getrf`.")]
            pub fn $getrf(m: usize, n: usize, a: &mut [$t], lda: usize, ipiv: &mut [usize]) -> Result<(), crate::SingularError> {
                crate::$f::reference::getrf(m, n, a, lda, ipiv)
            }
            #[doc = concat!("Naive row swaps plus forward/back substitution (`", stringify!($t), "`).")]
            #[allow(clippy::too_many_arguments)]
            pub fn $getrs(n: usize, nrhs: usize, a: &[$t], lda: usize, ipiv: &[usize], b: &mut [$t], ldb: usize) {
                crate::$f::reference::getrs(n, nrhs, a, lda, ipiv, b, ldb)
            }
            #[doc = concat!("Naive substitution `B = alpha * op(A)^-1 B` (`", stringify!($t), "`), left side.")]
            #[allow(clippy::too_many_arguments)]
            pub fn $trsm(uplo: crate::Uplo, trans: crate::Trans, diag: crate::Diag, m: usize, n: usize, alpha: $t, a: &[$t], lda: usize, b: &mut [$t], ldb: usize) {
                crate::$f::reference::trsm(uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
            }
            #[doc = concat!("Naive row-oriented lower Cholesky (`", stringify!($t), "`).")]
            pub fn $potrf(n: usize, a: &mut [$t], lda: usize) -> Result<(), crate::NotPositiveDefiniteError> {
                crate::$f::reference::potrf(n, a, lda)
            }
            #[doc = concat!("Naive solve with the lower Cholesky factor (`", stringify!($t), "`).")]
            pub fn $potrs(n: usize, nrhs: usize, a: &[$t], lda: usize, b: &mut [$t], ldb: usize) {
                crate::$f::reference::potrs(n, nrhs, a, lda, b, ldb)
            }
        };
    }
    lwrap!(
        fsingle, f32, getrf_f32, getrs_f32, trsm_f32, potrf_f32, potrs_f32
    );
    lwrap!(
        fdouble, f64, getrf_f64, getrs_f64, trsm_f64, potrf_f64, potrs_f64
    );
    wrap!(
        single, f32, axpy_f32, dot_f32, nrm2_f32, asum_f32, gemv_f32, gemv_t_f32, gemm_f32
    );
    wrap!(
        double, f64, axpy_f64, dot_f64, nrm2_f64, asum_f64, gemv_f64, gemv_t_f64, gemm_f64
    );
}

#[cfg(all(test, feature = "alloc"))]
mod tests;
#[cfg(all(test, feature = "alloc"))]
mod tests_lapack;
