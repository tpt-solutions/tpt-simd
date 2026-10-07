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
    wrap!(
        single, f32, axpy_f32, dot_f32, nrm2_f32, asum_f32, gemv_f32, gemv_t_f32, gemm_f32
    );
    wrap!(
        double, f64, axpy_f64, dot_f64, nrm2_f64, asum_f64, gemv_f64, gemv_t_f64, gemm_f64
    );
}

#[cfg(all(test, feature = "alloc"))]
mod tests;
