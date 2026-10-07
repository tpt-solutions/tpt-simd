//! Macros that expand to the public complex API (and its `reference` twins)
//! at the crate root, wrapping the per-precision modules from `complex.rs`.

macro_rules! complex_api {
    (
        $m:ident, $t:ident, $tn:literal,
        $axpy:ident, $scal:ident, $dotu:ident, $dotc:ident, $nrm2:ident, $asum:ident,
        $gemv:ident, $gemv_t:ident, $gemv_h:ident, $gemm:ident, $gemm_ws:ident, $ws_len:ident,
        $axpy_il:ident, $scal_il:ident, $dotu_il:ident, $dotc_il:ident, $nrm2_il:ident,
        $asum_il:ident, $gemv_il:ident, $gemv_t_il:ident, $gemv_h_il:ident, $gemm_il:ident,
        $deint:ident, $inter:ident
    ) => {
        #[doc = concat!("Complex `y += alpha * x` (", $tn, ", **split planes**: `xr`/`xi` and `yr`/`yi` are the real and imaginary parts). `alpha = [re, im]`. No-op when `alpha == 0`.\n\n# Panics\nIf any of the four slices differ in length.")]
        pub fn $axpy(alpha: [$t; 2], xr: &[$t], xi: &[$t], yr: &mut [$t], yi: &mut [$t]) {
            $m::axpy(alpha, xr, xi, yr, yi)
        }
        #[doc = concat!("Complex `x *= alpha` (", $tn, ", split planes). `alpha == 1` is a no-op; `alpha == 0` writes zeros (NaN/inf not propagated).\n\n# Panics\nIf `xr.len() != xi.len()`.")]
        pub fn $scal(alpha: [$t; 2], xr: &mut [$t], xi: &mut [$t]) {
            $m::scal(alpha, xr, xi)
        }
        #[doc = concat!("Unconjugated complex dot product `sum x_i * y_i` (", $tn, ", split planes), returned as `[re, im]`. Reassociated; empty input gives 0.\n\n# Panics\nIf the slice lengths differ.")]
        pub fn $dotu(xr: &[$t], xi: &[$t], yr: &[$t], yi: &[$t]) -> [$t; 2] {
            $m::dot(false, xr, xi, yr, yi)
        }
        #[doc = concat!("Conjugated complex dot product `sum conj(x_i) * y_i` (", $tn, ", split planes; the **first** argument is conjugated, as in BLAS `dotc` and `tpt-math`'s `ComplexDVector::dot`), returned as `[re, im]`.\n\n# Panics\nIf the slice lengths differ.")]
        pub fn $dotc(xr: &[$t], xi: &[$t], yr: &[$t], yi: &[$t]) -> [$t; 2] {
            $m::dot(true, xr, xi, yr, yi)
        }
        #[doc = concat!("Euclidean norm `sqrt(sum re^2 + im^2)` (", $tn, ", split planes), with no overflow/underflow scaling.\n\n# Panics\nIf `xr.len() != xi.len()`.")]
        pub fn $nrm2(xr: &[$t], xi: &[$t]) -> $t {
            $m::nrm2(xr, xi)
        }
        #[doc = concat!("BLAS-style complex `asum` `sum |re_i| + |im_i|` (", $tn, ", split planes; the cheap 1-norm, not `sum |z_i|`).\n\n# Panics\nIf `xr.len() != xi.len()`.")]
        pub fn $asum(xr: &[$t], xi: &[$t]) -> $t {
            $m::asum(xr, xi)
        }
        #[doc = concat!("Column-major complex `y = alpha * A x + beta * y` (", $tn, ", split planes). `A` is `m x n`; the real/imaginary planes `ar`/`ai` share the leading dimension `lda` (in elements). `x` has `n` and `y` has `m` elements. `beta == 0` overwrites `y`; `alpha == 0` skips reading `A` and `x`.\n\n# Panics\nOn inconsistent dimensions or slice lengths.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $gemv(m: usize, n: usize, alpha: [$t; 2], ar: &[$t], ai: &[$t], lda: usize, xr: &[$t], xi: &[$t], beta: [$t; 2], yr: &mut [$t], yi: &mut [$t]) {
            $m::gemv(m, n, alpha, ar, ai, lda, xr, xi, beta, yr, yi)
        }
        #[doc = concat!("Column-major complex `y = alpha * A^T x + beta * y` (", $tn, ", split planes, **no** conjugation). `x` has `m` and `y` has `n` elements.\n\n# Panics\nOn inconsistent dimensions or slice lengths.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $gemv_t(m: usize, n: usize, alpha: [$t; 2], ar: &[$t], ai: &[$t], lda: usize, xr: &[$t], xi: &[$t], beta: [$t; 2], yr: &mut [$t], yi: &mut [$t]) {
            $m::gemv_t(false, m, n, alpha, ar, ai, lda, xr, xi, beta, yr, yi)
        }
        #[doc = concat!("Column-major complex `y = alpha * A^H x + beta * y` (", $tn, ", split planes, conjugate transpose). `x` has `m` and `y` has `n` elements.\n\n# Panics\nOn inconsistent dimensions or slice lengths.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $gemv_h(m: usize, n: usize, alpha: [$t; 2], ar: &[$t], ai: &[$t], lda: usize, xr: &[$t], xi: &[$t], beta: [$t; 2], yr: &mut [$t], yi: &mut [$t]) {
            $m::gemv_t(true, m, n, alpha, ar, ai, lda, xr, xi, beta, yr, yi)
        }
        #[cfg(feature = "alloc")]
        #[doc = concat!("Complex `C = alpha * A B + beta * C` (", $tn, ", split planes, column-major) as four real packed `gemm` calls (the \"4M\" method), so it uses the AVX2+FMA microkernel when available. `A` is `m x k` (`lda`), `B` is `k x n` (`ldb`), `C` is `m x n` (`ldc`); each complex matrix is a pair of planes sharing one leading dimension. `beta == 0` overwrites `C`; `alpha == 0` or `k == 0` give `C = beta * C`. Allocates its workspace (including a `2 k n` copy of `alpha * B` when `alpha != 1`); see the `_with_workspace` variant.\n\n# Panics\nOn inconsistent dimensions or slice lengths.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $gemm(m: usize, n: usize, k: usize, alpha: [$t; 2], ar: &[$t], ai: &[$t], lda: usize, br: &[$t], bi: &[$t], ldb: usize, beta: [$t; 2], cr: &mut [$t], ci: &mut [$t], ldc: usize) {
            $m::gemm(m, n, k, alpha, ar, ai, lda, br, bi, ldb, beta, cr, ci, ldc)
        }
        #[doc = concat!("Like the allocating complex `gemm` (", $tn, ") but uses the caller-provided `workspace`, which must hold at least `gemm_workspace_len` elements (that length suffices for any `alpha`; with `alpha == 1` only the real packing part is used).\n\n# Panics\nOn inconsistent dimensions or slice lengths, or a too-small workspace.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $gemm_ws(m: usize, n: usize, k: usize, alpha: [$t; 2], ar: &[$t], ai: &[$t], lda: usize, br: &[$t], bi: &[$t], ldb: usize, beta: [$t; 2], cr: &mut [$t], ci: &mut [$t], ldc: usize, workspace: &mut [$t]) {
            $m::gemm_with_workspace(m, n, k, alpha, ar, ai, lda, br, bi, ldb, beta, cr, ci, ldc, workspace)
        }
        #[doc = concat!("Number of ", $tn, " elements of workspace that complex `gemm_with_workspace` needs for any `alpha` (0 if any dimension is 0): the real packing buffers plus `2 k n` for the scaled copy of `B`.")]
        pub fn $ws_len(m: usize, n: usize, k: usize) -> usize {
            $m::gemm_workspace_len(m, n, k)
        }

        #[doc = concat!("Copies interleaved `[re, im]` pairs into split planes (", $tn, "). Layout adapter; one pass over the data.\n\n# Panics\nIf `re` or `im` is not the same length as `src`.")]
        pub fn $deint(src: &[[$t; 2]], re: &mut [$t], im: &mut [$t]) {
            $m::deinterleave(src, re, im)
        }
        #[doc = concat!("Copies split planes into interleaved `[re, im]` pairs (", $tn, "). Layout adapter; one pass over the data.\n\n# Panics\nIf `re` or `im` is not the same length as `dst`.")]
        pub fn $inter(re: &[$t], im: &[$t], dst: &mut [[$t; 2]]) {
            $m::interleave(re, im, dst)
        }

        #[doc = concat!("Complex `y += alpha * x` on **interleaved** data (", $tn, "; each element is `[re, im]`, the memory layout of a `#[repr(C)]` complex struct or C `_Complex`). No-op when `alpha == 0`.\n\n# Panics\nIf `x.len() != y.len()`.")]
        pub fn $axpy_il(alpha: [$t; 2], x: &[[$t; 2]], y: &mut [[$t; 2]]) {
            $m::axpy_il(alpha, x, y)
        }
        #[doc = concat!("Complex `x *= alpha` on interleaved data (", $tn, "). `alpha == 1` is a no-op; `alpha == 0` writes zeros.")]
        pub fn $scal_il(alpha: [$t; 2], x: &mut [[$t; 2]]) {
            $m::scal_il(alpha, x)
        }
        #[doc = concat!("Unconjugated dot `sum x_i * y_i` on interleaved data (", $tn, ").\n\n# Panics\nIf the lengths differ.")]
        pub fn $dotu_il(x: &[[$t; 2]], y: &[[$t; 2]]) -> [$t; 2] {
            $m::dot_il(false, x, y)
        }
        #[doc = concat!("Conjugated dot `sum conj(x_i) * y_i` on interleaved data (", $tn, "; first argument conjugated).\n\n# Panics\nIf the lengths differ.")]
        pub fn $dotc_il(x: &[[$t; 2]], y: &[[$t; 2]]) -> [$t; 2] {
            $m::dot_il(true, x, y)
        }
        #[doc = concat!("Euclidean norm of interleaved complex data (", $tn, "), no overflow scaling.")]
        pub fn $nrm2_il(x: &[[$t; 2]]) -> $t {
            $m::nrm2_il(x)
        }
        #[doc = concat!("`sum |re_i| + |im_i|` of interleaved complex data (", $tn, ").")]
        pub fn $asum_il(x: &[[$t; 2]]) -> $t {
            $m::asum_il(x)
        }
        #[doc = concat!("Complex `y = alpha * A x + beta * y` on interleaved data (", $tn, "); column-major `A` (`m x n`, `lda` in complex elements). In place, no conversion.\n\n# Panics\nOn inconsistent dimensions or slice lengths.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $gemv_il(m: usize, n: usize, alpha: [$t; 2], a: &[[$t; 2]], lda: usize, x: &[[$t; 2]], beta: [$t; 2], y: &mut [[$t; 2]]) {
            $m::gemv_il(m, n, alpha, a, lda, x, beta, y)
        }
        #[doc = concat!("Complex `y = alpha * A^T x + beta * y` on interleaved data (", $tn, "), no conjugation.\n\n# Panics\nOn inconsistent dimensions or slice lengths.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $gemv_t_il(m: usize, n: usize, alpha: [$t; 2], a: &[[$t; 2]], lda: usize, x: &[[$t; 2]], beta: [$t; 2], y: &mut [[$t; 2]]) {
            $m::gemv_t_il(false, m, n, alpha, a, lda, x, beta, y)
        }
        #[doc = concat!("Complex `y = alpha * A^H x + beta * y` on interleaved data (", $tn, "), conjugate transpose.\n\n# Panics\nOn inconsistent dimensions or slice lengths.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $gemv_h_il(m: usize, n: usize, alpha: [$t; 2], a: &[[$t; 2]], lda: usize, x: &[[$t; 2]], beta: [$t; 2], y: &mut [[$t; 2]]) {
            $m::gemv_t_il(true, m, n, alpha, a, lda, x, beta, y)
        }
        #[cfg(feature = "alloc")]
        #[doc = concat!("Complex `C = alpha * A B + beta * C` on **interleaved** column-major data (", $tn, "). Converts `A`, `B` (and `C` unless `beta == 0`) to split planes, calls the split gemm and converts `C` back. The conversion is `O(mk + kn + mn)` copies plus matching allocations against `O(mnk)` flops: noise for large products, visible for small or skinny ones. Use the split entry points to avoid it.\n\n# Panics\nOn inconsistent dimensions or slice lengths.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $gemm_il(m: usize, n: usize, k: usize, alpha: [$t; 2], a: &[[$t; 2]], lda: usize, b: &[[$t; 2]], ldb: usize, beta: [$t; 2], c: &mut [[$t; 2]], ldc: usize) {
            $m::gemm_il(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        }
    };
}

macro_rules! complex_ref_api {
    ($m:ident, $t:ident, $axpy:ident, $scal:ident, $dotu:ident, $dotc:ident, $nrm2:ident, $asum:ident, $gemv:ident, $gemv_t:ident, $gemv_h:ident, $gemm:ident) => {
        #[doc = concat!("Naive complex `y += alpha * x` (`", stringify!($t), "`, split planes).")]
        pub fn $axpy(alpha: [$t; 2], xr: &[$t], xi: &[$t], yr: &mut [$t], yi: &mut [$t]) {
            crate::$m::reference::axpy(alpha, xr, xi, yr, yi)
        }
        #[doc = concat!("Naive complex `x *= alpha` (`", stringify!($t), "`, split planes; no special cases, so NaN/inf propagate).")]
        pub fn $scal(alpha: [$t; 2], xr: &mut [$t], xi: &mut [$t]) {
            crate::$m::reference::scal(alpha, xr, xi)
        }
        #[doc = concat!("Naive sequential unconjugated complex dot (`", stringify!($t), "`, split planes).")]
        pub fn $dotu(xr: &[$t], xi: &[$t], yr: &[$t], yi: &[$t]) -> [$t; 2] {
            crate::$m::reference::dot(false, xr, xi, yr, yi)
        }
        #[doc = concat!("Naive sequential conjugated complex dot (`", stringify!($t), "`, split planes).")]
        pub fn $dotc(xr: &[$t], xi: &[$t], yr: &[$t], yi: &[$t]) -> [$t; 2] {
            crate::$m::reference::dot(true, xr, xi, yr, yi)
        }
        #[doc = concat!("Naive complex `sqrt(sum re^2 + im^2)` (`", stringify!($t), "`).")]
        pub fn $nrm2(xr: &[$t], xi: &[$t]) -> $t {
            crate::$m::reference::nrm2(xr, xi)
        }
        #[doc = concat!("Naive complex `sum |re| + |im|` (`", stringify!($t), "`).")]
        pub fn $asum(xr: &[$t], xi: &[$t]) -> $t {
            crate::$m::reference::asum(xr, xi)
        }
        #[doc = concat!("Naive row-by-row complex `y = alpha * A x + beta * y` (`", stringify!($t), "`, split planes).")]
        #[allow(clippy::too_many_arguments)]
        pub fn $gemv(m: usize, n: usize, alpha: [$t; 2], ar: &[$t], ai: &[$t], lda: usize, xr: &[$t], xi: &[$t], beta: [$t; 2], yr: &mut [$t], yi: &mut [$t]) {
            crate::$m::reference::gemv(0, m, n, alpha, ar, ai, lda, xr, xi, beta, yr, yi)
        }
        #[doc = concat!("Naive complex `y = alpha * A^T x + beta * y` (`", stringify!($t), "`, split planes).")]
        #[allow(clippy::too_many_arguments)]
        pub fn $gemv_t(m: usize, n: usize, alpha: [$t; 2], ar: &[$t], ai: &[$t], lda: usize, xr: &[$t], xi: &[$t], beta: [$t; 2], yr: &mut [$t], yi: &mut [$t]) {
            crate::$m::reference::gemv(1, m, n, alpha, ar, ai, lda, xr, xi, beta, yr, yi)
        }
        #[doc = concat!("Naive complex `y = alpha * A^H x + beta * y` (`", stringify!($t), "`, split planes).")]
        #[allow(clippy::too_many_arguments)]
        pub fn $gemv_h(m: usize, n: usize, alpha: [$t; 2], ar: &[$t], ai: &[$t], lda: usize, xr: &[$t], xi: &[$t], beta: [$t; 2], yr: &mut [$t], yi: &mut [$t]) {
            crate::$m::reference::gemv(2, m, n, alpha, ar, ai, lda, xr, xi, beta, yr, yi)
        }
        #[doc = concat!("Naive triple-loop complex `C = alpha * A B + beta * C` (`", stringify!($t), "`, split planes, column-major).")]
        #[allow(clippy::too_many_arguments)]
        pub fn $gemm(m: usize, n: usize, k: usize, alpha: [$t; 2], ar: &[$t], ai: &[$t], lda: usize, br: &[$t], bi: &[$t], ldb: usize, beta: [$t; 2], cr: &mut [$t], ci: &mut [$t], ldc: usize) {
            crate::$m::reference::gemm(m, n, k, alpha, ar, ai, lda, br, bi, ldb, beta, cr, ci, ldc)
        }
    };
}
