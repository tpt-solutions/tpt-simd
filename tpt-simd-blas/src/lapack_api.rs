//! Public wrappers for the factorisation kernels (alloc convenience and
//! caller-workspace variants).

macro_rules! lapack_api {
    (
        $f:ident, $t:ident, $getrf:ident, $getrf_ws:ident, $getrf_len:ident,
        $getrs:ident, $getrs_ws:ident, $trsm:ident, $trsm_ws:ident, $trsm_len:ident,
        $potrf:ident, $potrf_ws:ident, $potrf_len:ident, $potrs:ident, $potrs_ws:ident
    ) => {
        #[cfg(feature = "alloc")]
        #[doc = concat!("Blocked LU with partial pivoting (`", stringify!($t), "`) of the `m x n` matrix `a`, in place: `P A = L U` with unit-lower `L`. `ipiv[i]` is the row swapped with row `i` (zero-based); `ipiv.len() >= min(m, n)`. `Err` carries the first exactly-zero pivot (the factorisation is still completed). Allocates its workspace; see the `_with_workspace` variant.\n\n# Panics\nOn inconsistent dimensions or slice lengths.")]
        pub fn $getrf(m: usize, n: usize, a: &mut [$t], lda: usize, ipiv: &mut [usize]) -> Result<(), crate::SingularError> {
            let mut ws = alloc::vec![0.0 as $t; crate::$f::getrf_workspace_len(m, n)];
            crate::$f::getrf_with_workspace(m, n, a, lda, ipiv, &mut ws)
        }
        #[doc = concat!("Like the allocating `getrf` (`", stringify!($t), "`) but with a caller workspace of at least `getrf_workspace_len` elements.\n\n# Panics\nOn inconsistent dimensions or slice lengths, or a too-small workspace.")]
        pub fn $getrf_ws(m: usize, n: usize, a: &mut [$t], lda: usize, ipiv: &mut [usize], workspace: &mut [$t]) -> Result<(), crate::SingularError> {
            crate::$f::getrf_with_workspace(m, n, a, lda, ipiv, workspace)
        }
        #[doc = concat!("Workspace elements (`", stringify!($t), "`) needed by `getrf_with_workspace` for an `m x n` matrix (0 for small matrices).")]
        pub fn $getrf_len(m: usize, n: usize) -> usize {
            crate::$f::getrf_workspace_len(m, n)
        }
        #[cfg(feature = "alloc")]
        #[doc = concat!("Solves `A X = B` (`", stringify!($t), "`, no transpose) in place in `b` (`n x nrhs`, `ldb`) from the factors and pivots of `getrf` (`n x n`, `lda`). A zero `U` diagonal gives inf/NaN. Allocates its workspace.\n\n# Panics\nOn inconsistent dimensions or slice lengths, or an invalid pivot index.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $getrs(n: usize, nrhs: usize, a: &[$t], lda: usize, ipiv: &[usize], b: &mut [$t], ldb: usize) {
            let mut ws = alloc::vec![0.0 as $t; crate::$f::trsm_workspace_len(n, nrhs)];
            crate::$f::getrs_with_workspace(n, nrhs, a, lda, ipiv, b, ldb, &mut ws)
        }
        #[doc = concat!("Like the allocating `getrs` (`", stringify!($t), "`) with a caller workspace of at least `trsm_workspace_len(n, nrhs)` elements.\n\n# Panics\nOn inconsistent dimensions or slice lengths, an invalid pivot index, or a too-small workspace.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $getrs_ws(n: usize, nrhs: usize, a: &[$t], lda: usize, ipiv: &[usize], b: &mut [$t], ldb: usize, workspace: &mut [$t]) {
            crate::$f::getrs_with_workspace(n, nrhs, a, lda, ipiv, b, ldb, workspace)
        }
        #[cfg(feature = "alloc")]
        #[doc = concat!("Left-side triangular solve `B = alpha * op(A)^-1 B` (`", stringify!($t), "`) in place; `A` is `m x m` triangular (`lda`), `B` is `m x n` (`ldb`). Only the `uplo` triangle of `A` is read. Blocked: diagonal blocks by substitution, the rest by `gemm`. Allocates its workspace.\n\n# Panics\nOn inconsistent dimensions or slice lengths.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $trsm(uplo: crate::Uplo, trans: crate::Trans, diag: crate::Diag, m: usize, n: usize, alpha: $t, a: &[$t], lda: usize, b: &mut [$t], ldb: usize) {
            let mut ws = alloc::vec![0.0 as $t; crate::$f::trsm_workspace_len(m, n)];
            crate::$f::trsm_with_workspace(uplo, trans, diag, m, n, alpha, a, lda, b, ldb, &mut ws)
        }
        #[doc = concat!("Like the allocating `trsm` (`", stringify!($t), "`) with a caller workspace of at least `trsm_workspace_len(m, n)` elements.\n\n# Panics\nOn inconsistent dimensions or slice lengths, or a too-small workspace.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $trsm_ws(uplo: crate::Uplo, trans: crate::Trans, diag: crate::Diag, m: usize, n: usize, alpha: $t, a: &[$t], lda: usize, b: &mut [$t], ldb: usize, workspace: &mut [$t]) {
            crate::$f::trsm_with_workspace(uplo, trans, diag, m, n, alpha, a, lda, b, ldb, workspace)
        }
        #[doc = concat!("Workspace elements (`", stringify!($t), "`) needed by `trsm_with_workspace` for `m x m` `A` and `m x n` `B`; also what `getrs`/`potrs` need (`m = n`, `n = nrhs`). 0 when `m <= 64`.")]
        pub fn $trsm_len(m: usize, n: usize) -> usize {
            crate::$f::trsm_workspace_len(m, n)
        }
        #[cfg(feature = "alloc")]
        #[doc = concat!("Blocked Cholesky `A = L L^T` (`", stringify!($t), "`) of the symmetric positive definite `n x n` matrix `a`, from and into its lower triangle (the strictly upper triangle is not touched). `Err` carries the first non-positive (or NaN) pivot. Allocates its workspace.\n\n# Panics\nOn inconsistent dimensions or slice lengths.")]
        pub fn $potrf(n: usize, a: &mut [$t], lda: usize) -> Result<(), crate::NotPositiveDefiniteError> {
            let mut ws = alloc::vec![0.0 as $t; crate::$f::potrf_workspace_len(n)];
            crate::$f::potrf_with_workspace(n, a, lda, &mut ws)
        }
        #[doc = concat!("Like the allocating `potrf` (`", stringify!($t), "`) with a caller workspace of at least `potrf_workspace_len` elements.\n\n# Panics\nOn inconsistent dimensions or slice lengths, or a too-small workspace.")]
        pub fn $potrf_ws(n: usize, a: &mut [$t], lda: usize, workspace: &mut [$t]) -> Result<(), crate::NotPositiveDefiniteError> {
            crate::$f::potrf_with_workspace(n, a, lda, workspace)
        }
        #[doc = concat!("Workspace elements (`", stringify!($t), "`) needed by `potrf_with_workspace` for an `n x n` matrix (0 when `n <= 32`).")]
        pub fn $potrf_len(n: usize) -> usize {
            crate::$f::potrf_workspace_len(n)
        }
        #[cfg(feature = "alloc")]
        #[doc = concat!("Solves `A X = B` (`", stringify!($t), "`) in place in `b` (`n x nrhs`) from the lower Cholesky factor produced by `potrf`. Allocates its workspace.\n\n# Panics\nOn inconsistent dimensions or slice lengths.")]
        pub fn $potrs(n: usize, nrhs: usize, a: &[$t], lda: usize, b: &mut [$t], ldb: usize) {
            let mut ws = alloc::vec![0.0 as $t; crate::$f::trsm_workspace_len(n, nrhs)];
            $potrs_ws(n, nrhs, a, lda, b, ldb, &mut ws)
        }
        #[doc = concat!("Like the allocating `potrs` (`", stringify!($t), "`) with a caller workspace of at least `trsm_workspace_len(n, nrhs)` elements.\n\n# Panics\nOn inconsistent dimensions or slice lengths, or a too-small workspace.")]
        pub fn $potrs_ws(n: usize, nrhs: usize, a: &[$t], lda: usize, b: &mut [$t], ldb: usize, workspace: &mut [$t]) {
            crate::$f::trsm_with_workspace(crate::Uplo::Lower, crate::Trans::NoTrans, crate::Diag::NonUnit, n, nrhs, 1.0, a, lda, b, ldb, workspace);
            crate::$f::trsm_with_workspace(crate::Uplo::Lower, crate::Trans::Trans, crate::Diag::NonUnit, n, nrhs, 1.0, a, lda, b, ldb, workspace);
        }
    };
}
