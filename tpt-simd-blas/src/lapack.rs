//! Dense factorisation kernels (LU, Cholesky, triangular solves) built on the
//! level-1 and `gemm` kernels. See the crate docs for the conventions.

use core::fmt;

/// Which triangle of a matrix is referenced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Uplo {
    /// The lower triangle (including the diagonal unless [`Diag::Unit`]).
    Lower,
    /// The upper triangle (including the diagonal unless [`Diag::Unit`]).
    Upper,
}

/// Whether the triangular matrix is used as is or transposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Trans {
    /// Use `A`.
    NoTrans,
    /// Use `A^T`.
    Trans,
}

/// Whether the triangular matrix has an implicit unit diagonal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Diag {
    /// The stored diagonal is used.
    NonUnit,
    /// The diagonal is taken to be all ones and is never read.
    Unit,
}

/// `getrf` found an exactly zero pivot: `U[index, index] == 0`, so the matrix
/// is singular. The factorisation was still completed (LAPACK `info > 0`
/// semantics) but solving with it divides by zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SingularError {
    /// Zero-based index of the first exactly zero pivot.
    pub index: usize,
}

impl fmt::Display for SingularError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "matrix is singular: U[{0}, {0}] is exactly zero",
            self.index
        )
    }
}

impl core::error::Error for SingularError {}

/// `potrf` found a pivot that is not strictly positive (or is NaN): the
/// leading minor of order `index + 1` is not positive definite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotPositiveDefiniteError {
    /// Zero-based index of the first failing pivot.
    pub index: usize,
}

impl fmt::Display for NotPositiveDefiniteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "matrix is not positive definite: pivot {} is not > 0",
            self.index
        )
    }
}

impl core::error::Error for NotPositiveDefiniteError {}

macro_rules! lapack_impl {
    (
        $modname:ident, $t:ident, blas = $blas:ident, sqrt = $sqrt:path,
        nb = $nb:expr, cb = $cb:expr, tb = $tb:expr
    ) => {
        pub(crate) mod $modname {
            //! Per-type factorisation kernels.
            use crate::check::{check_mat, check_ws};
            use crate::$blas::{
                axpy, dot, gemm_with_workspace, gemm_workspace_len, gemv, gemv_t, scal,
            };
            use crate::{Diag, NotPositiveDefiniteError, SingularError, Trans, Uplo};

            /// Panel width of the blocked LU.
            const NB: usize = $nb;
            /// Panel width of the blocked Cholesky.
            const CB: usize = $cb;
            /// Diagonal block size of the blocked triangular solve.
            const TB: usize = $tb;

            /// `col /= p`, via the reciprocal when that is safe.
            fn div_by(col: &mut [$t], p: $t) {
                if p.abs() >= $t::MIN_POSITIVE {
                    let r = 1.0 / p;
                    for v in col.iter_mut() {
                        *v *= r;
                    }
                } else {
                    for v in col.iter_mut() {
                        *v /= p;
                    }
                }
            }

            /// Swaps rows `r1` and `r2` in columns `c0..c1`.
            fn swap_rows(a: &mut [$t], lda: usize, c0: usize, c1: usize, r1: usize, r2: usize) {
                for c in c0..c1 {
                    a.swap(r1 + c * lda, r2 + c * lda);
                }
            }

            // ---------------------------------------------------------- LU

            /// Unblocked LU with partial pivoting of a `rows x cols` panel
            /// (`cols <= rows`). Pivot indices are relative to the panel.
            fn lu_panel(
                rows: usize,
                cols: usize,
                a: &mut [$t],
                lda: usize,
                ipiv: &mut [usize],
                first_zero: &mut Option<usize>,
                base: usize,
            ) {
                for c in 0..cols {
                    let col = &a[c * lda + c..][..rows - c];
                    let mut p = 0;
                    let mut best = col[0].abs();
                    for (i, v) in col.iter().enumerate().skip(1) {
                        let av = v.abs();
                        if av > best {
                            best = av;
                            p = i;
                        }
                    }
                    ipiv[c] = c + p;
                    if best == 0.0 {
                        first_zero.get_or_insert(base + c);
                        continue;
                    }
                    if p != 0 {
                        swap_rows(a, lda, 0, cols, c, c + p);
                    }
                    let piv = a[c + c * lda];
                    div_by(&mut a[c * lda + c + 1..][..rows - c - 1], piv);
                    for c2 in c + 1..cols {
                        let (left, right) = a.split_at_mut(c2 * lda);
                        let l = &left[c * lda + c + 1..][..rows - c - 1];
                        let t = -right[c];
                        axpy(t, l, &mut right[c + 1..][..rows - c - 1]);
                    }
                }
            }

            pub(crate) fn getrf_workspace_len(m: usize, n: usize) -> usize {
                if m.min(n) <= NB {
                    return 0;
                }
                NB * n + gemm_workspace_len(m, n, NB)
            }

            pub(crate) fn getrf_with_workspace(
                m: usize,
                n: usize,
                a: &mut [$t],
                lda: usize,
                ipiv: &mut [usize],
                ws: &mut [$t],
            ) -> Result<(), SingularError> {
                check_mat("getrf", "a", m, n, lda, a.len());
                let mn = m.min(n);
                assert!(
                    ipiv.len() >= mn,
                    "getrf: ipiv has {} elements, needs at least {}",
                    ipiv.len(),
                    mn
                );
                if mn == 0 {
                    return Ok(());
                }
                check_ws("getrf", ws.len(), getrf_workspace_len(m, n));
                let mut first_zero = None;
                let mut j = 0;
                while j < mn {
                    let jb = NB.min(mn - j);
                    lu_panel(
                        m - j,
                        jb,
                        &mut a[j + j * lda..],
                        lda,
                        &mut ipiv[j..j + jb],
                        &mut first_zero,
                        j,
                    );
                    for p in &mut ipiv[j..j + jb] {
                        *p += j;
                    }
                    for i in j..j + jb {
                        let r = ipiv[i];
                        if r != i {
                            swap_rows(a, lda, 0, j, i, r);
                            swap_rows(a, lda, j + jb, n, i, r);
                        }
                    }
                    let nc = n - j - jb;
                    if nc > 0 {
                        let (left, right) = a.split_at_mut((j + jb) * lda);
                        // A12 := L11^-1 A12 (unit lower).
                        tri_solve(
                            Uplo::Lower,
                            Trans::NoTrans,
                            Diag::Unit,
                            jb,
                            nc,
                            &left[j + j * lda..],
                            lda,
                            &mut right[j..],
                            lda,
                        );
                        let mr = m - j - jb;
                        if mr > 0 {
                            // A22 -= A21 * A12 (A12 is copied: it shares a slice with A22).
                            let (bc, gws) = ws.split_at_mut(NB * n);
                            for c in 0..nc {
                                bc[c * jb..][..jb].copy_from_slice(&right[j + c * lda..][..jb]);
                            }
                            gemm_with_workspace(
                                mr,
                                nc,
                                jb,
                                -1.0,
                                &left[j + jb + j * lda..],
                                lda,
                                bc,
                                jb,
                                1.0,
                                &mut right[j + jb..],
                                lda,
                                gws,
                            );
                        }
                    }
                    j += jb;
                }
                match first_zero {
                    None => Ok(()),
                    Some(index) => Err(SingularError { index }),
                }
            }

            // --------------------------------------------------------- trsm

            /// Unblocked triangular solve `op(A) X = B` on every column of `B`.
            #[allow(clippy::too_many_arguments)]
            fn tri_solve(
                uplo: Uplo,
                trans: Trans,
                diag: Diag,
                m: usize,
                n: usize,
                a: &[$t],
                lda: usize,
                b: &mut [$t],
                ldb: usize,
            ) {
                let nonunit = diag == Diag::NonUnit;
                for col in 0..n {
                    let x = &mut b[col * ldb..][..m];
                    match (uplo, trans) {
                        (Uplo::Lower, Trans::NoTrans) => {
                            for j in 0..m {
                                if nonunit {
                                    x[j] /= a[j + j * lda];
                                }
                                let t = -x[j];
                                axpy(t, &a[j * lda + j + 1..][..m - j - 1], &mut x[j + 1..]);
                            }
                        }
                        (Uplo::Upper, Trans::NoTrans) => {
                            for j in (0..m).rev() {
                                if nonunit {
                                    x[j] /= a[j + j * lda];
                                }
                                let t = -x[j];
                                axpy(t, &a[j * lda..][..j], &mut x[..j]);
                            }
                        }
                        (Uplo::Lower, Trans::Trans) => {
                            for i in (0..m).rev() {
                                let s = dot(&a[i * lda + i + 1..][..m - i - 1], &x[i + 1..]);
                                x[i] -= s;
                                if nonunit {
                                    x[i] /= a[i + i * lda];
                                }
                            }
                        }
                        (Uplo::Upper, Trans::Trans) => {
                            for i in 0..m {
                                let s = dot(&a[i * lda..][..i], &x[..i]);
                                x[i] -= s;
                                if nonunit {
                                    x[i] /= a[i + i * lda];
                                }
                            }
                        }
                    }
                }
            }

            pub(crate) fn trsm_workspace_len(m: usize, n: usize) -> usize {
                if m <= TB || n == 0 {
                    return 0;
                }
                gemm_workspace_len(m, n, TB) + TB * m + TB * n
            }

            #[allow(clippy::too_many_arguments)]
            pub(crate) fn trsm_with_workspace(
                uplo: Uplo,
                trans: Trans,
                diag: Diag,
                m: usize,
                n: usize,
                alpha: $t,
                a: &[$t],
                lda: usize,
                b: &mut [$t],
                ldb: usize,
                ws: &mut [$t],
            ) {
                check_mat("trsm", "a", m, m, lda, a.len());
                check_mat("trsm", "b", m, n, ldb, b.len());
                if m == 0 || n == 0 {
                    return;
                }
                if alpha != 1.0 {
                    for j in 0..n {
                        scal(alpha, &mut b[j * ldb..][..m]);
                    }
                    if alpha == 0.0 {
                        return;
                    }
                }
                if m <= TB {
                    tri_solve(uplo, trans, diag, m, n, a, lda, b, ldb);
                    return;
                }
                check_ws("trsm", ws.len(), trsm_workspace_len(m, n));
                let (atc, rest) = ws.split_at_mut(TB * m);
                let (xc, gws) = rest.split_at_mut(TB * n);
                let forward = matches!(
                    (uplo, trans),
                    (Uplo::Lower, Trans::NoTrans) | (Uplo::Upper, Trans::Trans)
                );
                let nblk = m.div_ceil(TB);
                for step in 0..nblk {
                    let bi = if forward { step } else { nblk - 1 - step };
                    let k = bi * TB;
                    let kb = TB.min(m - k);
                    tri_solve(
                        uplo,
                        trans,
                        diag,
                        kb,
                        n,
                        &a[k + k * lda..],
                        lda,
                        &mut b[k..],
                        ldb,
                    );
                    let (r0, rn) = if forward {
                        (k + kb, m - k - kb)
                    } else {
                        (0, k)
                    };
                    if rn == 0 {
                        continue;
                    }
                    for j in 0..n {
                        xc[j * kb..][..kb].copy_from_slice(&b[k + j * ldb..][..kb]);
                    }
                    if n == 1 {
                        let y = &mut b[r0..][..rn];
                        match trans {
                            Trans::NoTrans => {
                                gemv(rn, kb, -1.0, &a[r0 + k * lda..], lda, &xc[..kb], 1.0, y)
                            }
                            Trans::Trans => {
                                gemv_t(kb, rn, -1.0, &a[k + r0 * lda..], lda, &xc[..kb], 1.0, y)
                            }
                        }
                        continue;
                    }
                    match trans {
                        Trans::NoTrans => gemm_with_workspace(
                            rn,
                            n,
                            kb,
                            -1.0,
                            &a[r0 + k * lda..],
                            lda,
                            xc,
                            kb,
                            1.0,
                            &mut b[r0..],
                            ldb,
                            gws,
                        ),
                        Trans::Trans => {
                            for i in 0..rn {
                                let src = &a[k + (r0 + i) * lda..][..kb];
                                for (c, v) in src.iter().enumerate() {
                                    atc[i + c * rn] = *v;
                                }
                            }
                            gemm_with_workspace(
                                rn,
                                n,
                                kb,
                                -1.0,
                                atc,
                                rn,
                                xc,
                                kb,
                                1.0,
                                &mut b[r0..],
                                ldb,
                                gws,
                            )
                        }
                    }
                }
            }

            // ---------------------------------------------------------- getrs

            #[allow(clippy::too_many_arguments)]
            pub(crate) fn getrs_with_workspace(
                n: usize,
                nrhs: usize,
                a: &[$t],
                lda: usize,
                ipiv: &[usize],
                b: &mut [$t],
                ldb: usize,
                ws: &mut [$t],
            ) {
                check_mat("getrs", "a", n, n, lda, a.len());
                check_mat("getrs", "b", n, nrhs, ldb, b.len());
                assert!(
                    ipiv.len() >= n,
                    "getrs: ipiv has {} elements, needs at least {}",
                    ipiv.len(),
                    n
                );
                for (i, &r) in ipiv[..n].iter().enumerate() {
                    assert!(
                        r >= i && r < n,
                        "getrs: invalid pivot index ipiv[{i}] = {r} (n = {n})"
                    );
                }
                if n == 0 || nrhs == 0 {
                    return;
                }
                for col in 0..nrhs {
                    let x = &mut b[col * ldb..][..n];
                    for (i, &r) in ipiv[..n].iter().enumerate() {
                        if r != i {
                            x.swap(i, r);
                        }
                    }
                }
                trsm_with_workspace(
                    Uplo::Lower,
                    Trans::NoTrans,
                    Diag::Unit,
                    n,
                    nrhs,
                    1.0,
                    a,
                    lda,
                    b,
                    ldb,
                    ws,
                );
                trsm_with_workspace(
                    Uplo::Upper,
                    Trans::NoTrans,
                    Diag::NonUnit,
                    n,
                    nrhs,
                    1.0,
                    a,
                    lda,
                    b,
                    ldb,
                    ws,
                );
            }

            // ---------------------------------------------------------- Cholesky

            /// Unblocked right-looking Cholesky of a `rows x cols` panel
            /// (`cols <= rows`), lower triangle only.
            fn chol_panel(
                rows: usize,
                cols: usize,
                a: &mut [$t],
                lda: usize,
                base: usize,
            ) -> Result<(), NotPositiveDefiniteError> {
                for c in 0..cols {
                    let d = a[c + c * lda];
                    if d <= 0.0 || d.is_nan() {
                        return Err(NotPositiveDefiniteError { index: base + c });
                    }
                    let l = $sqrt(d);
                    a[c + c * lda] = l;
                    div_by(&mut a[c * lda + c + 1..][..rows - c - 1], l);
                    for c2 in c + 1..cols {
                        let (left, right) = a.split_at_mut(c2 * lda);
                        let t = -left[c * lda + c2];
                        axpy(
                            t,
                            &left[c * lda + c2..][..rows - c2],
                            &mut right[c2..][..rows - c2],
                        );
                    }
                }
                Ok(())
            }

            pub(crate) fn potrf_workspace_len(n: usize) -> usize {
                if n <= CB {
                    return 0;
                }
                2 * CB * CB + gemm_workspace_len(n, CB, CB)
            }

            pub(crate) fn potrf_with_workspace(
                n: usize,
                a: &mut [$t],
                lda: usize,
                ws: &mut [$t],
            ) -> Result<(), NotPositiveDefiniteError> {
                check_mat("potrf", "a", n, n, lda, a.len());
                if n == 0 {
                    return Ok(());
                }
                check_ws("potrf", ws.len(), potrf_workspace_len(n));
                let mut j = 0;
                while j < n {
                    let jb = CB.min(n - j);
                    chol_panel(n - j, jb, &mut a[j + j * lda..], lda, j)?;
                    let s = j + jb;
                    if s < n {
                        let (bt, rest) = ws.split_at_mut(CB * CB);
                        let (tmp, gws) = rest.split_at_mut(CB * CB);
                        let (left, right) = a.split_at_mut(s * lda);
                        let mut jj = s;
                        while jj < n {
                            let w = CB.min(n - jj);
                            for p in 0..jb {
                                for c in 0..w {
                                    bt[p + c * jb] = left[jj + c + (j + p) * lda];
                                }
                            }
                            // Diagonal tile: only its lower triangle may be written.
                            gemm_with_workspace(
                                w,
                                w,
                                jb,
                                1.0,
                                &left[jj + j * lda..],
                                lda,
                                bt,
                                jb,
                                0.0,
                                tmp,
                                w,
                                gws,
                            );
                            for c in 0..w {
                                let dst = &mut right[jj + c + (jj - s + c) * lda..][..w - c];
                                for (d, t) in dst.iter_mut().zip(&tmp[c * w + c..][..w - c]) {
                                    *d -= *t;
                                }
                            }
                            let below = n - jj - w;
                            if below > 0 {
                                gemm_with_workspace(
                                    below,
                                    w,
                                    jb,
                                    -1.0,
                                    &left[jj + w + j * lda..],
                                    lda,
                                    bt,
                                    jb,
                                    1.0,
                                    &mut right[jj + w + (jj - s) * lda..],
                                    lda,
                                    gws,
                                );
                            }
                            jj += w;
                        }
                    }
                    j += jb;
                }
                Ok(())
            }

            // ---------------------------------------------------------- reference

            /// Naive unblocked implementations (left-to-right sums, no gemm).
            pub(crate) mod reference {
                use crate::check::check_mat;
                use crate::{Diag, NotPositiveDefiniteError, SingularError, Trans, Uplo};

                pub(crate) fn getrf(
                    m: usize,
                    n: usize,
                    a: &mut [$t],
                    lda: usize,
                    ipiv: &mut [usize],
                ) -> Result<(), SingularError> {
                    check_mat("reference getrf", "a", m, n, lda, a.len());
                    let mn = m.min(n);
                    assert!(ipiv.len() >= mn, "reference getrf: ipiv too short");
                    let mut first_zero = None;
                    for k in 0..mn {
                        let mut p = k;
                        let mut best = a[k + k * lda].abs();
                        for i in k + 1..m {
                            let v = a[i + k * lda].abs();
                            if v > best {
                                best = v;
                                p = i;
                            }
                        }
                        ipiv[k] = p;
                        if best == 0.0 {
                            first_zero.get_or_insert(k);
                            continue;
                        }
                        if p != k {
                            for c in 0..n {
                                a.swap(k + c * lda, p + c * lda);
                            }
                        }
                        let piv = a[k + k * lda];
                        for i in k + 1..m {
                            a[i + k * lda] /= piv;
                        }
                        for c in k + 1..n {
                            let u = a[k + c * lda];
                            for i in k + 1..m {
                                a[i + c * lda] -= a[i + k * lda] * u;
                            }
                        }
                    }
                    match first_zero {
                        None => Ok(()),
                        Some(index) => Err(SingularError { index }),
                    }
                }

                #[allow(clippy::too_many_arguments)]
                pub(crate) fn trsm(
                    uplo: Uplo,
                    trans: Trans,
                    diag: Diag,
                    m: usize,
                    n: usize,
                    alpha: $t,
                    a: &[$t],
                    lda: usize,
                    b: &mut [$t],
                    ldb: usize,
                ) {
                    check_mat("reference trsm", "a", m, m, lda, a.len());
                    check_mat("reference trsm", "b", m, n, ldb, b.len());
                    let nonunit = diag == Diag::NonUnit;
                    for col in 0..n {
                        let x = &mut b[col * ldb..][..m];
                        for v in x.iter_mut() {
                            *v *= alpha;
                        }
                        let forward = matches!(
                            (uplo, trans),
                            (Uplo::Lower, Trans::NoTrans) | (Uplo::Upper, Trans::Trans)
                        );
                        for step in 0..m {
                            let i = if forward { step } else { m - 1 - step };
                            let mut s = x[i];
                            let range = if forward { 0..i } else { i + 1..m };
                            for p in range {
                                let aip = match trans {
                                    Trans::NoTrans => a[i + p * lda],
                                    Trans::Trans => a[p + i * lda],
                                };
                                s -= aip * x[p];
                            }
                            if nonunit {
                                s /= a[i + i * lda];
                            }
                            x[i] = s;
                        }
                    }
                }

                #[allow(clippy::too_many_arguments)]
                pub(crate) fn getrs(
                    n: usize,
                    nrhs: usize,
                    a: &[$t],
                    lda: usize,
                    ipiv: &[usize],
                    b: &mut [$t],
                    ldb: usize,
                ) {
                    check_mat("reference getrs", "b", n, nrhs, ldb, b.len());
                    for col in 0..nrhs {
                        for i in 0..n {
                            b.swap(i + col * ldb, ipiv[i] + col * ldb);
                        }
                    }
                    trsm(
                        Uplo::Lower,
                        Trans::NoTrans,
                        Diag::Unit,
                        n,
                        nrhs,
                        1.0,
                        a,
                        lda,
                        b,
                        ldb,
                    );
                    trsm(
                        Uplo::Upper,
                        Trans::NoTrans,
                        Diag::NonUnit,
                        n,
                        nrhs,
                        1.0,
                        a,
                        lda,
                        b,
                        ldb,
                    );
                }

                pub(crate) fn potrf(
                    n: usize,
                    a: &mut [$t],
                    lda: usize,
                ) -> Result<(), NotPositiveDefiniteError> {
                    check_mat("reference potrf", "a", n, n, lda, a.len());
                    for j in 0..n {
                        let mut d = a[j + j * lda];
                        for p in 0..j {
                            d -= a[j + p * lda] * a[j + p * lda];
                        }
                        if d <= 0.0 || d.is_nan() {
                            return Err(NotPositiveDefiniteError { index: j });
                        }
                        let l = $sqrt(d);
                        a[j + j * lda] = l;
                        for i in j + 1..n {
                            let mut s = a[i + j * lda];
                            for p in 0..j {
                                s -= a[i + p * lda] * a[j + p * lda];
                            }
                            a[i + j * lda] = s / l;
                        }
                    }
                    Ok(())
                }

                pub(crate) fn potrs(
                    n: usize,
                    nrhs: usize,
                    a: &[$t],
                    lda: usize,
                    b: &mut [$t],
                    ldb: usize,
                ) {
                    trsm(
                        Uplo::Lower,
                        Trans::NoTrans,
                        Diag::NonUnit,
                        n,
                        nrhs,
                        1.0,
                        a,
                        lda,
                        b,
                        ldb,
                    );
                    trsm(
                        Uplo::Lower,
                        Trans::Trans,
                        Diag::NonUnit,
                        n,
                        nrhs,
                        1.0,
                        a,
                        lda,
                        b,
                        ldb,
                    );
                }
            }
        }
    };
}
