//! Complex LU factorisation and solve on split re/im planes (column-major).
//!
//! `getrf_c*` is a blocked right-looking factorisation, like the real `getrf`.
//! Each `NB`-column panel is factored unblocked (largest-modulus pivot, first
//! maximum on ties, rank-1 updates through the complex `axpy` kernel). The block
//! row to its right is then solved with the unit-lower panel, and the trailing
//! matrix gets one complex `gemm` update. `getrs_c*` applies the row swaps and
//! solves with the unit-lower and upper factors, with `axpy`.
//!
//! Pivoting follows `tpt-math-linalg-complex` (modulus, first maximum), so both
//! produce the same permutation for the same input.
//!
//! Layout and conventions match the real LAPACK-style routines: `ipiv[k]` is the
//! zero-based row swapped with row `k`, and `getrf` completes the factorisation
//! even after an exactly zero pivot (reporting the first one).
//!
//! The blocked `getrf` copies panels for the `gemm` update, so this module
//! needs the `alloc` feature.

use crate::SingularError;
use alloc::vec;
use alloc::vec::Vec;

/// Column-panel width of the blocked `getrf`.
const NB: usize = 32;

/// Row-block width of the blocked `getrs`; larger than `NB` so the `gemm` updates
/// are fewer and the `axpy` calls longer.
const TRS_NB: usize = 64;

/// Complex division `(ar + i ai) / (br + i bi)` by Smith's method, which avoids
/// overflow in the intermediate `|b|^2`.
macro_rules! complex_lu_impl {
    (
        $t:ident, $getrf:ident, $getrs:ident, $axpy:path, $gemm:path,
        $sqrt:path, $cdiv:ident, $modulus:ident
    ) => {
        #[inline]
        fn $cdiv(ar: $t, ai: $t, br: $t, bi: $t) -> ($t, $t) {
            if br.abs() >= bi.abs() {
                let r = bi / br;
                let den = br + bi * r;
                ((ar + ai * r) / den, (ai - ar * r) / den)
            } else {
                let r = br / bi;
                let den = br * r + bi;
                ((ar * r + ai) / den, (ai * r - ar) / den)
            }
        }

        #[inline]
        fn $modulus(re: $t, im: $t) -> $t {
            $sqrt(re * re + im * im)
        }

        /// Factor the `n x n` complex matrix `(ar, ai)` in place as `P A = L U`.
        ///
        /// `ipiv[k]` (`>= k`) is the row swapped with row `k`. Returns the first
        /// exactly zero pivot as an error, after finishing the factorisation.
        pub fn $getrf(
            n: usize,
            ar: &mut [$t],
            ai: &mut [$t],
            lda: usize,
            ipiv: &mut [usize],
        ) -> Result<(), SingularError> {
            assert!(lda >= n.max(1), "getrf: lda {lda} < n {n}");
            if n == 0 {
                return Ok(());
            }
            assert!(
                ar.len() >= lda * (n - 1) + n && ai.len() == ar.len(),
                "getrf: bad matrix length"
            );
            assert!(ipiv.len() >= n, "getrf: ipiv needs {n} entries");

            let mut first_zero = None;
            let mut k0 = 0;
            while k0 < n {
                let kend = (k0 + NB).min(n);

                // 1. Unblocked factorisation of the panel (columns k0..kend, rows
                //    k0..n). Row swaps cover every column, so the blocks to the
                //    right see the permuted rows; only panel columns are updated.
                for k in k0..kend {
                    let mut p = k;
                    let mut max = $modulus(ar[k + k * lda], ai[k + k * lda]);
                    for i in (k + 1)..n {
                        let v = $modulus(ar[i + k * lda], ai[i + k * lda]);
                        if v > max {
                            max = v;
                            p = i;
                        }
                    }
                    ipiv[k] = p;
                    if p != k {
                        for j in 0..n {
                            ar.swap(k + j * lda, p + j * lda);
                            ai.swap(k + j * lda, p + j * lda);
                        }
                    }

                    let (dr, di) = (ar[k + k * lda], ai[k + k * lda]);
                    if dr == 0.0 && di == 0.0 {
                        // The column is zero below the diagonal too, so there are
                        // no multipliers and no update.
                        if first_zero.is_none() {
                            first_zero = Some(k);
                        }
                        continue;
                    }
                    for i in (k + 1)..n {
                        let (qr, qi) = $cdiv(ar[i + k * lda], ai[i + k * lda], dr, di);
                        ar[i + k * lda] = qr;
                        ai[i + k * lda] = qi;
                    }

                    // Column k is in the left half of the split; the panel
                    // columns k+1..kend are at the start of the right half.
                    if k + 1 < kend {
                        let (left_r, right_r) = ar.split_at_mut((k + 1) * lda);
                        let (left_i, right_i) = ai.split_at_mut((k + 1) * lda);
                        let lr = &left_r[k * lda + k + 1..k * lda + n];
                        let li = &left_i[k * lda + k + 1..k * lda + n];
                        for j in (k + 1)..kend {
                            let c = (j - k - 1) * lda;
                            let (ur, ui) = (right_r[c + k], right_i[c + k]);
                            $axpy(
                                [-ur, -ui],
                                lr,
                                li,
                                &mut right_r[c + k + 1..c + n],
                                &mut right_i[c + k + 1..c + n],
                            );
                        }
                    }
                }

                if kend < n {
                    let kb = kend - k0;
                    let m2 = n - kend;

                    // 2. Block row U12 = L11^-1 A12 (unit lower, kb x m2): forward
                    //    substitution inside the block, one axpy per row.
                    {
                        let (left_r, right_r) = ar.split_at_mut(kend * lda);
                        let (left_i, right_i) = ai.split_at_mut(kend * lda);
                        for j in 0..m2 {
                            let c = j * lda;
                            for r in k0..kend {
                                let (tr, ti) = (right_r[c + r], right_i[c + r]);
                                if r + 1 < kend {
                                    let lr = &left_r[r * lda + r + 1..r * lda + kend];
                                    let li = &left_i[r * lda + r + 1..r * lda + kend];
                                    $axpy(
                                        [-tr, -ti],
                                        lr,
                                        li,
                                        &mut right_r[c + r + 1..c + kend],
                                        &mut right_i[c + r + 1..c + kend],
                                    );
                                }
                            }
                        }
                    }

                    // 3. Trailing update A22 -= L21 * U12 as one complex gemm. The
                    //    two panels are copied out so the gemm does not alias A22.
                    let mut l21r: Vec<$t> = vec![0.0; m2 * kb];
                    let mut l21i: Vec<$t> = vec![0.0; m2 * kb];
                    for c in 0..kb {
                        for i in 0..m2 {
                            l21r[c * m2 + i] = ar[(k0 + c) * lda + kend + i];
                            l21i[c * m2 + i] = ai[(k0 + c) * lda + kend + i];
                        }
                    }
                    let mut u12r: Vec<$t> = vec![0.0; kb * m2];
                    let mut u12i: Vec<$t> = vec![0.0; kb * m2];
                    for j in 0..m2 {
                        for c in 0..kb {
                            u12r[j * kb + c] = ar[(kend + j) * lda + k0 + c];
                            u12i[j * kb + c] = ai[(kend + j) * lda + k0 + c];
                        }
                    }
                    let (cr, ci) = (&mut ar[kend + kend * lda..], &mut ai[kend + kend * lda..]);
                    $gemm(
                        m2,
                        m2,
                        kb,
                        [-1.0, 0.0],
                        &l21r,
                        &l21i,
                        m2,
                        &u12r,
                        &u12i,
                        kb,
                        [1.0, 0.0],
                        cr,
                        ci,
                        lda,
                    );
                }
                k0 = kend;
            }

            match first_zero {
                Some(index) => Err(SingularError { index }),
                None => Ok(()),
            }
        }

        /// Solve `A X = B` in place from the factors of `getrf`: `B` is
        /// `n x nrhs` (column-major, leading dimension `ldb`) and holds `X` on exit.
        ///
        /// Blocked like `getrf`: each `NB`-row block is solved unblocked, and its
        /// effect on the rest of `B` is one complex `gemm`.
        ///
        /// Like the real `getrs`, a zero `U` diagonal is not checked: it yields
        /// inf/NaN, as in LAPACK.
        // The argument list mirrors the LAPACK `getrs` interface (separate real and imaginary planes).
        #[allow(clippy::too_many_arguments)]
        pub fn $getrs(
            n: usize,
            nrhs: usize,
            ar: &[$t],
            ai: &[$t],
            lda: usize,
            ipiv: &[usize],
            br: &mut [$t],
            bi: &mut [$t],
            ldb: usize,
        ) {
            if n == 0 || nrhs == 0 {
                return;
            }
            assert!(lda >= n && ldb >= n, "getrs: leading dimension too small");
            assert!(ipiv.len() >= n, "getrs: ipiv needs {n} entries");

            // Apply the row interchanges in order, column by column.
            for c in 0..nrhs {
                let xr = &mut br[c * ldb..][..n];
                let xi = &mut bi[c * ldb..][..n];
                for (i, &p) in ipiv[..n].iter().enumerate() {
                    if p != i {
                        xr.swap(i, p);
                        xi.swap(i, p);
                    }
                }
            }

            // Forward substitution with the unit-lower factor, block by block.
            let mut k0 = 0;
            while k0 < n {
                let kend = (k0 + TRS_NB).min(n);
                let kb = kend - k0;
                for c in 0..nrhs {
                    let xr = &mut br[c * ldb + k0..c * ldb + kend];
                    let xi = &mut bi[c * ldb + k0..c * ldb + kend];
                    for j in 0..kb {
                        let (tr, ti) = (xr[j], xi[j]);
                        if j + 1 < kb {
                            let lr = &ar[(k0 + j) * lda + k0 + j + 1..(k0 + j) * lda + kend];
                            let li = &ai[(k0 + j) * lda + k0 + j + 1..(k0 + j) * lda + kend];
                            $axpy([-tr, -ti], lr, li, &mut xr[j + 1..], &mut xi[j + 1..]);
                        }
                    }
                }
                if kend < n {
                    // B[kend.., :] -= L21 * B[k0..kend, :].
                    let m2 = n - kend;
                    let mut b1r: Vec<$t> = vec![0.0; kb * nrhs];
                    let mut b1i: Vec<$t> = vec![0.0; kb * nrhs];
                    for c in 0..nrhs {
                        for j in 0..kb {
                            b1r[c * kb + j] = br[c * ldb + k0 + j];
                            b1i[c * kb + j] = bi[c * ldb + k0 + j];
                        }
                    }
                    let mut l21r: Vec<$t> = vec![0.0; m2 * kb];
                    let mut l21i: Vec<$t> = vec![0.0; m2 * kb];
                    for j in 0..kb {
                        for i in 0..m2 {
                            l21r[j * m2 + i] = ar[(k0 + j) * lda + kend + i];
                            l21i[j * m2 + i] = ai[(k0 + j) * lda + kend + i];
                        }
                    }
                    $gemm(
                        m2,
                        nrhs,
                        kb,
                        [-1.0, 0.0],
                        &l21r,
                        &l21i,
                        m2,
                        &b1r,
                        &b1i,
                        kb,
                        [1.0, 0.0],
                        &mut br[kend..],
                        &mut bi[kend..],
                        ldb,
                    );
                }
                k0 = kend;
            }

            // Back substitution with the upper factor, block by block from the bottom.
            let mut kend = n;
            while kend > 0 {
                let k0 = kend.saturating_sub(TRS_NB);
                let kb = kend - k0;
                for c in 0..nrhs {
                    let xr = &mut br[c * ldb + k0..c * ldb + kend];
                    let xi = &mut bi[c * ldb + k0..c * ldb + kend];
                    for j in (0..kb).rev() {
                        let (qr, qi) = $cdiv(
                            xr[j],
                            xi[j],
                            ar[(k0 + j) * lda + k0 + j],
                            ai[(k0 + j) * lda + k0 + j],
                        );
                        xr[j] = qr;
                        xi[j] = qi;
                        if j > 0 {
                            let ur = &ar[(k0 + j) * lda + k0..(k0 + j) * lda + k0 + j];
                            let ui = &ai[(k0 + j) * lda + k0..(k0 + j) * lda + k0 + j];
                            $axpy([-qr, -qi], ur, ui, &mut xr[..j], &mut xi[..j]);
                        }
                    }
                }
                if k0 > 0 {
                    // B[0..k0, :] -= U12 * B[k0..kend, :].
                    let mut b2r: Vec<$t> = vec![0.0; kb * nrhs];
                    let mut b2i: Vec<$t> = vec![0.0; kb * nrhs];
                    for c in 0..nrhs {
                        for j in 0..kb {
                            b2r[c * kb + j] = br[c * ldb + k0 + j];
                            b2i[c * kb + j] = bi[c * ldb + k0 + j];
                        }
                    }
                    let mut u12r: Vec<$t> = vec![0.0; k0 * kb];
                    let mut u12i: Vec<$t> = vec![0.0; k0 * kb];
                    for j in 0..kb {
                        for i in 0..k0 {
                            u12r[j * k0 + i] = ar[(k0 + j) * lda + i];
                            u12i[j * k0 + i] = ai[(k0 + j) * lda + i];
                        }
                    }
                    $gemm(
                        k0,
                        nrhs,
                        kb,
                        [-1.0, 0.0],
                        &u12r,
                        &u12i,
                        k0,
                        &b2r,
                        &b2i,
                        kb,
                        [1.0, 0.0],
                        &mut br[..],
                        &mut bi[..],
                        ldb,
                    );
                }
                kend = k0;
            }
        }
    };
}

complex_lu_impl!(
    f64,
    getrf_c64,
    getrs_c64,
    crate::axpy_c64,
    crate::gemm_c64,
    libm::sqrt,
    cdiv_f64,
    modulus_f64
);
complex_lu_impl!(
    f32,
    getrf_c32,
    getrs_c32,
    crate::axpy_c32,
    crate::gemm_c32,
    libm::sqrtf,
    cdiv_f32,
    modulus_f32
);

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;
    use std::vec::Vec;

    /// Deterministic pseudo-random values in `[-1, 1)`.
    fn fill(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 52) as f64) - 1.0
    }

    /// `||A x - b||_inf` for a column-major complex `A` (planes) and `x`, `b`.
    fn residual(
        n: usize,
        ar: &[f64],
        ai: &[f64],
        xr: &[f64],
        xi: &[f64],
        br: &[f64],
        bi: &[f64],
    ) -> f64 {
        let mut worst = 0.0_f64;
        for i in 0..n {
            let (mut sr, mut si) = (0.0, 0.0);
            for j in 0..n {
                let (a, b) = (ar[i + j * n], ai[i + j * n]);
                sr += a * xr[j] - b * xi[j];
                si += a * xi[j] + b * xr[j];
            }
            worst = worst.max((sr - br[i]).abs().max((si - bi[i]).abs()));
        }
        worst
    }

    /// Diagonally dominant random complex matrix in planes.
    fn dominant(n: usize, seed: &mut u64) -> (Vec<f64>, Vec<f64>) {
        let mut ar: Vec<f64> = (0..n * n).map(|_| fill(seed)).collect();
        let ai: Vec<f64> = (0..n * n).map(|_| fill(seed)).collect();
        for i in 0..n {
            ar[i + i * n] += n as f64;
        }
        (ar, ai)
    }

    #[test]
    fn solve_residual_is_small_across_sizes() {
        // Sizes straddle the NB panel width, so both the unblocked and the blocked
        // paths run.
        let mut seed = 7u64;
        for n in [1usize, 2, 3, 5, 16, 31, 32, 33, 40, 64, 100] {
            let (ar0, ai0) = dominant(n, &mut seed);
            let (mut ar, mut ai) = (ar0.clone(), ai0.clone());
            let br: Vec<f64> = (0..n).map(|_| fill(&mut seed)).collect();
            let bi: Vec<f64> = (0..n).map(|_| fill(&mut seed)).collect();
            let (mut xr, mut xi) = (br.clone(), bi.clone());

            let mut ipiv = vec![0usize; n];
            getrf_c64(n, &mut ar, &mut ai, n, &mut ipiv).unwrap();
            getrs_c64(n, 1, &ar, &ai, n, &ipiv, &mut xr, &mut xi, n);

            let r = residual(n, &ar0, &ai0, &xr, &xi, &br, &bi);
            assert!(r < 1e-10 * n as f64, "n = {n}: residual {r}");
        }
    }

    #[test]
    fn blocked_and_unblocked_factors_agree() {
        // A plain `P A = L U` check on a size with several panels, against the
        // reconstruction of the factors.
        let n = 70;
        let mut seed = 21u64;
        let (ar0, ai0) = dominant(n, &mut seed);
        let (mut ar, mut ai) = (ar0.clone(), ai0.clone());
        let mut ipiv = vec![0usize; n];
        getrf_c64(n, &mut ar, &mut ai, n, &mut ipiv).unwrap();
        // Rebuild P A from the swaps applied to the original rows and compare with L U.
        let mut pa_r = ar0.clone();
        let mut pa_i = ai0.clone();
        for (k, &p) in ipiv.iter().enumerate() {
            if p != k {
                for j in 0..n {
                    pa_r.swap(k + j * n, p + j * n);
                    pa_i.swap(k + j * n, p + j * n);
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                let (mut sr, mut si) = (0.0, 0.0);
                for k in 0..=i.min(j) {
                    let (lr, li) = if k == i {
                        (1.0, 0.0)
                    } else {
                        (ar[i + k * n], ai[i + k * n])
                    };
                    let (ur, ui) = (ar[k + j * n], ai[k + j * n]);
                    sr += lr * ur - li * ui;
                    si += lr * ui + li * ur;
                }
                let (pr, pi) = (pa_r[i + j * n], pa_i[i + j * n]);
                assert!(
                    (sr - pr).abs() < 1e-9 && (si - pi).abs() < 1e-9,
                    "LU mismatch at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn multiple_rhs_match_single_rhs() {
        let n = 9;
        let mut seed = 11u64;
        let (mut ar, mut ai) = dominant(n, &mut seed);
        let mut ipiv = vec![0usize; n];
        getrf_c64(n, &mut ar, &mut ai, n, &mut ipiv).unwrap();
        let nrhs = 3;
        let br: Vec<f64> = (0..n * nrhs).map(|_| fill(&mut seed)).collect();
        let bi: Vec<f64> = (0..n * nrhs).map(|_| fill(&mut seed)).collect();
        let (mut xr, mut xi) = (br.clone(), bi.clone());
        getrs_c64(n, nrhs, &ar, &ai, n, &ipiv, &mut xr, &mut xi, n);
        for c in 0..nrhs {
            let (mut sr, mut si) = (br[c * n..][..n].to_vec(), bi[c * n..][..n].to_vec());
            getrs_c64(n, 1, &ar, &ai, n, &ipiv, &mut sr, &mut si, n);
            assert_eq!(&xr[c * n..][..n], &sr[..]);
            assert_eq!(&xi[c * n..][..n], &si[..]);
        }
    }

    #[test]
    fn pivoting_handles_a_zero_leading_diagonal() {
        // [[0, 1], [1, 0]] needs a row swap; the solution of A x = (2, 3) is (3, 2).
        let mut ar = vec![0.0, 1.0, 1.0, 0.0];
        let mut ai = vec![0.0; 4];
        let mut ipiv = vec![0usize; 2];
        getrf_c64(2, &mut ar, &mut ai, 2, &mut ipiv).unwrap();
        assert_eq!(ipiv[0], 1);
        let (mut xr, mut xi) = (vec![2.0, 3.0], vec![0.0, 0.0]);
        getrs_c64(2, 1, &ar, &ai, 2, &ipiv, &mut xr, &mut xi, 2);
        assert!((xr[0] - 3.0).abs() < 1e-15 && (xr[1] - 2.0).abs() < 1e-15);
        assert!(xi.iter().all(|v| v.abs() < 1e-15));
    }

    #[test]
    fn complex_pivot_and_division_are_correct() {
        // A = [[i, 1], [2, 1 + i]] and x = (1, -1), so b = A x = (-1 + i, 1 - i).
        let mut ar = vec![0.0, 2.0, 1.0, 1.0];
        let mut ai = vec![1.0, 0.0, 0.0, 1.0];
        let mut ipiv = vec![0usize; 2];
        getrf_c64(2, &mut ar, &mut ai, 2, &mut ipiv).unwrap();
        let (mut xr, mut xi) = (vec![-1.0, 1.0], vec![1.0, -1.0]);
        getrs_c64(2, 1, &ar, &ai, 2, &ipiv, &mut xr, &mut xi, 2);
        assert!(
            (xr[0] - 1.0).abs() < 1e-14 && xi[0].abs() < 1e-14,
            "x0 = {xr:?} {xi:?}"
        );
        assert!(
            (xr[1] + 1.0).abs() < 1e-14 && xi[1].abs() < 1e-14,
            "x1 = {xr:?} {xi:?}"
        );
    }

    #[test]
    fn zero_column_is_reported_with_its_index() {
        let n = 3;
        // Column 1 is entirely zero.
        let mut ar = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0];
        let mut ai = vec![0.0; 9];
        let mut ipiv = vec![0usize; n];
        let err = getrf_c64(n, &mut ar, &mut ai, n, &mut ipiv).unwrap_err();
        assert_eq!(err.index, 1);
    }

    #[test]
    fn single_precision_solve_is_accurate() {
        let n = 40;
        let mut seed = 3u64;
        let (ar64, ai64) = dominant(n, &mut seed);
        let mut ar: Vec<f32> = ar64.iter().map(|&v| v as f32).collect();
        let mut ai: Vec<f32> = ai64.iter().map(|&v| v as f32).collect();
        let orig_r = ar.clone();
        let orig_i = ai.clone();
        let mut ipiv = vec![0usize; n];
        getrf_c32(n, &mut ar, &mut ai, n, &mut ipiv).unwrap();
        let (mut xr, mut xi): (Vec<f32>, Vec<f32>) = (vec![1.0; n], vec![0.5; n]);
        let (br, bi) = (xr.clone(), xi.clone());
        getrs_c32(n, 1, &ar, &ai, n, &ipiv, &mut xr, &mut xi, n);
        for i in 0..n {
            let (mut sr, mut si) = (0.0f32, 0.0f32);
            for j in 0..n {
                let (a, b) = (orig_r[i + j * n], orig_i[i + j * n]);
                sr += a * xr[j] - b * xi[j];
                si += a * xi[j] + b * xr[j];
            }
            assert!((sr - br[i]).abs() < 1e-3 && (si - bi[i]).abs() < 1e-3);
        }
    }
}
