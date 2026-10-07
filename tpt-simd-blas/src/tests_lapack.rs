//! Tests for the factorisation kernels (LU, Cholesky, trsm).

use crate::reference as r;
use crate::*;
use proptest::prelude::*;
use std::vec;
use std::vec::Vec;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    }
}

/// Value written into leading-dimension padding; must survive untouched.
const PAD: f64 = 7.25;

macro_rules! suite {
    (
        $modname:ident, $t:ident, $getrf:ident, $getrf_ws:ident, $getrf_len:ident, $getrs:ident,
        $getrs_ws:ident, $trsm:ident, $trsm_ws:ident, $trsm_len:ident, $potrf:ident,
        $potrf_ws:ident, $potrf_len:ident, $potrs:ident, $potrs_ws:ident
    ) => {
        mod $modname {
            use super::*;

            const EPS: $t = $t::EPSILON;

            fn rand_vec(rng: &mut Rng, len: usize) -> Vec<$t> {
                (0..len).map(|_| rng.next() as $t).collect()
            }

            /// `rows x cols` random matrix with `lda = rows + pad`, padding = PAD.
            fn rand_mat(rng: &mut Rng, rows: usize, cols: usize, pad: usize) -> (Vec<$t>, usize) {
                let lda = rows.max(1) + pad;
                let mut a = vec![PAD as $t; lda * cols.max(1)];
                for j in 0..cols {
                    for i in 0..rows {
                        a[i + j * lda] = rng.next() as $t;
                    }
                }
                (a, lda)
            }

            /// Symmetric diagonally dominant (hence SPD) matrix; the strictly
            /// upper triangle and padding hold sentinels.
            fn spd(rng: &mut Rng, n: usize, pad: usize) -> (Vec<$t>, usize) {
                let lda = n.max(1) + pad;
                let mut a = vec![PAD as $t; lda * n.max(1)];
                for j in 0..n {
                    for i in j..n {
                        let v = rng.next() as $t;
                        a[i + j * lda] = if i == j { v.abs() + n as $t + 1.0 } else { v };
                    }
                }
                (a, lda)
            }

            /// Full symmetric copy of the lower triangle of `a` (`n x n`, ld `lda`) as dense `n x n`.
            fn sym_full(a: &[$t], n: usize, lda: usize) -> Vec<$t> {
                let mut f = vec![0.0 as $t; n * n];
                for j in 0..n {
                    for i in j..n {
                        f[i + j * n] = a[i + j * lda];
                        f[j + i * n] = a[i + j * lda];
                    }
                }
                f
            }

            fn dense(a: &[$t], rows: usize, cols: usize, lda: usize) -> Vec<$t> {
                let mut f = vec![0.0 as $t; rows * cols];
                for j in 0..cols {
                    for i in 0..rows {
                        f[i + j * rows] = a[i + j * lda];
                    }
                }
                f
            }

            fn padding_intact(a: &[$t], rows: usize, cols: usize, lda: usize) {
                for j in 0..cols {
                    for i in rows..lda {
                        assert_eq!(a[i + j * lda], PAD as $t, "padding ({i},{j}) modified");
                    }
                }
            }

            /// `max_i sum_j |a_ij|` for dense `rows x cols`.
            fn norm_inf(a: &[$t], rows: usize, cols: usize) -> $t {
                let mut best = 0.0 as $t;
                for i in 0..rows {
                    let s: $t = (0..cols).map(|j| a[i + j * rows].abs()).sum();
                    best = best.max(s);
                }
                best
            }

            /// Checks `P A = L U` for factors in `lu` (m x n, ld `lda`).
            fn check_lu_reconstruction(
                orig: &[$t],
                lu: &[$t],
                ipiv: &[usize],
                m: usize,
                n: usize,
                lda: usize,
            ) {
                let mn = m.min(n);
                let mut pa = dense(orig, m, n, lda);
                for (i, &p) in ipiv[..mn].iter().enumerate() {
                    assert!(p >= i && p < m);
                    for j in 0..n {
                        pa.swap(i + j * m, p + j * m);
                    }
                }
                for j in 0..n {
                    for i in 0..m {
                        let mut s = 0.0 as $t;
                        let mut bound = 0.0 as $t;
                        for p in 0..=i.min(j) {
                            if p >= mn {
                                break;
                            }
                            let l = if p == i { 1.0 } else { lu[i + p * lda] };
                            let u = lu[p + j * lda];
                            s += l * u;
                            bound += (l * u).abs();
                        }
                        let tol = 8.0 * (mn as $t + 2.0) * EPS * bound + $t::MIN_POSITIVE;
                        let w = pa[i + j * m];
                        assert!(
                            (s - w).abs() <= tol,
                            "LU recon ({i},{j}) m{m} n{n}: {s} vs {w} tol {tol}"
                        );
                    }
                }
            }

            #[test]
            fn getrf_reconstruction_all_sizes() {
                let mut rng = Rng(11);
                let mut sizes: Vec<usize> = (0..=70).collect();
                sizes.push(200);
                for &n in &sizes {
                    let (orig, lda) = rand_mat(&mut rng, n, n, (n % 3) + 1);
                    let mut a = orig.clone();
                    let mut ipiv = vec![0usize; n];
                    let res = $getrf(n, n, &mut a, lda, &mut ipiv);
                    assert!(res.is_ok(), "n{n}: {res:?}");
                    padding_intact(&a, n, n, lda);
                    check_lu_reconstruction(&orig, &a, &ipiv, n, n, lda);
                }
            }

            #[test]
            fn getrf_rectangular() {
                let mut rng = Rng(12);
                for &(m, n) in &[
                    (70usize, 33usize),
                    (33, 70),
                    (100, 1),
                    (1, 100),
                    (65, 64),
                    (64, 65),
                    (0, 5),
                    (5, 0),
                    (130, 97),
                    (97, 130),
                    (3, 40),
                    (40, 3),
                ] {
                    let (orig, lda) = rand_mat(&mut rng, m, n, 2);
                    let mut a = orig.clone();
                    let mut ipiv = vec![0usize; m.min(n)];
                    assert!($getrf(m, n, &mut a, lda, &mut ipiv).is_ok());
                    padding_intact(&a, m, n, lda);
                    check_lu_reconstruction(&orig, &a, &ipiv, m, n, lda);
                    // Matches the naive reference up to rounding.
                    let mut b = orig.clone();
                    let mut ipiv2 = vec![0usize; m.min(n)];
                    assert!(r::$getrf(m, n, &mut b, lda, &mut ipiv2).is_ok());
                    check_lu_reconstruction(&orig, &b, &ipiv2, m, n, lda);
                }
            }

            /// Solves `A X = B` and checks the normwise relative residual.
            fn check_solve_lu(n: usize, nrhs: usize, pad: usize, seed: u64, zero_diag: bool) {
                let mut rng = Rng(seed);
                let (mut orig, lda) = rand_mat(&mut rng, n, n, pad);
                if zero_diag {
                    for i in 0..n {
                        orig[i + i * lda] = 0.0;
                    }
                }
                let (b0, ldb) = rand_mat(&mut rng, n, nrhs, pad + 1);
                let mut a = orig.clone();
                let mut ipiv = vec![0usize; n];
                $getrf(n, n, &mut a, lda, &mut ipiv).unwrap();
                if zero_diag && n > 1 {
                    assert!(ipiv.iter().enumerate().any(|(i, &p)| p != i), "no pivoting");
                }
                let mut x = b0.clone();
                $getrs(n, nrhs, &a, lda, &ipiv, &mut x, ldb);
                padding_intact(&x, n, nrhs, ldb);
                let ad = dense(&orig, n, n, lda);
                let xd = dense(&x, n, nrhs, ldb);
                let bd = dense(&b0, n, nrhs, ldb);
                let an = norm_inf(&ad, n, n);
                for k in 0..nrhs {
                    let mut res = 0.0 as $t;
                    let mut xn = 0.0 as $t;
                    for i in 0..n {
                        let mut s = 0.0 as $t;
                        for j in 0..n {
                            s += ad[i + j * n] * xd[j + k * n];
                        }
                        res = res.max((s - bd[i + k * n]).abs());
                        xn = xn.max(xd[i + k * n].abs());
                    }
                    let bn = (0..n).map(|i| bd[i + k * n].abs()).fold(0.0 as $t, $t::max);
                    let rel = res / (an * xn + bn).max($t::MIN_POSITIVE);
                    assert!(
                        rel <= 8.0 * (n as $t + 4.0) * EPS,
                        "n{n} nrhs{nrhs} k{k}: rel residual {rel}"
                    );
                }
                // And agree with the naive reference solution.
                let mut a2 = orig.clone();
                let mut ip2 = vec![0usize; n];
                r::$getrf(n, n, &mut a2, lda, &mut ip2).unwrap();
                let mut x2 = b0.clone();
                r::$getrs(n, nrhs, &a2, lda, &ip2, &mut x2, ldb);
                let x2d = dense(&x2, n, nrhs, ldb);
                let xmax = xd.iter().fold(1.0 as $t, |m, v| m.max(v.abs()));
                for (g, w) in xd.iter().zip(&x2d) {
                    assert!(
                        (g - w).abs() <= 1e3 * (n as $t + 4.0) * EPS * xmax * (an + 1.0),
                        "n{n}: {g} vs {w}"
                    );
                }
            }

            #[test]
            fn getrs_residuals_many_sizes() {
                let mut seed = 100;
                let mut sizes: Vec<usize> = (0..=70).collect();
                sizes.push(200);
                for &n in &sizes {
                    for &nrhs in &[1usize, 3] {
                        seed += 1;
                        check_solve_lu(n, nrhs, (seed % 3) as usize, seed, false);
                    }
                }
            }

            #[test]
            fn getrs_multiple_rhs_and_blocks() {
                for &n in &[33usize, 64, 65, 100, 130] {
                    for &nrhs in &[2usize, 5, 64, 90] {
                        check_solve_lu(n, nrhs, 2, (n * 1000 + nrhs) as u64, false);
                    }
                }
            }

            #[test]
            fn pivoting_exercised_zero_diagonal() {
                for &n in &[2usize, 3, 5, 31, 32, 33, 64, 70, 100] {
                    check_solve_lu(n, 2, 1, 900 + n as u64, true);
                }
                // [[0, 1], [1, 0]] needs a swap and gives an exact answer.
                let mut a = [0.0 as $t, 1.0, 1.0, 0.0];
                let mut ipiv = [0usize; 2];
                $getrf(2, 2, &mut a, 2, &mut ipiv).unwrap();
                assert_eq!(ipiv, [1, 1]);
                let mut b = [3.0 as $t, 5.0];
                $getrs(2, 1, &a, 2, &ipiv, &mut b, 2);
                assert_eq!(b, [5.0, 3.0]);
            }

            #[test]
            fn getrf_detects_singular() {
                let mut rng = Rng(77);
                for &n in &[1usize, 2, 5, 31, 33, 40, 70, 100] {
                    for k in [0, n / 2, n - 1] {
                        let (mut a, lda) = rand_mat(&mut rng, n, n, 1);
                        for i in 0..n {
                            a[i + k * lda] = 0.0;
                        }
                        let mut ipiv = vec![0usize; n];
                        let err = $getrf(n, n, &mut a, lda, &mut ipiv).unwrap_err();
                        assert_eq!(err, SingularError { index: k }, "n{n}");
                        padding_intact(&a, n, n, lda);
                        // The reference agrees.
                        let (mut b, lda2) = rand_mat(&mut rng, n, n, 0);
                        for i in 0..n {
                            b[i + k * lda2] = 0.0;
                        }
                        let e2 = r::$getrf(n, n, &mut b, lda2, &mut ipiv).unwrap_err();
                        assert_eq!(e2.index, k);
                    }
                }
                let mut z = vec![0.0 as $t; 40 * 40];
                let mut ipiv = vec![0usize; 40];
                assert_eq!($getrf(40, 40, &mut z, 40, &mut ipiv).unwrap_err().index, 0);
                let e = SingularError { index: 3 };
                assert!(std::format!("{e}").contains("singular"));
            }

            #[test]
            fn getrf_nan_does_not_panic() {
                let n = 40;
                let mut rng = Rng(5);
                let (mut a, lda) = rand_mat(&mut rng, n, n, 0);
                a[7 + 3 * lda] = $t::NAN;
                a[0] = $t::INFINITY;
                let mut ipiv = vec![0usize; n];
                let _ = $getrf(n, n, &mut a, lda, &mut ipiv);
                let mut b = vec![1.0 as $t; n];
                $getrs(n, 1, &a, lda, &ipiv, &mut b, n);
            }

            #[test]
            fn getrf_workspace_variant_matches() {
                let n = 100;
                let mut rng = Rng(21);
                let (orig, lda) = rand_mat(&mut rng, n, n, 0);
                let (mut a1, mut a2) = (orig.clone(), orig.clone());
                let (mut p1, mut p2) = (vec![0usize; n], vec![0usize; n]);
                $getrf(n, n, &mut a1, lda, &mut p1).unwrap();
                let mut ws = vec![0.0 as $t; $getrf_len(n, n)];
                assert!(!ws.is_empty());
                $getrf_ws(n, n, &mut a2, lda, &mut p2, &mut ws).unwrap();
                assert_eq!(a1, a2);
                assert_eq!(p1, p2);
                assert_eq!($getrf_len(0, 5), 0);
                assert_eq!($getrf_len(20, 20), 0);
                let mut b1: Vec<$t> = (0..n * 3).map(|i| i as $t).collect();
                let mut b2 = b1.clone();
                $getrs(n, 3, &a1, lda, &p1, &mut b1, n);
                let mut ws = vec![0.0 as $t; $trsm_len(n, 3)];
                $getrs_ws(n, 3, &a1, lda, &p1, &mut b2, n, &mut ws);
                assert_eq!(b1, b2);
            }

            #[test]
            fn zero_dims_are_valid() {
                let mut ipiv: [usize; 0] = [];
                assert!($getrf(0, 0, &mut [], 1, &mut ipiv).is_ok());
                assert!($potrf(0, &mut [], 1).is_ok());
                $getrs(0, 3, &[], 1, &[], &mut [], 1);
                $getrs(3, 0, &[1.0; 9], 3, &[0, 1, 2], &mut [], 3);
                $potrs(0, 3, &[], 1, &mut [], 1);
                $trsm(
                    Uplo::Lower,
                    Trans::NoTrans,
                    Diag::Unit,
                    0,
                    4,
                    1.0,
                    &[],
                    1,
                    &mut [],
                    1,
                );
                $trsm(
                    Uplo::Lower,
                    Trans::NoTrans,
                    Diag::Unit,
                    4,
                    0,
                    1.0,
                    &[1.0; 16],
                    4,
                    &mut [],
                    4,
                );
            }

            // ------------------------------------------------------------ trsm

            fn tri_matrix(rng: &mut Rng, m: usize, uplo: Uplo, pad: usize) -> (Vec<$t>, usize) {
                let lda = m.max(1) + pad;
                // The unreferenced triangle (and for Unit the diagonal) is NaN.
                let mut a = vec![$t::NAN; lda * m.max(1)];
                for j in 0..m {
                    for i in 0..m {
                        let inside = match uplo {
                            Uplo::Lower => i > j,
                            Uplo::Upper => i < j,
                        };
                        if inside {
                            a[i + j * lda] = rng.next() as $t / (m as $t);
                        } else if i == j {
                            a[i + j * lda] = 1.5 + rng.next().abs() as $t;
                        }
                    }
                    for i in m..lda {
                        a[i + j * lda] = PAD as $t;
                    }
                }
                (a, lda)
            }

            #[allow(clippy::too_many_arguments)]
            fn check_trsm(
                m: usize,
                n: usize,
                uplo: Uplo,
                trans: Trans,
                diag: Diag,
                alpha: $t,
                seed: u64,
            ) {
                let mut rng = Rng(seed);
                let (mut a, lda) = tri_matrix(&mut rng, m, uplo, 2);
                if diag == Diag::Unit {
                    for i in 0..m {
                        a[i + i * lda] = $t::NAN;
                    }
                }
                let (b0, ldb) = rand_mat(&mut rng, m, n, 3);
                let mut x = b0.clone();
                $trsm(uplo, trans, diag, m, n, alpha, &a, lda, &mut x, ldb);
                padding_intact(&x, m, n, ldb);
                let mut want = b0.clone();
                r::$trsm(uplo, trans, diag, m, n, alpha, &a, lda, &mut want, ldb);
                // Residual: op(A) X == alpha B using only the stored triangle.
                let get = |i: usize, p: usize| -> $t {
                    // element (i,p) of op(A)
                    let (r_, c_) = if trans == Trans::NoTrans {
                        (i, p)
                    } else {
                        (p, i)
                    };
                    if r_ == c_ {
                        if diag == Diag::Unit {
                            1.0
                        } else {
                            a[r_ + c_ * lda]
                        }
                    } else if (uplo == Uplo::Lower) == (r_ > c_) {
                        a[r_ + c_ * lda]
                    } else {
                        0.0
                    }
                };
                for k in 0..n {
                    let xmax = (0..m).fold(0.0 as $t, |s, i| s.max(x[i + k * ldb].abs()));
                    for i in 0..m {
                        let mut s = 0.0 as $t;
                        for p in 0..m {
                            s += get(i, p) * x[p + k * ldb];
                        }
                        let w = alpha * b0[i + k * ldb];
                        assert!(
                            (s - w).abs() <= 16.0 * (m as $t + 4.0) * EPS * (xmax * 3.0 + w.abs()),
                            "{uplo:?} {trans:?} {diag:?} m{m} n{n}: {s} vs {w}"
                        );
                        let g = x[i + k * ldb];
                        let wv = want[i + k * ldb];
                        assert!(
                            (g - wv).abs() <= 64.0 * (m as $t + 4.0) * EPS * (xmax + 1.0),
                            "vs reference {uplo:?} {trans:?} {diag:?} m{m} n{n}: {g} vs {wv}"
                        );
                    }
                }
            }

            #[test]
            fn trsm_all_variants_and_sizes() {
                let mut seed = 5000;
                for &uplo in &[Uplo::Lower, Uplo::Upper] {
                    for &trans in &[Trans::NoTrans, Trans::Trans] {
                        for &diag in &[Diag::NonUnit, Diag::Unit] {
                            for &m in &[0usize, 1, 2, 31, 63, 64, 65, 100, 129, 130, 200] {
                                for &n in &[1usize, 2, 5, 40] {
                                    seed += 1;
                                    let alpha = if seed % 2 == 0 { 1.0 } else { 0.5 };
                                    check_trsm(m, n, uplo, trans, diag, alpha, seed);
                                }
                            }
                        }
                    }
                }
            }

            #[test]
            fn trsm_alpha_zero_writes_zeros() {
                let (m, n) = (70, 3);
                let a = vec![$t::NAN; m * m];
                let mut b = vec![$t::NAN; m * n];
                $trsm(
                    Uplo::Upper,
                    Trans::Trans,
                    Diag::Unit,
                    m,
                    n,
                    0.0,
                    &a,
                    m,
                    &mut b,
                    m,
                );
                assert!(b.iter().all(|&v| v == 0.0));
            }

            #[test]
            fn trsm_workspace_variant() {
                let (m, n) = (150, 7);
                let mut rng = Rng(3);
                let (a, lda) = tri_matrix(&mut rng, m, Uplo::Lower, 0);
                let (b0, ldb) = rand_mat(&mut rng, m, n, 0);
                let (mut b1, mut b2) = (b0.clone(), b0.clone());
                $trsm(
                    Uplo::Lower,
                    Trans::Trans,
                    Diag::NonUnit,
                    m,
                    n,
                    1.0,
                    &a,
                    lda,
                    &mut b1,
                    ldb,
                );
                let mut ws = vec![0.0 as $t; $trsm_len(m, n)];
                $trsm_ws(
                    Uplo::Lower,
                    Trans::Trans,
                    Diag::NonUnit,
                    m,
                    n,
                    1.0,
                    &a,
                    lda,
                    &mut b2,
                    ldb,
                    &mut ws,
                );
                assert_eq!(b1, b2);
                assert_eq!($trsm_len(64, 10), 0);
                assert_eq!($trsm_len(65, 0), 0);
            }

            // -------------------------------------------------------- Cholesky

            fn check_chol(n: usize, nrhs: usize, pad: usize, seed: u64) {
                let mut rng = Rng(seed);
                let (orig, lda) = spd(&mut rng, n, pad);
                let mut a = orig.clone();
                $potrf(n, &mut a, lda).unwrap();
                // Upper triangle + padding untouched.
                for j in 0..n {
                    for i in 0..j {
                        assert_eq!(a[i + j * lda], orig[i + j * lda], "upper ({i},{j}) touched");
                    }
                }
                padding_intact(&a, n, n, lda);
                // L L^T == A.
                let full = sym_full(&orig, n, lda);
                for j in 0..n {
                    for i in j..n {
                        let mut s = 0.0 as $t;
                        let mut bound = 0.0 as $t;
                        for p in 0..=j {
                            s += a[i + p * lda] * a[j + p * lda];
                            bound += (a[i + p * lda] * a[j + p * lda]).abs();
                        }
                        assert!(
                            (s - full[i + j * n]).abs() <= 8.0 * (n as $t + 2.0) * EPS * bound,
                            "n{n} L L^T ({i},{j}): {s} vs {}",
                            full[i + j * n]
                        );
                    }
                }
                // Reference factor agrees.
                let mut a2 = orig.clone();
                r::$potrf(n, &mut a2, lda).unwrap();
                for j in 0..n {
                    for i in j..n {
                        assert!(
                            (a[i + j * lda] - a2[i + j * lda]).abs()
                                <= 64.0 * (n as $t + 2.0) * EPS * (a2[j + j * lda].abs() + 1.0),
                            "n{n} vs reference ({i},{j})"
                        );
                    }
                }
                if nrhs > 0 {
                    let (b0, ldb) = rand_mat(&mut rng, n, nrhs, 1);
                    let mut x = b0.clone();
                    $potrs(n, nrhs, &a, lda, &mut x, ldb);
                    padding_intact(&x, n, nrhs, ldb);
                    let an = norm_inf(&full, n, n);
                    for k in 0..nrhs {
                        let mut res = 0.0 as $t;
                        let mut xn = 0.0 as $t;
                        for i in 0..n {
                            let mut s = 0.0 as $t;
                            for j in 0..n {
                                s += full[i + j * n] * x[j + k * ldb];
                            }
                            res = res.max((s - b0[i + k * ldb]).abs());
                            xn = xn.max(x[i + k * ldb].abs());
                        }
                        assert!(
                            res / (an * xn).max($t::MIN_POSITIVE) <= 8.0 * (n as $t + 4.0) * EPS,
                            "n{n} k{k} residual"
                        );
                    }
                }
            }

            #[test]
            fn potrf_potrs_all_sizes() {
                let mut sizes: Vec<usize> = (0..=70).collect();
                sizes.push(200);
                for &n in &sizes {
                    check_chol(n, 1 + (n % 3), n % 3, 300 + n as u64);
                }
            }

            #[test]
            fn potrs_multiple_rhs_blocks() {
                for &n in &[33usize, 64, 65, 130] {
                    for &nrhs in &[2usize, 7, 70] {
                        check_chol(n, nrhs, 1, (n * 100 + nrhs) as u64);
                    }
                }
            }

            #[test]
            fn potrf_detects_non_spd() {
                let mut rng = Rng(8);
                for &n in &[1usize, 2, 5, 31, 33, 40, 70, 100] {
                    for k in [0, n / 2, n - 1] {
                        let (mut a, lda) = spd(&mut rng, n, 1);
                        a[k + k * lda] = -1.0;
                        let err = $potrf(n, &mut a, lda).unwrap_err();
                        assert_eq!(err, NotPositiveDefiniteError { index: k }, "n{n}");
                        let (mut b, lda2) = spd(&mut rng, n, 0);
                        b[k + k * lda2] = -1.0;
                        assert_eq!(r::$potrf(n, &mut b, lda2).unwrap_err().index, k);
                    }
                }
                // Indefinite [[1, 2], [2, 1]] fails at the second pivot.
                let mut a = [1.0 as $t, 2.0, 99.0, 1.0];
                assert_eq!($potrf(2, &mut a, 2).unwrap_err().index, 1);
                // Zero and NaN pivots.
                let mut z = vec![0.0 as $t; 36 * 36];
                assert_eq!($potrf(36, &mut z, 36).unwrap_err().index, 0);
                let (mut a, lda) = spd(&mut rng, 50, 0);
                a[40 + 40 * lda] = $t::NAN;
                assert_eq!($potrf(50, &mut a, lda).unwrap_err().index, 40);
                let e = NotPositiveDefiniteError { index: 2 };
                assert!(std::format!("{e}").contains("positive definite"));
            }

            #[test]
            fn potrf_workspace_variant_matches() {
                let n = 100;
                let mut rng = Rng(31);
                let (orig, lda) = spd(&mut rng, n, 0);
                let (mut a1, mut a2) = (orig.clone(), orig.clone());
                $potrf(n, &mut a1, lda).unwrap();
                let mut ws = vec![0.0 as $t; $potrf_len(n)];
                assert!(!ws.is_empty());
                $potrf_ws(n, &mut a2, lda, &mut ws).unwrap();
                assert_eq!(a1, a2);
                assert_eq!($potrf_len(32), 0);
                let mut b1: Vec<$t> = (0..n * 2).map(|i| i as $t).collect();
                let mut b2 = b1.clone();
                $potrs(n, 2, &a1, lda, &mut b1, n);
                let mut ws = vec![0.0 as $t; $trsm_len(n, 2)];
                $potrs_ws(n, 2, &a1, lda, &mut b2, n, &mut ws);
                assert_eq!(b1, b2);
            }

            // ----------------------------------------------------------- panics

            #[test]
            #[should_panic(expected = "ipiv")]
            fn getrf_short_ipiv_panics() {
                let mut a = [1.0 as $t; 16];
                $getrf(4, 4, &mut a, 4, &mut [0usize; 3]).unwrap();
            }

            #[test]
            #[should_panic(expected = "leading dimension")]
            fn getrf_bad_lda_panics() {
                let mut a = [1.0 as $t; 16];
                let _ = $getrf(4, 4, &mut a, 3, &mut [0usize; 4]);
            }

            #[test]
            #[should_panic(expected = "elements")]
            fn potrf_short_slice_panics() {
                let mut a = [1.0 as $t; 15];
                let _ = $potrf(4, &mut a, 4);
            }

            #[test]
            #[should_panic(expected = "workspace")]
            fn getrf_small_workspace_panics() {
                let mut a = vec![1.0 as $t; 100 * 100];
                let _ = $getrf_ws(100, 100, &mut a, 100, &mut [0usize; 100], &mut []);
            }

            #[test]
            #[should_panic(expected = "workspace")]
            fn potrf_small_workspace_panics() {
                let mut a = vec![1.0 as $t; 100 * 100];
                let _ = $potrf_ws(100, &mut a, 100, &mut []);
            }

            #[test]
            #[should_panic(expected = "workspace")]
            fn trsm_small_workspace_panics() {
                let a = vec![1.0 as $t; 100 * 100];
                let mut b = vec![1.0 as $t; 100 * 2];
                $trsm_ws(
                    Uplo::Lower,
                    Trans::NoTrans,
                    Diag::NonUnit,
                    100,
                    2,
                    1.0,
                    &a,
                    100,
                    &mut b,
                    100,
                    &mut [],
                );
            }

            #[test]
            #[should_panic(expected = "invalid pivot")]
            fn getrs_bad_pivot_panics() {
                let a = [1.0 as $t; 4];
                let mut b = [1.0 as $t; 2];
                $getrs(2, 1, &a, 2, &[0, 5], &mut b, 2);
            }

            #[test]
            #[should_panic(expected = "elements")]
            fn trsm_short_b_panics() {
                let a = [1.0 as $t; 9];
                let mut b = [1.0 as $t; 5];
                $trsm(
                    Uplo::Lower,
                    Trans::NoTrans,
                    Diag::NonUnit,
                    3,
                    2,
                    1.0,
                    &a,
                    3,
                    &mut b,
                    3,
                );
            }

            #[test]
            fn solve_matches_known_system() {
                // 3x3 with a known solution x = [1, 2, 3].
                let a = [2.0 as $t, 4.0, -2.0, 1.0, -6.0, 7.0, 1.0, 0.0, 2.0];
                let mut lu = a;
                let mut ipiv = [0usize; 3];
                $getrf(3, 3, &mut lu, 3, &mut ipiv).unwrap();
                // A x for x = [1,2,3]: [2+2+3, 4-12+0, -2+14+6] = [7, -8, 18]
                let mut b = [7.0 as $t, -8.0, 18.0];
                $getrs(3, 1, &lu, 3, &ipiv, &mut b, 3);
                for (g, w) in b.iter().zip([1.0 as $t, 2.0, 3.0]) {
                    assert!((g - w).abs() < 1e-4);
                }
            }
        }
    };
}

suite!(
    s,
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
suite!(
    d,
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn prop_lu_solve_f64(n in 0usize..80, nrhs in 1usize..6, pad in 0usize..3, seed in any::<u64>()) {
        let mut rng = Rng(seed);
        let lda = n.max(1) + pad;
        let a0: Vec<f64> = (0..lda * n.max(1)).map(|_| rng.next()).collect();
        let b0: Vec<f64> = (0..n.max(1) * nrhs).map(|_| rng.next()).collect();
        let mut a = a0.clone();
        let mut ipiv = vec![0usize; n];
        if getrf_f64(n, n, &mut a, lda, &mut ipiv).is_ok() && n > 0 {
            let mut x = b0.clone();
            getrs_f64(n, nrhs, &a, lda, &ipiv, &mut x, n.max(1));
            // Backward-error style bound: |A x - b|_i <= c n eps (|A||x| + |b|)_i.
            for k in 0..nrhs {
                for i in 0..n {
                    let mut s = 0.0;
                    let mut bound = b0[i + k * n].abs();
                    for j in 0..n {
                        s += a0[i + j * lda] * x[j + k * n];
                        bound += (a0[i + j * lda] * x[j + k * n]).abs();
                    }
                    prop_assert!((s - b0[i + k * n]).abs() <= 1e-11 * (n as f64 + 1.0) * bound);
                }
            }
        }
    }

    #[test]
    fn prop_chol_f32(n in 0usize..80, seed in any::<u64>()) {
        let mut rng = Rng(seed);
        let lda = n.max(1);
        let mut a = vec![0.0f32; lda * n.max(1)];
        for j in 0..n {
            for i in j..n {
                let v = rng.next() as f32;
                a[i + j * lda] = if i == j { v.abs() + n as f32 + 1.0 } else { v };
            }
        }
        let orig = a.clone();
        prop_assert!(potrf_f32(n, &mut a, lda).is_ok());
        for j in 0..n {
            for i in j..n {
                let mut s = 0.0f32;
                let mut bound = 0.0f32;
                for p in 0..=j {
                    s += a[i + p * lda] * a[j + p * lda];
                    bound += (a[i + p * lda] * a[j + p * lda]).abs();
                }
                prop_assert!((s - orig[i + j * lda]).abs() <= 8.0 * (n as f32 + 2.0) * f32::EPSILON * bound);
            }
        }
    }
}
