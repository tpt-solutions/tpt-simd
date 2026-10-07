//! Complex kernels against the naive split-plane reference.

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

macro_rules! csuite {
    ($modname:ident, $t:ident, $axpy:ident, $scal:ident, $dotu:ident, $dotc:ident, $nrm2:ident,
     $asum:ident, $gemv:ident, $gemv_t:ident, $gemv_h:ident, $gemm:ident, $gemm_ws:ident,
     $ws_len:ident, $axpy_il:ident, $scal_il:ident, $dotu_il:ident, $dotc_il:ident,
     $nrm2_il:ident, $asum_il:ident, $gemv_il:ident, $gemv_t_il:ident, $gemv_h_il:ident,
     $gemm_il:ident, $deint:ident, $inter:ident) => {
        mod $modname {
            use super::*;

            const EPS: $t = $t::EPSILON;
            const SIZES: [usize; 14] = [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 100];
            const SCALARS: [[$t; 2]; 5] = [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [-0.5, 0.75], [2.0, 0.0]];

            fn data(rng: &mut Rng, n: usize) -> Vec<$t> {
                (0..n).map(|_| rng.next() as $t).collect()
            }

            fn tol(k: usize, scale: $t) -> $t {
                16.0 * (k as $t + 4.0) * EPS * scale
            }

            fn close(got: $t, want: $t, t: $t) -> bool {
                if got.is_nan() || want.is_nan() {
                    return got.is_nan() && want.is_nan();
                }
                if got.is_infinite() || want.is_infinite() {
                    return got == want;
                }
                (got - want).abs() <= t
            }

            fn assert_close(got: &[$t], want: &[$t], t: $t, what: &str) {
                assert_eq!(got.len(), want.len());
                for i in 0..got.len() {
                    assert!(close(got[i], want[i], t), "{what}[{i}]: got {} want {} tol {}", got[i], want[i], t);
                }
            }

            fn mag(z: [$t; 2]) -> $t {
                (z[0] * z[0] + z[1] * z[1]).sqrt()
            }

            #[test]
            fn level1_matches_reference() {
                let mut rng = Rng(7);
                for &n in &SIZES {
                    let (xr, xi, yr0, yi0) = (data(&mut rng, n), data(&mut rng, n), data(&mut rng, n), data(&mut rng, n));
                    for &alpha in &SCALARS {
                        let (mut yr, mut yi) = (yr0.clone(), yi0.clone());
                        let (mut wr, mut wi) = (yr0.clone(), yi0.clone());
                        $axpy(alpha, &xr, &xi, &mut yr, &mut yi);
                        r::$axpy(alpha, &xr, &xi, &mut wr, &mut wi);
                        let t = tol(2, 4.0);
                        assert_close(&yr, &wr, t, "axpy re");
                        assert_close(&yi, &wi, t, "axpy im");

                        let (mut sr, mut si) = (xr.clone(), xi.clone());
                        let (mut tr, mut ti) = (xr.clone(), xi.clone());
                        $scal(alpha, &mut sr, &mut si);
                        r::$scal(alpha, &mut tr, &mut ti);
                        assert_close(&sr, &tr, t, "scal re");
                        assert_close(&si, &ti, t, "scal im");
                    }
                    let bound = 2.0 * n as $t;
                    let t = tol(n, bound);
                    let (got, want) = ($dotu(&xr, &xi, &yr0, &yi0), r::$dotu(&xr, &xi, &yr0, &yi0));
                    assert_close(&got, &want, t, "dotu");
                    let (got, want) = ($dotc(&xr, &xi, &yr0, &yi0), r::$dotc(&xr, &xi, &yr0, &yi0));
                    assert_close(&got, &want, t, "dotc");
                    let (got, want) = ($nrm2(&xr, &xi), r::$nrm2(&xr, &xi));
                    assert!(close(got, want, tol(n, (n as $t).sqrt() + 1.0)), "nrm2 {got} {want}");
                    let (got, want) = ($asum(&xr, &xi), r::$asum(&xr, &xi));
                    assert!(close(got, want, tol(n, 2.0 * n as $t)), "asum {got} {want}");
                }
            }

            #[test]
            fn conjugation_semantics() {
                // conj(x).y with x = i, y = 1: conj(i) * 1 = -i; unconjugated: i.
                assert_eq!($dotc(&[0.0], &[1.0], &[1.0], &[0.0]), [0.0, -1.0]);
                assert_eq!($dotu(&[0.0], &[1.0], &[1.0], &[0.0]), [0.0, 1.0]);
                // dotc(x, x) is real and equals nrm2^2.
                let mut rng = Rng(11);
                let (xr, xi) = (data(&mut rng, 37), data(&mut rng, 37));
                let d = $dotc(&xr, &xi, &xr, &xi);
                let n = $nrm2(&xr, &xi);
                assert!(d[1].abs() <= tol(37, 4.0));
                assert!((d[0] - n * n).abs() <= tol(37, 40.0));
                // dotc(x, y) == conj(dotc(y, x)).
                let (yr, yi) = (data(&mut rng, 37), data(&mut rng, 37));
                let a = $dotc(&xr, &xi, &yr, &yi);
                let b = $dotc(&yr, &yi, &xr, &xi);
                assert!((a[0] - b[0]).abs() <= tol(37, 80.0) && (a[1] + b[1]).abs() <= tol(37, 80.0));
                // asum is |re|+|im|.
                assert_eq!($asum(&[3.0, -1.0], &[-4.0, 2.0]), 10.0);
            }

            #[test]
            fn special_values() {
                for &n in &[1usize, 5, 9, 40] {
                    for pos in [0, n / 2, n - 1] {
                        let mut xr = vec![1.0 as $t; n];
                        let mut xi = vec![0.5 as $t; n];
                        let one = vec![1.0 as $t; n];
                        xr[pos] = $t::NAN;
                        assert!($nrm2(&xr, &xi).is_nan());
                        assert!($asum(&xr, &xi).is_nan());
                        assert!($dotu(&xr, &xi, &one, &one)[0].is_nan());
                        assert!($dotc(&xr, &xi, &one, &one)[0].is_nan());
                        xr[pos] = $t::INFINITY;
                        assert_eq!($nrm2(&xr, &xi), $t::INFINITY);
                        assert_eq!($asum(&xr, &xi), $t::INFINITY);
                        assert_eq!($dotu(&xr, &xi, &one, &one)[0], $t::INFINITY);
                        xr[pos] = 1.0;
                        xi[pos] = $t::NEG_INFINITY;
                        assert_eq!($asum(&xr, &xi), $t::INFINITY);
                        let (mut yr, mut yi) = (vec![0.0 as $t; n], vec![0.0 as $t; n]);
                        $axpy([1.0, 0.0], &xr, &xi, &mut yr, &mut yi);
                        assert_eq!(yi[pos], $t::NEG_INFINITY);
                        // alpha == 0 skips inputs: NaN/inf are not propagated.
                        let (mut yr, mut yi) = (vec![0.0 as $t; n], vec![0.0 as $t; n]);
                        $axpy([0.0, 0.0], &xr, &xi, &mut yr, &mut yi);
                        assert!(yr.iter().chain(&yi).all(|v| *v == 0.0));
                        let (mut sr, mut si) = (xr.clone(), xi.clone());
                        $scal([0.0, 0.0], &mut sr, &mut si);
                        assert!(sr.iter().chain(&si).all(|v| *v == 0.0));
                    }
                }
            }

            #[test]
            fn gemv_variants_match_reference() {
                let mut rng = Rng(21);
                for &(m, n) in &[(0usize, 3usize), (3, 0), (1, 1), (5, 3), (17, 9), (16, 8), (33, 31)] {
                    for pad in [0usize, 3] {
                        let lda = m + pad;
                        let (ar, ai) = (data(&mut rng, lda * n + 1), data(&mut rng, lda * n + 1));
                        for &alpha in &SCALARS {
                            for &beta in &SCALARS {
                                // N
                                let (xr, xi) = (data(&mut rng, n), data(&mut rng, n));
                                let (y0r, y0i) = (data(&mut rng, m), data(&mut rng, m));
                                let (mut yr, mut yi) = (y0r.clone(), y0i.clone());
                                let (mut wr, mut wi) = (y0r.clone(), y0i.clone());
                                $gemv(m, n, alpha, &ar, &ai, lda, &xr, &xi, beta, &mut yr, &mut yi);
                                r::$gemv(m, n, alpha, &ar, &ai, lda, &xr, &xi, beta, &mut wr, &mut wi);
                                let t = tol(n, 2.0 * mag(alpha) * n as $t + 2.0 * mag(beta) + 1.0);
                                assert_close(&yr, &wr, t, "gemv re");
                                assert_close(&yi, &wi, t, "gemv im");
                                // T and H
                                let (xr, xi) = (data(&mut rng, m), data(&mut rng, m));
                                let (y0r, y0i) = (data(&mut rng, n), data(&mut rng, n));
                                let t = tol(m, 2.0 * mag(alpha) * m as $t + 2.0 * mag(beta) + 1.0);
                                let (mut yr, mut yi) = (y0r.clone(), y0i.clone());
                                let (mut wr, mut wi) = (y0r.clone(), y0i.clone());
                                $gemv_t(m, n, alpha, &ar, &ai, lda, &xr, &xi, beta, &mut yr, &mut yi);
                                r::$gemv_t(m, n, alpha, &ar, &ai, lda, &xr, &xi, beta, &mut wr, &mut wi);
                                assert_close(&yr, &wr, t, "gemv_t re");
                                assert_close(&yi, &wi, t, "gemv_t im");
                                let (mut yr, mut yi) = (y0r.clone(), y0i.clone());
                                let (mut wr, mut wi) = (y0r.clone(), y0i.clone());
                                $gemv_h(m, n, alpha, &ar, &ai, lda, &xr, &xi, beta, &mut yr, &mut yi);
                                r::$gemv_h(m, n, alpha, &ar, &ai, lda, &xr, &xi, beta, &mut wr, &mut wi);
                                assert_close(&yr, &wr, t, "gemv_h re");
                                assert_close(&yi, &wi, t, "gemv_h im");
                            }
                        }
                    }
                }
            }

            #[test]
            fn gemv_beta_zero_overwrites_nan_and_alpha_zero_skips_inputs() {
                let (m, n) = (5, 4);
                let (ar, ai) = (vec![1.0 as $t; m * n], vec![1.0 as $t; m * n]);
                let (xr, xi) = (vec![1.0 as $t; n], vec![0.0 as $t; n]);
                let (mut yr, mut yi) = (vec![$t::NAN; m], vec![$t::NAN; m]);
                $gemv(m, n, [1.0, 0.0], &ar, &ai, m, &xr, &xi, [0.0, 0.0], &mut yr, &mut yi);
                assert!(yr.iter().all(|v| *v == n as $t) && yi.iter().all(|v| *v == n as $t));
                let (nr, ni) = (vec![$t::NAN; m * n], vec![$t::NAN; m * n]);
                let (mut yr, mut yi) = (vec![2.0 as $t; m], vec![0.0 as $t; m]);
                $gemv(m, n, [0.0, 0.0], &nr, &ni, m, &xr, &xi, [0.5, 0.0], &mut yr, &mut yi);
                assert!(yr.iter().all(|v| *v == 1.0) && yi.iter().all(|v| *v == 0.0));
            }

            /// One gemm configuration against the reference, plus the interleaved path.
            #[allow(clippy::too_many_arguments)]
            fn check_gemm(rng: &mut Rng, m: usize, n: usize, k: usize, pad: usize, alpha: [$t; 2], beta: [$t; 2]) {
                let (lda, ldb, ldc) = (m + pad, k + pad, m + pad);
                let (ar, ai) = (data(rng, lda * k + 1), data(rng, lda * k + 1));
                let (br, bi) = (data(rng, ldb * n + 1), data(rng, ldb * n + 1));
                let (c0r, c0i) = (data(rng, ldc * n + 1), data(rng, ldc * n + 1));
                let (mut cr, mut ci) = (c0r.clone(), c0i.clone());
                let (mut wr, mut wi) = (c0r.clone(), c0i.clone());
                $gemm(m, n, k, alpha, &ar, &ai, lda, &br, &bi, ldb, beta, &mut cr, &mut ci, ldc);
                r::$gemm(m, n, k, alpha, &ar, &ai, lda, &br, &bi, ldb, beta, &mut wr, &mut wi, ldc);
                let t = tol(k, 2.0 * mag(alpha) * k as $t + 2.0 * mag(beta) + 1.0);
                assert_close(&cr, &wr, t, "gemm re");
                assert_close(&ci, &wi, t, "gemm im");

                // Workspace variant with the documented length.
                let (mut vr, mut vi) = (c0r.clone(), c0i.clone());
                let mut ws = vec![0.0 as $t; $ws_len(m, n, k)];
                $gemm_ws(m, n, k, alpha, &ar, &ai, lda, &br, &bi, ldb, beta, &mut vr, &mut vi, ldc, &mut ws);
                assert_eq!(vr, cr);
                assert_eq!(vi, ci);

                // Interleaved entry point agrees with split (same arithmetic).
                let il = |re: &[$t], im: &[$t]| -> Vec<[$t; 2]> { (0..re.len()).map(|i| [re[i], im[i]]).collect() };
                let (a, b, mut c) = (il(&ar, &ai), il(&br, &bi), il(&c0r, &c0i));
                $gemm_il(m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c, ldc);
                for j in 0..n {
                    for i in 0..m {
                        let z = c[i + j * ldc];
                        assert!(close(z[0], cr[i + j * ldc], t) && close(z[1], ci[i + j * ldc], t), "gemm_il ({i},{j})");
                    }
                }
                // Padding rows of C are untouched.
                for j in 0..n {
                    for i in m..ldc {
                        assert_eq!(cr[i + j * ldc], c0r[i + j * ldc]);
                        assert_eq!(c[i + j * ldc], [c0r[i + j * ldc], c0i[i + j * ldc]]);
                    }
                }
            }

            #[test]
            fn gemm_matches_reference() {
                let mut rng = Rng(33);
                let shapes = [
                    (0usize, 4usize, 4usize), (4, 0, 4), (4, 4, 0), (1, 1, 1), (3, 5, 7), (17, 9, 13),
                    (16, 4, 8), (33, 31, 19), (20, 7, 300), (65, 3, 2),
                ];
                for &(m, n, k) in &shapes {
                    for pad in [0usize, 2] {
                        for &alpha in &SCALARS {
                            for &beta in &SCALARS {
                                check_gemm(&mut rng, m, n, k, pad, alpha, beta);
                            }
                        }
                    }
                }
            }

            #[test]
            fn gemm_special_cases() {
                let (m, n, k) = (7, 5, 3);
                let (ar, ai) = (vec![1.0 as $t; m * k], vec![1.0 as $t; m * k]);
                let (br, bi) = (vec![1.0 as $t; k * n], vec![-1.0 as $t; k * n]);
                // (1+i)(1-i) = 2 per term, so 2k = 6 real, 0 imag.
                let (mut cr, mut ci) = (vec![$t::NAN; m * n], vec![$t::NAN; m * n]);
                $gemm(m, n, k, [1.0, 0.0], &ar, &ai, m, &br, &bi, k, [0.0, 0.0], &mut cr, &mut ci, m);
                assert!(cr.iter().all(|v| *v == 6.0) && ci.iter().all(|v| *v == 0.0));
                // alpha == 0 skips NaN inputs.
                let (nr, ni) = (vec![$t::NAN; m * k], vec![$t::NAN; m * k]);
                let (mut cr, mut ci) = (vec![1.0 as $t; m * n], vec![2.0 as $t; m * n]);
                $gemm(m, n, k, [0.0, 0.0], &nr, &ni, m, &br, &bi, k, [0.0, 1.0], &mut cr, &mut ci, m);
                assert!(cr.iter().all(|v| *v == -2.0) && ci.iter().all(|v| *v == 1.0));
                // NaN/inf in A propagate.
                let (mut ar2, ai2) = (ar.clone(), ai.clone());
                ar2[m + 2] = $t::INFINITY;
                let (mut cr, mut ci) = (vec![0.0 as $t; m * n], vec![0.0 as $t; m * n]);
                $gemm(m, n, k, [1.0, 0.0], &ar2, &ai2, m, &br, &bi, k, [0.0, 0.0], &mut cr, &mut ci, m);
                assert_eq!(cr[2], $t::INFINITY);
                assert!(cr[0].is_finite() && cr[3].is_finite());
            }

            #[test]
            #[should_panic(expected = "gemm")]
            fn gemm_short_slice_panics() {
                let a = vec![0.0 as $t; 3];
                let mut c = vec![0.0 as $t; 4];
                let mut ci = vec![0.0 as $t; 4];
                $gemm(2, 2, 2, [1.0, 0.0], &a, &a, 2, &a, &a, 2, [0.0, 0.0], &mut c, &mut ci, 2);
            }

            #[test]
            fn interleaved_level1_and_gemv_match_split() {
                let mut rng = Rng(55);
                for &n in &SIZES {
                    let (xr, xi, yr0, yi0) = (data(&mut rng, n), data(&mut rng, n), data(&mut rng, n), data(&mut rng, n));
                    let mut x = vec![[0.0 as $t; 2]; n];
                    let mut y0 = vec![[0.0 as $t; 2]; n];
                    $inter(&xr, &xi, &mut x);
                    $inter(&yr0, &yi0, &mut y0);
                    let (mut a, mut b) = (vec![0.0 as $t; n], vec![0.0 as $t; n]);
                    $deint(&x, &mut a, &mut b);
                    assert_eq!((&a, &b), (&xr, &xi));
                    let t = tol(n, 2.0 * n as $t + 4.0);
                    assert_close(&$dotu_il(&x, &y0), &$dotu(&xr, &xi, &yr0, &yi0), t, "dotu_il");
                    assert_close(&$dotc_il(&x, &y0), &$dotc(&xr, &xi, &yr0, &yi0), t, "dotc_il");
                    assert!(close($nrm2_il(&x), $nrm2(&xr, &xi), t));
                    assert!(close($asum_il(&x), $asum(&xr, &xi), t));
                    for &alpha in &SCALARS {
                        let mut y = y0.clone();
                        $axpy_il(alpha, &x, &mut y);
                        let (mut wr, mut wi) = (yr0.clone(), yi0.clone());
                        r::$axpy(alpha, &xr, &xi, &mut wr, &mut wi);
                        for i in 0..n {
                            assert!(close(y[i][0], wr[i], t) && close(y[i][1], wi[i], t));
                        }
                        let mut s = x.clone();
                        $scal_il(alpha, &mut s);
                        let (mut tr, mut ti) = (xr.clone(), xi.clone());
                        r::$scal(alpha, &mut tr, &mut ti);
                        for i in 0..n {
                            assert!(close(s[i][0], tr[i], t) && close(s[i][1], ti[i], t));
                        }
                    }
                }
                for &(m, n) in &[(5usize, 3usize), (17, 9), (0, 2), (2, 0)] {
                    let lda = m + 1;
                    let (ar, ai) = (data(&mut rng, lda * n + 1), data(&mut rng, lda * n + 1));
                    let a: Vec<[$t; 2]> = (0..ar.len()).map(|i| [ar[i], ai[i]]).collect();
                    for &alpha in &SCALARS {
                        for &beta in &SCALARS {
                            for op in 0..3 {
                                let (lx, ly) = if op == 0 { (n, m) } else { (m, n) };
                                let (xr, xi) = (data(&mut rng, lx), data(&mut rng, lx));
                                let (yr0, yi0) = (data(&mut rng, ly), data(&mut rng, ly));
                                let x: Vec<[$t; 2]> = (0..lx).map(|i| [xr[i], xi[i]]).collect();
                                let mut y: Vec<[$t; 2]> = (0..ly).map(|i| [yr0[i], yi0[i]]).collect();
                                let (mut wr, mut wi) = (yr0.clone(), yi0.clone());
                                match op {
                                    0 => {
                                        $gemv_il(m, n, alpha, &a, lda, &x, beta, &mut y);
                                        r::$gemv(m, n, alpha, &ar, &ai, lda, &xr, &xi, beta, &mut wr, &mut wi);
                                    }
                                    1 => {
                                        $gemv_t_il(m, n, alpha, &a, lda, &x, beta, &mut y);
                                        r::$gemv_t(m, n, alpha, &ar, &ai, lda, &xr, &xi, beta, &mut wr, &mut wi);
                                    }
                                    _ => {
                                        $gemv_h_il(m, n, alpha, &a, lda, &x, beta, &mut y);
                                        r::$gemv_h(m, n, alpha, &ar, &ai, lda, &xr, &xi, beta, &mut wr, &mut wi);
                                    }
                                }
                                let t = tol(m + n, 2.0 * mag(alpha) * (m + n) as $t + 2.0 * mag(beta) + 1.0);
                                for i in 0..ly {
                                    assert!(close(y[i][0], wr[i], t) && close(y[i][1], wi[i], t), "gemv il op {op}");
                                }
                            }
                        }
                    }
                }
            }

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(48))]
                #[test]
                fn prop_gemm(m in 0usize..24, n in 0usize..12, k in 0usize..40, pad in 0usize..3,
                             ar in -2.0f64..2.0, ai in -2.0f64..2.0, br in -2.0f64..2.0, bi in -2.0f64..2.0,
                             seed in 0u64..1000) {
                    let mut rng = Rng(seed + 1);
                    check_gemm(&mut rng, m, n, k, pad, [ar as $t, ai as $t], [br as $t, bi as $t]);
                }

                #[test]
                fn prop_dot(n in 0usize..200, seed in 0u64..1000) {
                    let mut rng = Rng(seed + 1);
                    let (xr, xi, yr, yi) = (data(&mut rng, n), data(&mut rng, n), data(&mut rng, n), data(&mut rng, n));
                    let t = tol(n, 2.0 * n as $t + 4.0);
                    assert_close(&$dotu(&xr, &xi, &yr, &yi), &r::$dotu(&xr, &xi, &yr, &yi), t, "dotu");
                    assert_close(&$dotc(&xr, &xi, &yr, &yi), &r::$dotc(&xr, &xi, &yr, &yi), t, "dotc");
                }
            }
        }
    };
}

csuite!(
    c32,
    f32,
    axpy_c32,
    scal_c32,
    dotu_c32,
    dotc_c32,
    nrm2_c32,
    asum_c32,
    gemv_c32,
    gemv_t_c32,
    gemv_h_c32,
    gemm_c32,
    gemm_with_workspace_c32,
    gemm_workspace_len_c32,
    axpy_il_c32,
    scal_il_c32,
    dotu_il_c32,
    dotc_il_c32,
    nrm2_il_c32,
    asum_il_c32,
    gemv_il_c32,
    gemv_t_il_c32,
    gemv_h_il_c32,
    gemm_il_c32,
    deinterleave_c32,
    interleave_c32
);
csuite!(
    c64,
    f64,
    axpy_c64,
    scal_c64,
    dotu_c64,
    dotc_c64,
    nrm2_c64,
    asum_c64,
    gemv_c64,
    gemv_t_c64,
    gemv_h_c64,
    gemm_c64,
    gemm_with_workspace_c64,
    gemm_workspace_len_c64,
    axpy_il_c64,
    scal_il_c64,
    dotu_il_c64,
    dotc_il_c64,
    nrm2_il_c64,
    asum_il_c64,
    gemv_il_c64,
    gemv_t_il_c64,
    gemv_h_il_c64,
    gemm_il_c64,
    deinterleave_c64,
    interleave_c64
);
