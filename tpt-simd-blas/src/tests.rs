//! Tests against the naive scalar reference.

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

macro_rules! suite {
    ($modname:ident, $t:ident, $eps:expr, $axpy:ident, $scal:ident, $dot:ident, $nrm2:ident,
     $asum:ident, $gemv:ident, $gemv_t:ident, $gemm:ident, $gemm_ws:ident, $ws_len:ident) => {
        mod $modname {
            use super::*;

            const EPS: $t = $eps;

            fn data(rng: &mut Rng, n: usize) -> Vec<$t> {
                (0..n).map(|_| rng.next() as $t).collect()
            }

            fn close(got: $t, want: $t, bound: $t, n: usize) -> bool {
                if got.is_nan() || want.is_nan() {
                    return got.is_nan() && want.is_nan();
                }
                if got.is_infinite() || want.is_infinite() {
                    return got == want;
                }
                (got - want).abs() <= 8.0 * (n as $t + 2.0) * EPS * bound + $t::MIN_POSITIVE
            }

            /// Checks gemm against the reference for one configuration.
            #[allow(clippy::too_many_arguments)]
            fn check_gemm(
                m: usize,
                n: usize,
                k: usize,
                pa: usize,
                pb: usize,
                pc: usize,
                alpha: $t,
                beta: $t,
                seed: u64,
            ) {
                let (lda, ldb, ldc) = (m.max(1) + pa, k.max(1) + pb, m.max(1) + pc);
                let mut rng = Rng(seed);
                let a = data(&mut rng, lda * k.max(1));
                let b = data(&mut rng, ldb * n.max(1));
                let mut c0 = data(&mut rng, ldc * n.max(1));
                // NaN in C must be overwritten when beta == 0.
                if beta == 0.0 {
                    for v in c0.iter_mut().step_by(3) {
                        *v = $t::NAN;
                    }
                }
                let mut want = c0.clone();
                let mut got = c0.clone();
                r::$gemm(m, n, k, alpha, &a, lda, &b, ldb, beta, &mut want, ldc);
                $gemm(m, n, k, alpha, &a, lda, &b, ldb, beta, &mut got, ldc);
                for j in 0..n {
                    for i in 0..m {
                        let mut bound = if beta == 0.0 {
                            0.0
                        } else {
                            beta.abs() * c0[i + j * ldc].abs()
                        };
                        let mut s = 0.0;
                        for p in 0..k {
                            s += (a[i + p * lda] * b[p + j * ldb]).abs();
                        }
                        bound += alpha.abs() * s;
                        let (g, w) = (got[i + j * ldc], want[i + j * ldc]);
                        assert!(
                            close(g, w, bound, k),
                            "m{m} n{n} k{k} ({i},{j}): got {g} want {w} bound {bound}"
                        );
                    }
                }
                // Padding rows (ldc > m) must be untouched.
                for j in 0..n {
                    for i in m..ldc {
                        assert_eq!(got[i + j * ldc].to_bits(), c0[i + j * ldc].to_bits());
                    }
                }
            }

            #[test]
            fn gemm_sizes_and_tails() {
                let dims = [
                    0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 67, 130,
                ];
                let mut seed = 1;
                for &m in &dims {
                    for &n in &[0usize, 1, 3, 4, 5, 9, 70] {
                        for &k in &[0usize, 1, 2, 7, 33, 300] {
                            seed += 1;
                            check_gemm(m, n, k, 0, 0, 0, 1.0, 0.0, seed);
                        }
                    }
                }
            }

            #[test]
            fn gemm_block_boundaries() {
                // Larger than MC/KC/NC so every blocking loop iterates more than once.
                check_gemm(150, 40, 520, 0, 0, 0, 1.0, 0.0, 99);
                check_gemm(37, 530, 20, 0, 0, 0, 1.0, 0.0, 98);
            }

            #[test]
            fn gemm_alpha_beta_and_ld() {
                for &(alpha, beta) in &[
                    (1.0, 1.0),
                    (2.5, 0.5),
                    (-1.0, 2.0),
                    (0.0, 3.0),
                    (0.0, 0.0),
                    (1.5, 0.0),
                ] {
                    check_gemm(19, 11, 23, 5, 3, 7, alpha, beta, 7);
                }
            }

            #[test]
            fn gemm_beta_zero_overwrites_nan() {
                let (m, n, k) = (5, 6, 4);
                let a = vec![1.0 as $t; m * k];
                let b = vec![1.0 as $t; k * n];
                let mut c = vec![$t::NAN; m * n];
                $gemm(m, n, k, 1.0, &a, m, &b, k, 0.0, &mut c, m);
                assert!(c.iter().all(|&v| v == 4.0));
                // k == 0 and alpha == 0 also overwrite.
                let mut c = vec![$t::NAN; m * n];
                $gemm(m, n, 0, 1.0, &[], m, &[], 1, 0.0, &mut c, m);
                assert!(c.iter().all(|&v| v == 0.0));
                let mut c = vec![$t::INFINITY; m * n];
                $gemm(m, n, k, 0.0, &a, m, &b, k, 0.0, &mut c, m);
                assert!(c.iter().all(|&v| v == 0.0));
            }

            #[test]
            fn gemm_nan_inf_propagation() {
                let (m, n, k) = (20, 7, 9);
                let mut rng = Rng(5);
                let mut a: Vec<$t> = data(&mut rng, m * k)
                    .iter()
                    .map(|v| v.abs() + 1.0)
                    .collect();
                let b: Vec<$t> = data(&mut rng, k * n)
                    .iter()
                    .map(|v| v.abs() + 1.0)
                    .collect();
                a[3 + 2 * m] = $t::NAN;
                a[17 + 4 * m] = $t::INFINITY;
                let mut got = vec![0.0 as $t; m * n];
                let mut want = got.clone();
                $gemm(m, n, k, 1.0, &a, m, &b, k, 0.0, &mut got, m);
                r::$gemm(m, n, k, 1.0, &a, m, &b, k, 0.0, &mut want, m);
                for j in 0..n {
                    assert!(got[3 + j * m].is_nan());
                    assert_eq!(got[17 + j * m], $t::INFINITY);
                    for i in [0, 1, 10, 19] {
                        assert!(got[i + j * m].is_finite());
                    }
                }
                assert_eq!(
                    got.iter().map(|v| v.is_nan()).collect::<Vec<_>>(),
                    want.iter().map(|v| v.is_nan()).collect::<Vec<_>>()
                );
            }

            #[test]
            fn gemm_workspace_variant() {
                let (m, n, k) = (33, 21, 40);
                let mut rng = Rng(11);
                let a = data(&mut rng, m * k);
                let b = data(&mut rng, k * n);
                let mut c1 = vec![0.0 as $t; m * n];
                let mut c2 = c1.clone();
                let mut ws = vec![0.0 as $t; $ws_len(m, n, k)];
                $gemm(m, n, k, 1.0, &a, m, &b, k, 0.0, &mut c1, m);
                $gemm_ws(m, n, k, 1.0, &a, m, &b, k, 0.0, &mut c2, m, &mut ws);
                assert_eq!(c1, c2);
                assert_eq!($ws_len(0, 5, 5), 0);
            }

            #[test]
            #[should_panic(expected = "workspace")]
            fn gemm_small_workspace_panics() {
                let a = [1.0 as $t; 16];
                let mut c = [0.0 as $t; 16];
                $gemm_ws(4, 4, 4, 1.0, &a, 4, &a, 4, 0.0, &mut c, 4, &mut []);
            }

            #[test]
            #[should_panic(expected = "leading dimension")]
            fn gemm_bad_lda_panics() {
                let a = [1.0 as $t; 16];
                let mut c = [0.0 as $t; 16];
                $gemm(4, 4, 4, 1.0, &a, 3, &a, 4, 0.0, &mut c, 4);
            }

            #[test]
            #[should_panic(expected = "elements")]
            fn gemm_short_slice_panics() {
                let a = [1.0 as $t; 8];
                let mut c = [0.0 as $t; 16];
                $gemm(4, 4, 4, 1.0, &a, 4, &a, 4, 0.0, &mut c, 4);
            }

            #[test]
            fn level1_all_lengths() {
                let mut rng = Rng(3);
                for n in (0..140).chain([255, 256, 257, 1000, 1023]) {
                    let x = data(&mut rng, n);
                    let y = data(&mut rng, n);
                    let bound: $t = x.iter().zip(&y).map(|(a, b)| (a * b).abs()).sum();
                    assert!(close($dot(&x, &y), r::$dot(&x, &y), bound, n), "dot n={n}");
                    let sq: $t = x.iter().map(|a| a * a).sum();
                    assert!(close($nrm2(&x), r::$nrm2(&x), sq.sqrt(), n), "nrm2 n={n}");
                    let ab: $t = x.iter().map(|a| a.abs()).sum();
                    assert!(close($asum(&x), r::$asum(&x), ab, n), "asum n={n}");
                    let mut y1 = y.clone();
                    let mut y2 = y.clone();
                    $axpy(0.75, &x, &mut y1);
                    r::$axpy(0.75, &x, &mut y2);
                    assert_eq!(y1, y2, "axpy n={n}");
                    let mut s1 = x.clone();
                    $scal(-1.5, &mut s1);
                    assert!(s1.iter().zip(&x).all(|(s, v)| *s == -1.5 * v));
                }
            }

            #[test]
            fn level1_special_values() {
                assert_eq!($dot(&[], &[]), 0.0);
                assert_eq!($nrm2(&[]), 0.0);
                assert_eq!($asum(&[]), 0.0);
                let mut x = vec![1.0 as $t; 50];
                x[41] = $t::NAN;
                assert!($dot(&x, &x).is_nan());
                assert!($nrm2(&x).is_nan());
                assert!($asum(&x).is_nan());
                x[41] = $t::INFINITY;
                assert_eq!($dot(&x, &x), $t::INFINITY);
                assert_eq!($asum(&x), $t::INFINITY);
                assert_eq!($nrm2(&x), $t::INFINITY);
                // alpha == 0: axpy is a no-op, scal writes zeros (even over NaN).
                let mut y = vec![2.0 as $t; 50];
                $axpy(0.0, &x, &mut y);
                assert!(y.iter().all(|&v| v == 2.0));
                let mut z = vec![$t::NAN; 10];
                $scal(0.0, &mut z);
                assert!(z.iter().all(|&v| v == 0.0));
                assert_eq!($nrm2(&[3.0, 4.0]), 5.0);
            }

            #[test]
            #[should_panic(expected = "length mismatch")]
            fn dot_mismatch_panics() {
                $dot(&[1.0], &[1.0, 2.0]);
            }

            #[allow(clippy::too_many_arguments)]
            fn check_gemv(m: usize, n: usize, pad: usize, alpha: $t, beta: $t, seed: u64) {
                let lda = m.max(1) + pad;
                let mut rng = Rng(seed);
                let a = data(&mut rng, lda * n.max(1));
                for trans in [false, true] {
                    let (xl, yl) = if trans { (m, n) } else { (n, m) };
                    let x = data(&mut rng, xl);
                    let mut y0 = data(&mut rng, yl);
                    if beta == 0.0 {
                        y0.iter_mut().for_each(|v| *v = $t::NAN);
                    }
                    let (mut got, mut want) = (y0.clone(), y0.clone());
                    if trans {
                        $gemv_t(m, n, alpha, &a, lda, &x, beta, &mut got);
                        r::$gemv_t(m, n, alpha, &a, lda, &x, beta, &mut want);
                    } else {
                        $gemv(m, n, alpha, &a, lda, &x, beta, &mut got);
                        r::$gemv(m, n, alpha, &a, lda, &x, beta, &mut want);
                    }
                    for o in 0..yl {
                        let mut bound = if beta == 0.0 {
                            0.0
                        } else {
                            beta.abs() * y0[o].abs()
                        };
                        let inner = if trans { m } else { n };
                        for p in 0..inner {
                            let (i, j) = if trans { (p, o) } else { (o, p) };
                            bound += alpha.abs() * (a[i + j * lda] * x[p]).abs();
                        }
                        assert!(
                            close(got[o], want[o], bound, inner),
                            "gemv trans={trans} m{m} n{n}: {} vs {}",
                            got[o],
                            want[o]
                        );
                    }
                }
            }

            #[test]
            fn gemv_sizes() {
                let mut seed = 40;
                for m in [0usize, 1, 3, 8, 17, 64, 100] {
                    for n in [0usize, 1, 2, 4, 5, 9, 33] {
                        for &(alpha, beta) in &[(1.0, 0.0), (2.0, 0.5), (0.0, 2.0), (-1.0, 1.0)] {
                            seed += 1;
                            check_gemv(m, n, (seed % 4) as usize, alpha, beta, seed);
                        }
                    }
                }
            }

            #[test]
            fn gemv_nan_in_x_propagates() {
                let (m, n) = (9, 6);
                let a = vec![1.0 as $t; m * n];
                let mut x = vec![1.0 as $t; n];
                x[2] = $t::NAN;
                let mut y = vec![0.0 as $t; m];
                $gemv(m, n, 1.0, &a, m, &x, 0.0, &mut y);
                assert!(y.iter().all(|v| v.is_nan()));
            }
        }
    };
}

suite!(
    s,
    f32,
    f32::EPSILON,
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
suite!(
    d,
    f64,
    f64::EPSILON,
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn prop_gemm_f32(
        m in 0usize..40, n in 0usize..40, k in 0usize..60,
        pa in 0usize..4, pb in 0usize..4, pc in 0usize..4,
        alpha in -2.0f32..2.0, beta in prop::sample::select(vec![0.0f32, 1.0, -0.5, 2.0]),
        seed in any::<u64>(),
    ) {
        let (lda, ldb, ldc) = (m.max(1) + pa, k.max(1) + pb, m.max(1) + pc);
        let mut rng = Rng(seed);
        let a: Vec<f32> = (0..lda * k.max(1)).map(|_| rng.next() as f32).collect();
        let b: Vec<f32> = (0..ldb * n.max(1)).map(|_| rng.next() as f32).collect();
        let c0: Vec<f32> = (0..ldc * n.max(1)).map(|_| rng.next() as f32).collect();
        let (mut got, mut want) = (c0.clone(), c0.clone());
        gemm_f32(m, n, k, alpha, &a, lda, &b, ldb, beta, &mut got, ldc);
        r::gemm_f32(m, n, k, alpha, &a, lda, &b, ldb, beta, &mut want, ldc);
        for j in 0..n {
            for i in 0..m {
                let mut bound = beta.abs() * c0[i + j * ldc].abs();
                for p in 0..k {
                    bound += alpha.abs() * (a[i + p * lda] * b[p + j * ldb]).abs();
                }
                let tol = 8.0 * (k as f32 + 2.0) * f32::EPSILON * bound + f32::MIN_POSITIVE;
                prop_assert!((got[i + j * ldc] - want[i + j * ldc]).abs() <= tol);
            }
        }
    }

    #[test]
    fn prop_dot_gemv_f64(
        m in 0usize..50, n in 0usize..50, seed in any::<u64>(),
    ) {
        let mut rng = Rng(seed);
        let a: Vec<f64> = (0..m.max(1) * n.max(1)).map(|_| rng.next()).collect();
        let x: Vec<f64> = (0..m.max(n)).map(|_| rng.next()).collect();
        let (mut g1, mut w1) = (vec![0.0; m], vec![0.0; m]);
        gemv_f64(m, n, 1.0, &a, m.max(1), &x[..n], 0.0, &mut g1);
        r::gemv_f64(m, n, 1.0, &a, m.max(1), &x[..n], 0.0, &mut w1);
        let (mut g2, mut w2) = (vec![0.0; n], vec![0.0; n]);
        gemv_t_f64(m, n, 1.0, &a, m.max(1), &x[..m], 0.0, &mut g2);
        r::gemv_t_f64(m, n, 1.0, &a, m.max(1), &x[..m], 0.0, &mut w2);
        for (g, w) in g1.iter().zip(&w1).chain(g2.iter().zip(&w2)) {
            prop_assert!((g - w).abs() <= 1e-12 * (m + n + 2) as f64);
        }
        let d = dot_f64(&x, &x);
        prop_assert!((d - r::dot_f64(&x, &x)).abs() <= 1e-12 * (x.len() + 2) as f64);
    }
}
