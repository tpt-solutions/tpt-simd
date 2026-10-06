extern crate std;

use crate::*;
use proptest::prelude::*;
use std::vec;
use std::vec::Vec;

const STRATEGIES: [RowStrategy; 4] = [
    RowStrategy::Scalar,
    RowStrategy::Lanes4,
    RowStrategy::Lanes8,
    RowStrategy::Hybrid,
];

/// Small deterministic generator (xorshift64*).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 11) as usize % n.max(1)
    }
    /// Value in roughly [-4, 4) on a 1/16 grid.
    fn val(&mut self) -> f64 {
        (self.below(128) as f64 - 64.0) / 16.0
    }
}

fn random_triplets(rng: &mut Rng, nr: usize, nc: usize, nnz: usize) -> Vec<(usize, usize, f64)> {
    if nr == 0 || nc == 0 {
        return Vec::new();
    }
    (0..nnz)
        .map(|_| (rng.below(nr), rng.below(nc), rng.val()))
        .collect()
}

/// Dense model `A` (row-major, duplicates summed) and `|A|`.
fn dense(nr: usize, nc: usize, t: &[(usize, usize, f64)]) -> (Vec<f64>, Vec<f64>) {
    let mut a = vec![0.0; nr * nc];
    let mut abs = vec![0.0; nr * nc];
    for &(r, c, v) in t {
        a[r * nc + c] += v;
        abs[r * nc + c] += v.abs();
    }
    (a, abs)
}

/// `y_exp[i] = alpha * sum_j op(A)[i][j] x[j] + beta * y[i]`, and the
/// magnitude bound used for the tolerance.
#[allow(clippy::too_many_arguments)]
fn model(
    nr: usize,
    nc: usize,
    t: &[(usize, usize, f64)],
    transposed: bool,
    alpha: f64,
    x: &[f64],
    beta: f64,
    y: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let (a, abs) = dense(nr, nc, t);
    let (m, n) = if transposed { (nc, nr) } else { (nr, nc) };
    let mut e = vec![0.0; m];
    let mut b = vec![0.0; m];
    for i in 0..m {
        let (mut s, mut sa) = (0.0, 0.0);
        for j in 0..n {
            let (v, va) = if transposed {
                (a[j * nc + i], abs[j * nc + i])
            } else {
                (a[i * nc + j], abs[i * nc + j])
            };
            s += v * x[j];
            sa += va * x[j].abs();
        }
        e[i] = alpha * s + if beta == 0.0 { 0.0 } else { beta * y[i] };
        b[i] = alpha.abs() * sa
            + if beta == 0.0 {
                0.0
            } else {
                (beta * y[i]).abs()
            };
    }
    (e, b)
}

fn tol<T: Real>(rowlen: usize) -> f64 {
    let eps = if core::mem::size_of::<T>() == 4 {
        f64::from(f32::EPSILON)
    } else {
        f64::EPSILON
    };
    (rowlen as f64 + 32.0) * eps
}

macro_rules! typed_tests {
    ($m:ident, $t:ident) => {
        mod $m {
            use super::*;

            type T = $t;

            fn cv<'a>(m: &'a CsrMatrix<T>) -> CsrView<'a, T> {
                m.view()
            }
            fn build(nr: usize, nc: usize, t: &[(usize, usize, f64)]) -> CsrMatrix<T> {
                let tt: Vec<(usize, usize, T)> = t.iter().map(|&(r, c, v)| (r, c, v as T)).collect();
                CsrMatrix::from_triplets(nr, nc, &tt).unwrap()
            }
            fn vecs(rng: &mut Rng, n: usize) -> (Vec<f64>, Vec<T>) {
                let v: Vec<f64> = (0..n).map(|_| rng.val()).collect();
                let w = v.iter().map(|&a| a as T).collect();
                (v, w)
            }

            /// Checks all four operators and all strategies against the f64
            /// dense model and the scalar reference.
            fn check_all(nr: usize, nc: usize, t: &[(usize, usize, f64)], alpha: f64, beta: f64, rng: &mut Rng) {
                let m = build(nr, nc, t);
                let csc = m.to_csc();
                let maxrow = t.len().max(1);
                for transposed in [false, true] {
                    let (xl, yl) = if transposed { (nr, nc) } else { (nc, nr) };
                    let (x64, x) = vecs(rng, xl);
                    let (y64, y0) = vecs(rng, yl);
                    let (e, bound) = model(nr, nc, t, transposed, alpha, &x64, beta, &y64);
                    let check = |name: &str, got: &[T]| {
                        for i in 0..got.len() {
                            let err = (f64::from(got[i]) - e[i]).abs();
                            assert!(
                                err <= tol::<T>(maxrow) * bound[i] + 1e-30,
                                "{name} t={transposed} {nr}x{nc} i={i}: got {} want {} (alpha {alpha} beta {beta})",
                                got[i], e[i]
                            );
                        }
                    };
                    let (a, b) = (alpha as T, beta as T);
                    let mut y = y0.clone();
                    if transposed {
                        spmv_csr_t(a, &cv(&m), &x, b, &mut y);
                        check("csr_t", &y);
                        let mut r = y0.clone();
                        reference::spmv_csr_t(a, &cv(&m), &x, b, &mut r);
                        check("ref csr_t", &r);
                        let mut y = y0.clone();
                        spmv_csc_t(a, &csc.view(), &x, b, &mut y);
                        check("csc_t", &y);
                        let mut r = y0.clone();
                        reference::spmv_csc_t(a, &csc.view(), &x, b, &mut r);
                        check("ref csc_t", &r);
                    } else {
                        spmv_csr(a, &cv(&m), &x, b, &mut y);
                        check("csr", &y);
                        for s in STRATEGIES {
                            let mut y = y0.clone();
                            spmv_csr_with(s, a, &cv(&m), &x, b, &mut y);
                            check("csr_with", &y);
                        }
                        let mut r = y0.clone();
                        reference::spmv_csr(a, &cv(&m), &x, b, &mut r);
                        check("ref csr", &r);
                        let mut y = y0.clone();
                        spmv_csc(a, &csc.view(), &x, b, &mut y);
                        check("csc", &y);
                        let mut r = y0.clone();
                        reference::spmv_csc(a, &csc.view(), &x, b, &mut r);
                        check("ref csc", &r);
                    }
                }
            }

            #[test]
            fn degenerate_shapes() {
                let mut rng = Rng(1);
                for (nr, nc) in [(0, 0), (0, 5), (5, 0), (1, 1), (3, 3), (9, 2), (2, 9)] {
                    // all-empty
                    check_all(nr, nc, &[], 1.5, 0.5, &mut rng);
                    check_all(nr, nc, &[], 1.0, 0.0, &mut rng);
                }
                // single nnz
                check_all(4, 6, &[(2, 5, -3.25)], 2.0, 1.0, &mut rng);
                check_all(1, 1, &[(0, 0, 7.0)], 1.0, 0.0, &mut rng);
            }

            #[test]
            fn rows_shorter_and_longer_than_lane_width() {
                let mut rng = Rng(7);
                for len in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100] {
                    let nc = 120;
                    // rows: empty, len, len, empty, len
                    let mut t = Vec::new();
                    for r in [1usize, 2, 4] {
                        for k in 0..len {
                            t.push((r, (k * 7 + r) % nc, rng.val()));
                        }
                    }
                    for (alpha, beta) in [(1.0, 0.0), (1.0, 1.0), (-2.0, 0.5), (0.0, 1.0)] {
                        check_all(5, nc, &t, alpha, beta, &mut rng);
                    }
                }
            }

            #[test]
            fn random_and_poisson() {
                let mut rng = Rng(99);
                for (nr, nc, nnz) in [(50, 50, 400), (37, 91, 1000), (91, 37, 1000), (200, 200, 6400)] {
                    let t = random_triplets(&mut rng, nr, nc, nnz);
                    check_all(nr, nc, &t, 1.25, -0.75, &mut rng);
                }
                let p = CsrMatrix::<T>::poisson_2d(9);
                let mut t = Vec::new();
                for r in 0..p.nrows() {
                    let v = p.view();
                    for k in v.indptr()[r]..v.indptr()[r + 1] {
                        t.push((r, v.indices()[k] as usize, f64::from(v.data()[k])));
                    }
                }
                assert_eq!(t.len(), 5 * 81 - 4 * 9);
                check_all(81, 81, &t, 1.0, 0.0, &mut rng);
                check_all(81, 81, &t, -1.0, 1.0, &mut rng);
            }

            #[test]
            fn duplicates_are_summed_and_unsorted_is_fine() {
                let mut rng = Rng(5);
                let mut t = Vec::new();
                for r in 0..6 {
                    for k in 0..12usize {
                        // duplicated and reverse-ordered columns
                        t.push((r, (11 - k) % 4 + 20 * (r % 2), rng.val()));
                    }
                }
                check_all(6, 40, &t, 1.0, 0.0, &mut rng);
                // Raw view with a duplicated, unsorted row: [3, 1, 3] -> A[0][3] = 1+3, A[0][1] = 2.
                let ip = [0usize, 3];
                let ix = [3u32, 1, 3];
                let d = [1.0 as T, 2.0, 3.0];
                let v = CsrView::try_new(1, 4, &ip, &ix, &d).unwrap();
                let mut y = [0.0 as T];
                spmv_csr(1.0, &v, &[10.0, 100.0, 0.0, 1000.0], 0.0, &mut y);
                assert_eq!(y[0], 4.0 * 1000.0 + 2.0 * 100.0);
            }

            #[test]
            fn beta_zero_overwrites_nan_and_alpha_zero_does_not_hide_nan() {
                let mut rng = Rng(3);
                let t = random_triplets(&mut rng, 20, 20, 120);
                let m = build(20, 20, &t);
                let csc = m.to_csc();
                let x = vec![1.0 as T; 20];
                let mut r = vec![0.0 as T; 20];
                reference::spmv_csr(2.0, &cv(&m), &x, 0.0, &mut r);
                let nan = T::NAN;
                let inf = T::INFINITY;
                let mut y = vec![nan; 20];
                spmv_csr(2.0, &cv(&m), &x, 0.0, &mut y);
                assert_eq!(y.iter().map(|v| v.is_nan()).collect::<Vec<_>>(), vec![false; 20]);
                for s in STRATEGIES {
                    let mut y = vec![inf; 20];
                    spmv_csr_with(s, 2.0, &cv(&m), &x, 0.0, &mut y);
                    for i in 0..20 {
                        assert!((y[i] - r[i]).abs() <= 1e-4 * (1.0 + r[i].abs()));
                    }
                }
                for f in 0..3 {
                    let mut y = vec![nan; 20];
                    match f {
                        0 => spmv_csr_t(2.0, &cv(&m), &x, 0.0, &mut y),
                        1 => spmv_csc(2.0, &csc.view(), &x, 0.0, &mut y),
                        _ => spmv_csc_t(2.0, &csc.view(), &x, 0.0, &mut y),
                    }
                    assert!(y.iter().all(|v| !v.is_nan() && v.is_finite()), "f={f}");
                }
                // beta != 0 does propagate NaN in y
                let mut y = vec![nan; 20];
                spmv_csr(1.0, &cv(&m), &x, 1.0, &mut y);
                assert!(y.iter().all(|v| v.is_nan()));
                // alpha == 0 still propagates NaN from x
                let mut xn = x.clone();
                xn[0] = nan;
                let mut y = vec![0.0 as T; 20];
                spmv_csr(0.0, &cv(&m), &xn, 0.0, &mut y);
                let mut want = vec![0.0 as T; 20];
                reference::spmv_csr(0.0, &cv(&m), &xn, 0.0, &mut want);
                for i in 0..20 {
                    assert_eq!(y[i].is_nan(), want[i].is_nan());
                }
                assert!(want.iter().any(|v| v.is_nan()));
            }

            #[test]
            fn nan_and_inf_match_reference() {
                let mut rng = Rng(11);
                let t = random_triplets(&mut rng, 40, 60, 500);
                let m = build(40, 60, &t);
                let csc = m.to_csc();
                for special in [T::NAN, T::INFINITY, T::NEG_INFINITY] {
                    let mut x: Vec<T> = (0..60).map(|i| (i % 5) as T - 2.0).collect();
                    x[13] = special;
                    x[44] = special;
                    let agree = |name: &str, a: &[T], b: &[T]| {
                        for i in 0..a.len() {
                            if b[i].is_nan() {
                                assert!(a[i].is_nan(), "{name} i={i}");
                            } else if b[i].is_infinite() {
                                assert_eq!(a[i], b[i], "{name} i={i}");
                            } else {
                                assert!(!a[i].is_nan() && !a[i].is_infinite(), "{name} i={i}");
                                assert!((a[i] - b[i]).abs() <= 1e-3 * (1.0 + b[i].abs()));
                            }
                        }
                    };
                    let (mut a, mut b) = (vec![0.0 as T; 40], vec![0.0 as T; 40]);
                    spmv_csr(1.0, &cv(&m), &x, 0.0, &mut a);
                    reference::spmv_csr(1.0, &cv(&m), &x, 0.0, &mut b);
                    agree("csr", &a, &b);
                    spmv_csc(1.0, &csc.view(), &x, 0.0, &mut a);
                    reference::spmv_csc(1.0, &csc.view(), &x, 0.0, &mut b);
                    agree("csc", &a, &b);
                    let mut xt = x.clone();
                    xt.truncate(40);
                    xt[13] = special;
                    let (mut a, mut b) = (vec![0.0 as T; 60], vec![0.0 as T; 60]);
                    spmv_csr_t(1.0, &cv(&m), &xt, 0.0, &mut a);
                    reference::spmv_csr_t(1.0, &cv(&m), &xt, 0.0, &mut b);
                    agree("csr_t", &a, &b);
                    spmv_csc_t(1.0, &csc.view(), &xt, 0.0, &mut a);
                    reference::spmv_csc_t(1.0, &csc.view(), &xt, 0.0, &mut b);
                    agree("csc_t", &a, &b);
                }
                // NaN stored in the matrix
                let tt = [(0usize, 0usize, T::NAN), (1, 1, 2.0)];
                let m = CsrMatrix::from_triplets(2, 2, &tt).unwrap();
                let mut y = [0.0 as T; 2];
                spmv_csr(1.0, &m.view(), &[1.0, 1.0], 0.0, &mut y);
                assert!(y[0].is_nan());
                assert_eq!(y[1], 2.0);
            }

            #[test]
            fn vector_kernels_match_reference() {
                let mut rng = Rng(21);
                for n in (0..140).chain([255, 256, 1000, 4097]) {
                    let (x64, x) = vecs(&mut rng, n);
                    let (y64, y0) = vecs(&mut rng, n);
                    let (z64, z) = vecs(&mut rng, n);
                    let alpha = 0.75 as T;
                    let abs: f64 = x64.iter().zip(&y64).map(|(a, b)| (a * b).abs()).sum();
                    let t = tol::<T>(n);
                    let exact: f64 = x64.iter().zip(&y64).map(|(a, b)| a * b).sum();
                    assert!((f64::from(dot(&x, &y0)) - exact).abs() <= t * abs + 1e-30, "dot n={n}");
                    assert!((f64::from(reference::dot(&x, &y0)) - exact).abs() <= t * abs + 1e-30);

                    // axpy / xpay
                    let mut y = y0.clone();
                    axpy(alpha, &x, &mut y);
                    let mut r = y0.clone();
                    reference::axpy(alpha, &x, &mut r);
                    assert_eq!(y, r);
                    let mut p = y0.clone();
                    xpay(&x, alpha, &mut p);
                    for i in 0..n {
                        assert_eq!(p[i], x[i] + alpha * y0[i]);
                    }
                    let mut p = y0.clone();
                    bicgstab_p_update(alpha, 0.5, &x, &z, &mut p);
                    for i in 0..n {
                        assert_eq!(p[i], x[i] + alpha * (y0[i] - 0.5 * z[i]));
                    }

                    // fused == unfused, bit for bit
                    let mut yf = y0.clone();
                    let got = axpy_dot(alpha, &x, &mut yf, &z);
                    let mut yu = y0.clone();
                    axpy(alpha, &x, &mut yu);
                    assert_eq!(yf, yu);
                    assert_eq!(got.to_bits(), dot(&yu, &z).to_bits(), "axpy_dot n={n}");
                    let mut yf = y0.clone();
                    let got = axpy_sqnorm(alpha, &x, &mut yf);
                    assert_eq!(yf, yu);
                    assert_eq!(got.to_bits(), sqnorm(&yu).to_bits(), "axpy_sqnorm n={n}");
                    let _ = z64;

                    let (mut xf, mut rf) = (y0.clone(), z.clone());
                    let got = cg_update(alpha, &x, &y0, &mut xf, &mut rf);
                    let (mut xu, mut ru) = (y0.clone(), z.clone());
                    axpy(alpha, &x, &mut xu);
                    axpy(-alpha, &y0, &mut ru);
                    assert_eq!((xf, rf.clone()), (xu, ru.clone()));
                    assert_eq!(got.to_bits(), sqnorm(&ru).to_bits(), "cg_update n={n}");
                }
                assert_eq!(dot::<T>(&[], &[]), 0.0);
                assert_eq!(sqnorm::<T>(&[]), 0.0);
            }

            #[test]
            fn vector_kernels_nan_inf() {
                let mut x = vec![1.0 as T; 70];
                let y = vec![2.0 as T; 70];
                x[40] = T::NAN;
                assert!(dot(&x, &y).is_nan());
                x[40] = T::INFINITY;
                assert_eq!(dot(&x, &y), T::INFINITY);
                x[41] = T::NEG_INFINITY;
                assert!(dot(&x, &y).is_nan());
                let mut r = vec![1.0 as T; 70];
                r[69] = T::INFINITY;
                assert_eq!(axpy_sqnorm(0.0, &y, &mut r), T::INFINITY);
            }

            #[test]
            #[should_panic(expected = "length mismatch")]
            fn dot_length_mismatch_panics() {
                let _ = dot::<T>(&[1.0], &[1.0, 2.0]);
            }

            #[test]
            #[should_panic(expected = "wrong length")]
            fn spmv_wrong_x_length_panics() {
                let m = CsrMatrix::<T>::poisson_2d(3);
                let mut y = vec![0.0; 9];
                spmv_csr(1.0, &m.view(), &[0.0; 8], 0.0, &mut y);
            }

            #[test]
            fn cg_solves_poisson() {
                let n = 24;
                let a = CsrMatrix::<T>::poisson_2d(n);
                let dim = n * n;
                let truth: Vec<T> = (0..dim).map(|i| ((i * 37 % 11) as T) - 5.0).collect();
                let mut b = vec![0.0 as T; dim];
                spmv_csr(1.0, &a.view(), &truth, 0.0, &mut b);
                // CG built only from this crate's kernels.
                let mut x = vec![0.0 as T; dim];
                let mut r = b.clone();
                let mut p = r.clone();
                let mut q = vec![0.0 as T; dim];
                let mut rr = sqnorm(&r);
                let bnorm = rr;
                let stop = if core::mem::size_of::<T>() == 4 { 1e-12 as T } else { 1e-24 as T };
                let mut iters = 0;
                while rr > bnorm * stop && iters < 400 {
                    spmv_csr(1.0, &a.view(), &p, 0.0, &mut q);
                    let alpha = rr / dot(&p, &q);
                    let rr_new = cg_update(alpha, &p, &q, &mut x, &mut r);
                    xpay(&r, rr_new / rr, &mut p);
                    rr = rr_new;
                    iters += 1;
                }
                assert!(iters < 400, "no convergence");
                let err = x.iter().zip(&truth).map(|(a, b)| (a - b).abs()).fold(0.0 as T, T::max);
                assert!(f64::from(err) < 5e-2 * if core::mem::size_of::<T>() == 4 { 1.0 } else { 1e-6 }, "err {err} iters {iters}");
            }
        }
    };
}

typed_tests!(f32_tests, f32);
typed_tests!(f64_tests, f64);

#[test]
fn invalid_input_is_rejected() {
    let d = [1.0f32, 2.0];
    // indptr length
    assert_eq!(
        CsrView::try_new(2, 3, &[0, 1], &[0], &[1.0f32]).unwrap_err(),
        SparseError::IndptrLength {
            expected: 3,
            got: 2
        }
    );
    // indptr start
    assert_eq!(
        CsrView::try_new(1, 3, &[1, 2], &[0, 1], &d).unwrap_err(),
        SparseError::IndptrStart(1)
    );
    // not monotone
    assert_eq!(
        CsrView::try_new(2, 3, &[0, 2, 1], &[0, 1], &d).unwrap_err(),
        SparseError::IndptrNotMonotone { at: 1 }
    );
    // nnz mismatches
    assert_eq!(
        CsrView::try_new(1, 3, &[0, 2], &[0, 1], &d[..1]).unwrap_err(),
        SparseError::NnzMismatch {
            indptr_last: 2,
            indices: 2,
            data: 1
        }
    );
    assert!(matches!(
        CsrView::try_new(1, 3, &[0, 3], &[0, 1], &d).unwrap_err(),
        SparseError::NnzMismatch { .. }
    ));
    // index out of range (CSR: >= ncols, CSC: >= nrows)
    assert_eq!(
        CsrView::try_new(1, 3, &[0, 2], &[0, 3], &d).unwrap_err(),
        SparseError::IndexOutOfRange {
            pos: 1,
            index: 3,
            bound: 3
        }
    );
    assert_eq!(
        CscView::try_new(2, 2, &[0, 1, 2], &[0, 2], &d).unwrap_err(),
        SparseError::IndexOutOfRange {
            pos: 1,
            index: 2,
            bound: 2
        }
    );
    assert!(CsrMatrix::<f32>::from_triplets(2, 2, &[(2, 0, 1.0)]).is_err());
    assert!(CsrMatrix::<f32>::from_triplets(2, 2, &[(0, 2, 1.0)]).is_err());
    assert!(CsrMatrix::<f32>::try_new(1, 1, vec![0, 1], vec![5], vec![1.0]).is_err());
    // valid edge: empty matrix
    assert!(CsrView::<f32>::try_new(0, 0, &[0], &[], &[]).is_ok());
    assert!(CsrView::<f32>::try_new(0, 0, &[], &[], &[]).is_err());
}

fn small_matrix() -> impl Strategy<Value = (usize, usize, Vec<(usize, usize, i8)>)> {
    (1usize..40, 1usize..40).prop_flat_map(|(nr, nc)| {
        (
            Just(nr),
            Just(nc),
            prop::collection::vec((0..nr, 0..nc, -64i8..64), 0..300),
        )
    })
}

proptest! {
    #[test]
    fn spmv_matches_dense_model(
        (nr, nc, t) in small_matrix(),
        alpha in -4i8..5,
        beta in -4i8..5,
        seed in 1u64..u64::MAX,
    ) {
        let mut rng = Rng(seed);
        let t64: Vec<(usize, usize, f64)> =
            t.iter().map(|&(r, c, v)| (r, c, f64::from(v) / 16.0)).collect();
        let m = CsrMatrix::<f64>::from_triplets(nr, nc, &t64).unwrap();
        let csc = m.to_csc();
        let (alpha, beta) = (f64::from(alpha) * 0.5, f64::from(beta) * 0.5);
        for transposed in [false, true] {
            let (xl, yl) = if transposed { (nr, nc) } else { (nc, nr) };
            let x: Vec<f64> = (0..xl).map(|_| rng.val()).collect();
            let y0: Vec<f64> = (0..yl).map(|_| rng.val()).collect();
            let (e, bound) = model(nr, nc, &t64, transposed, alpha, &x, beta, &y0);
            let mut ys = vec![y0.clone(), y0.clone()];
            if transposed {
                spmv_csr_t(alpha, &m.view(), &x, beta, &mut ys[0]);
                spmv_csc_t(alpha, &csc.view(), &x, beta, &mut ys[1]);
            } else {
                spmv_csr(alpha, &m.view(), &x, beta, &mut ys[0]);
                spmv_csc(alpha, &csc.view(), &x, beta, &mut ys[1]);
            }
            for y in &ys {
                for i in 0..yl {
                    prop_assert!((y[i] - e[i]).abs() <= tol::<f64>(t.len()) * bound[i] + 1e-300);
                }
            }
        }
    }

    #[test]
    fn poisson_matches_stencil(n in 1usize..14, seed in 1u64..u64::MAX) {
        let mut rng = Rng(seed);
        let a = CsrMatrix::<f64>::poisson_2d(n);
        let x: Vec<f64> = (0..n * n).map(|_| rng.val()).collect();
        let mut y = vec![0.0; n * n];
        spmv_csr(1.0, &a.view(), &x, 0.0, &mut y);
        for i in 0..n {
            for j in 0..n {
                let at = |a: isize, b: isize| {
                    if a < 0 || b < 0 || a >= n as isize || b >= n as isize { 0.0 } else { x[a as usize * n + b as usize] }
                };
                let (ii, jj) = (i as isize, j as isize);
                let want = 4.0 * at(ii, jj) - at(ii - 1, jj) - at(ii + 1, jj) - at(ii, jj - 1) - at(ii, jj + 1);
                prop_assert!((y[i * n + j] - want).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn dot_fused_equals_unfused(n in 0usize..300, seed in 1u64..u64::MAX) {
        let mut rng = Rng(seed);
        let x: Vec<f64> = (0..n).map(|_| rng.val()).collect();
        let y: Vec<f64> = (0..n).map(|_| rng.val()).collect();
        let z: Vec<f64> = (0..n).map(|_| rng.val()).collect();
        let mut a = y.clone();
        let got = axpy_dot(0.3, &x, &mut a, &z);
        let mut b = y.clone();
        axpy(0.3, &x, &mut b);
        prop_assert_eq!(&a, &b);
        prop_assert_eq!(got.to_bits(), dot(&b, &z).to_bits());
    }
}
