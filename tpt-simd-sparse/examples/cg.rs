//! Conjugate gradient on the 2D Poisson matrix, built only from
//! `tpt-simd-sparse` kernels: one SpMV, one dot, one fused vector update
//! and one `xpay` per iteration.

use tpt_simd_sparse::{CsrMatrix, cg_update, dot, spmv_csr, sqnorm, xpay};

fn main() {
    let n = 128;
    let a = CsrMatrix::<f64>::poisson_2d(n);
    let dim = n * n;
    let truth: Vec<f64> = (0..dim).map(|i| ((i * 37 % 11) as f64) - 5.0).collect();
    let mut b = vec![0.0; dim];
    spmv_csr(1.0, &a.view(), &truth, 0.0, &mut b);

    let mut x = vec![0.0; dim];
    let mut r = b.clone();
    let mut p = r.clone();
    let mut q = vec![0.0; dim];
    let mut rr = sqnorm(&r);
    let target = rr * 1e-20;
    let mut iters = 0;
    while rr > target && iters < 2000 {
        spmv_csr(1.0, &a.view(), &p, 0.0, &mut q);
        let alpha = rr / dot(&p, &q);
        let rr_new = cg_update(alpha, &p, &q, &mut x, &mut r);
        xpay(&r, rr_new / rr, &mut p);
        rr = rr_new;
        iters += 1;
    }
    let err = x
        .iter()
        .zip(&truth)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("{dim} unknowns: {iters} iterations, |r|^2 = {rr:.3e}, max error = {err:.3e}");
    assert!(err < 1e-6);
}
