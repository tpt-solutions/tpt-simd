//! BLAS-style kernels: dot products, matrix multiply (column-major) and complex GEMM.
//!
//! Run with: `cargo run -p tpt-simd-blas --example blas`

use tpt_simd_blas::{
    dot_f32, dotc_c32, gemm_c32, gemm_f32, gemm_with_workspace_f32, gemm_workspace_len_f32,
};

fn main() {
    assert_eq!(dot_f32(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);

    // C (2x2) = A (2x3) * B (3x2), column-major.
    let a = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let b = [7.0, 9.0, 11.0, 8.0, 10.0, 12.0];
    let mut c = [0.0f32; 4];
    gemm_f32(2, 2, 3, 1.0, &a, 2, &b, 3, 0.0, &mut c, 2);
    assert_eq!(c, [58.0, 139.0, 64.0, 154.0]);

    // Same product with a caller-owned workspace, so repeated calls do not allocate.
    let (m, n, k) = (2, 2, 3);
    let mut ws = vec![0.0f32; gemm_workspace_len_f32(m, n, k)];
    let mut c2 = [0.0f32; 4];
    gemm_with_workspace_f32(m, n, k, 1.0, &a, 2, &b, 3, 0.0, &mut c2, 2, &mut ws);
    assert_eq!(c2, c);

    // Complex: conj(i) * 1 = -i.
    assert_eq!(dotc_c32(&[0.0], &[1.0], &[1.0], &[0.0]), [0.0, -1.0]);

    // Complex GEMM for 1x1 matrices: (1+2i)(3+4i) = -5+10i.
    let (mut cr, mut ci) = ([0.0f32], [0.0f32]);
    gemm_c32(
        1,
        1,
        1,
        [1.0, 0.0],
        &[1.0],
        &[2.0],
        1,
        &[3.0],
        &[4.0],
        1,
        [0.0, 0.0],
        &mut cr,
        &mut ci,
        1,
    );
    assert_eq!((cr[0], ci[0]), (-5.0, 10.0));

    println!("blas: all checks passed");
}
