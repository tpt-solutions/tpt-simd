//! Single-threaded f64 gemm (column-major): tpt-simd-blas vs gemm, matrixmultiply, faer. Phase 12 overlap review.
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn fill(n: usize, seed: u32) -> Vec<f64> {
    (0..n)
        .map(|i| ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) as f64 / 16777216.0 - 0.5)
        .collect()
}

fn bench(c: &mut Criterion) {
    for &n in &[64usize, 256, 512] {
        let mut g = c.benchmark_group(format!("gemm_f64_{n}"));
        g.throughput(Throughput::Elements(2 * (n * n * n) as u64));
        if n >= 512 {
            g.sample_size(30);
            g.measurement_time(std::time::Duration::from_secs(10));
        }
        let (a, b) = (fill(n * n, 1), fill(n * n, 2));
        let mut cm = vec![0.0f64; n * n];
        let mut ws = vec![0.0f64; tpt_simd_blas::gemm_workspace_len_f64(n, n, n)];
        g.bench_function("tpt_workspace", |bn| {
            bn.iter(|| {
                tpt_simd_blas::gemm_with_workspace_f64(
                    n, n, n, 1.0, black_box(&a), n, black_box(&b), n, 0.0, &mut cm, n, &mut ws,
                )
            })
        });
        // Column-major: element (i, j) lives at i + j * n, so row stride = 1, column stride = n.
        g.bench_function("gemm", |bn| {
            bn.iter(|| unsafe {
                gemm::gemm(
                    n, n, n,
                    cm.as_mut_ptr(), n as isize, 1, false,
                    black_box(a.as_ptr()), n as isize, 1,
                    black_box(b.as_ptr()), n as isize, 1,
                    1.0f64, 0.0f64, false, false, false, gemm::Parallelism::None,
                )
            })
        });
        g.bench_function("matrixmultiply", |bn| {
            bn.iter(|| unsafe {
                matrixmultiply::dgemm(
                    n, n, n, 1.0, black_box(a.as_ptr()), 1, n as isize,
                    black_box(b.as_ptr()), 1, n as isize,
                    0.0, cm.as_mut_ptr(), 1, n as isize,
                )
            })
        });
        g.bench_function("faer", |bn| {
            let fa = faer::mat::MatRef::from_column_major_slice(&a, n, n);
            let fb = faer::mat::MatRef::from_column_major_slice(&b, n, n);
            bn.iter(|| {
                let fc = faer::mat::MatMut::from_column_major_slice_mut(&mut cm, n, n);
                faer::linalg::matmul::matmul(
                    fc, faer::Accum::Replace, black_box(fa), black_box(fb), 1.0f64, faer::Par::Seq,
                )
            })
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
