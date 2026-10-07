//! gemm / gemv / dot, and LU / Cholesky solves, vs the naive scalar loops in `tpt_simd_blas::reference`.
//!
//! Throughput is reported as flop/s via `Throughput::Elements(flops)`
//! (criterion prints it as "elem/s", i.e. GFLOP/s when divided by 1e9).
//! Build with `RUSTFLAGS="-C target-cpu=native"` to enable the AVX2+FMA kernel.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_blas::{self as blas, reference as naive};

fn fill32(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) as f32 / 16777216.0 - 0.5
        })
        .collect()
}
fn fill64(n: usize, seed: u32) -> Vec<f64> {
    fill32(n, seed).into_iter().map(f64::from).collect()
}

fn gemm(c: &mut Criterion) {
    for &n in &[64usize, 256, 512] {
        let mut g = c.benchmark_group(format!("gemm_{n}"));
        g.throughput(Throughput::Elements(2 * (n * n * n) as u64));
        if n >= 512 {
            g.sample_size(10);
        }
        let (a, b) = (fill32(n * n, 1), fill32(n * n, 2));
        let mut cm = vec![0.0f32; n * n];
        g.bench_function("f32/naive", |bn| {
            bn.iter(|| {
                naive::gemm_f32(
                    n,
                    n,
                    n,
                    1.0,
                    black_box(&a),
                    n,
                    black_box(&b),
                    n,
                    0.0,
                    &mut cm,
                    n,
                )
            })
        });
        g.bench_function("f32/tpt", |bn| {
            bn.iter(|| {
                blas::gemm_f32(
                    n,
                    n,
                    n,
                    1.0,
                    black_box(&a),
                    n,
                    black_box(&b),
                    n,
                    0.0,
                    &mut cm,
                    n,
                )
            })
        });
        let mut ws = vec![0.0f32; blas::gemm_workspace_len_f32(n, n, n)];
        g.bench_function("f32/tpt_workspace", |bn| {
            bn.iter(|| {
                blas::gemm_with_workspace_f32(
                    n,
                    n,
                    n,
                    1.0,
                    black_box(&a),
                    n,
                    black_box(&b),
                    n,
                    0.0,
                    &mut cm,
                    n,
                    &mut ws,
                )
            })
        });
        let (a, b) = (fill64(n * n, 1), fill64(n * n, 2));
        let mut cm = vec![0.0f64; n * n];
        g.bench_function("f64/naive", |bn| {
            bn.iter(|| {
                naive::gemm_f64(
                    n,
                    n,
                    n,
                    1.0,
                    black_box(&a),
                    n,
                    black_box(&b),
                    n,
                    0.0,
                    &mut cm,
                    n,
                )
            })
        });
        g.bench_function("f64/tpt", |bn| {
            bn.iter(|| {
                blas::gemm_f64(
                    n,
                    n,
                    n,
                    1.0,
                    black_box(&a),
                    n,
                    black_box(&b),
                    n,
                    0.0,
                    &mut cm,
                    n,
                )
            })
        });
        g.finish();
    }
}

fn gemv(c: &mut Criterion) {
    let n = 1024usize;
    let mut g = c.benchmark_group("gemv_1024");
    g.throughput(Throughput::Elements(2 * (n * n) as u64));
    let a = fill32(n * n, 3);
    let x = fill32(n, 4);
    let mut y = vec![0.0f32; n];
    g.bench_function("f32/naive_n", |b| {
        b.iter(|| naive::gemv_f32(n, n, 1.0, black_box(&a), n, black_box(&x), 0.0, &mut y))
    });
    g.bench_function("f32/tpt_n", |b| {
        b.iter(|| blas::gemv_f32(n, n, 1.0, black_box(&a), n, black_box(&x), 0.0, &mut y))
    });
    g.bench_function("f32/naive_t", |b| {
        b.iter(|| naive::gemv_t_f32(n, n, 1.0, black_box(&a), n, black_box(&x), 0.0, &mut y))
    });
    g.bench_function("f32/tpt_t", |b| {
        b.iter(|| blas::gemv_t_f32(n, n, 1.0, black_box(&a), n, black_box(&x), 0.0, &mut y))
    });
    g.finish();
}

fn level1(c: &mut Criterion) {
    for &n in &[4096usize, 1 << 20] {
        let mut g = c.benchmark_group(format!("level1_{n}"));
        g.throughput(Throughput::Elements(2 * n as u64));
        let (x, mut y) = (fill32(n, 5), fill32(n, 6));
        g.bench_function("dot_f32/naive", |b| {
            b.iter(|| naive::dot_f32(black_box(&x), black_box(&y)))
        });
        g.bench_function("dot_f32/tpt", |b| {
            b.iter(|| blas::dot_f32(black_box(&x), black_box(&y)))
        });
        g.bench_function("nrm2_f32/naive", |b| {
            b.iter(|| naive::nrm2_f32(black_box(&x)))
        });
        g.bench_function("nrm2_f32/tpt", |b| b.iter(|| blas::nrm2_f32(black_box(&x))));
        g.bench_function("asum_f32/naive", |b| {
            b.iter(|| naive::asum_f32(black_box(&x)))
        });
        g.bench_function("asum_f32/tpt", |b| b.iter(|| blas::asum_f32(black_box(&x))));
        g.bench_function("axpy_f32/naive", |b| {
            b.iter(|| naive::axpy_f32(0.5, black_box(&x), &mut y))
        });
        g.bench_function("axpy_f32/tpt", |b| {
            b.iter(|| blas::axpy_f32(0.5, black_box(&x), &mut y))
        });
        let (x, y) = (fill64(n, 5), fill64(n, 6));
        g.bench_function("dot_f64/naive", |b| {
            b.iter(|| naive::dot_f64(black_box(&x), black_box(&y)))
        });
        g.bench_function("dot_f64/tpt", |b| {
            b.iter(|| blas::dot_f64(black_box(&x), black_box(&y)))
        });
        g.finish();
    }
}

fn xorshift_fill(n: usize, mut s: u64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

/// Generates the LU-solve and Cholesky benchmark groups for one element type.
macro_rules! factor_benches {
    ($lu:ident, $chol:ident, $t:ident, $getrf:ident, $getrs:ident, $potrf:ident, $potrs:ident, $tag:literal) => {
        fn $lu(c: &mut Criterion) {
            for &n in &[64usize, 256, 512] {
                let mut g = c.benchmark_group(format!("lu_solve_{}_{n}", $tag));
                g.throughput(Throughput::Elements((2 * n * n * n / 3 + 2 * n * n) as u64));
                if n >= 512 {
                    g.sample_size(10);
                }
                let a: Vec<$t> = xorshift_fill(n * n, 42)
                    .into_iter()
                    .map(|v| v as $t)
                    .collect();
                let b: Vec<$t> = xorshift_fill(n, 43).into_iter().map(|v| v as $t).collect();
                let mut work = a.clone();
                let mut x = b.clone();
                let mut ipiv = vec![0usize; n];
                g.bench_function("naive", |bn| {
                    bn.iter(|| {
                        work.copy_from_slice(black_box(&a));
                        x.copy_from_slice(&b);
                        naive::$getrf(n, n, &mut work, n, &mut ipiv).unwrap();
                        naive::$getrs(n, 1, &work, n, &ipiv, &mut x, n);
                    })
                });
                g.bench_function("tpt", |bn| {
                    bn.iter(|| {
                        work.copy_from_slice(black_box(&a));
                        x.copy_from_slice(&b);
                        blas::$getrf(n, n, &mut work, n, &mut ipiv).unwrap();
                        blas::$getrs(n, 1, &work, n, &ipiv, &mut x, n);
                    })
                });
                g.finish();
            }
        }

        fn $chol(c: &mut Criterion) {
            for &n in &[64usize, 256, 512] {
                let mut g = c.benchmark_group(format!("cholesky_{}_{n}", $tag));
                g.throughput(Throughput::Elements((n * n * n / 3 + 2 * n * n) as u64));
                if n >= 512 {
                    g.sample_size(10);
                }
                // Symmetric, diagonally dominant => SPD.
                let mut a: Vec<$t> = xorshift_fill(n * n, 44)
                    .into_iter()
                    .map(|v| v as $t)
                    .collect();
                for j in 0..n {
                    for i in 0..j {
                        a[i + j * n] = a[j + i * n];
                    }
                    a[j + j * n] = a[j + j * n].abs() + n as $t;
                }
                let b: Vec<$t> = xorshift_fill(n, 45).into_iter().map(|v| v as $t).collect();
                let mut work = a.clone();
                let mut x = b.clone();
                g.bench_function("naive", |bn| {
                    bn.iter(|| {
                        work.copy_from_slice(black_box(&a));
                        x.copy_from_slice(&b);
                        naive::$potrf(n, &mut work, n).unwrap();
                        naive::$potrs(n, 1, &work, n, &mut x, n);
                    })
                });
                g.bench_function("tpt", |bn| {
                    bn.iter(|| {
                        work.copy_from_slice(black_box(&a));
                        x.copy_from_slice(&b);
                        blas::$potrf(n, &mut work, n).unwrap();
                        blas::$potrs(n, 1, &work, n, &mut x, n);
                    })
                });
                g.finish();
            }
        }
    };
}

factor_benches!(
    lu_f64, chol_f64, f64, getrf_f64, getrs_f64, potrf_f64, potrs_f64, "f64"
);
factor_benches!(
    lu_f32, chol_f32, f32, getrf_f32, getrs_f32, potrf_f32, potrs_f32, "f32"
);

/// Factorisation only (no solve), f64, to separate the blocked kernels' cost.
fn factor_only(c: &mut Criterion) {
    for &n in &[64usize, 256, 512] {
        let mut g = c.benchmark_group(format!("getrf_f64_{n}"));
        g.throughput(Throughput::Elements((2 * n * n * n / 3) as u64));
        if n >= 512 {
            g.sample_size(10);
        }
        let a = xorshift_fill(n * n, 42);
        let mut work = a.clone();
        let mut ipiv = vec![0usize; n];
        g.bench_function("naive", |bn| {
            bn.iter(|| {
                work.copy_from_slice(black_box(&a));
                naive::getrf_f64(n, n, &mut work, n, &mut ipiv).unwrap();
            })
        });
        g.bench_function("tpt", |bn| {
            bn.iter(|| {
                work.copy_from_slice(black_box(&a));
                blas::getrf_f64(n, n, &mut work, n, &mut ipiv).unwrap();
            })
        });
        g.finish();
    }
}

criterion_group!(
    benches,
    gemm,
    gemv,
    level1,
    lu_f64,
    chol_f64,
    lu_f32,
    chol_f32,
    factor_only
);
criterion_main!(benches);
