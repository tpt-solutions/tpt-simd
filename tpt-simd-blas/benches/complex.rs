//! Complex gemm / gemv / level-1 vs the naive scalar split-plane loops in `tpt_simd_blas::reference`.
//!
//! Throughput is reported as real flop/s via `Throughput::Elements`
//! (complex multiply-add = 8 flops). Build with
//! `RUSTFLAGS="-C target-cpu=native"` to enable the AVX2+FMA gemm kernel.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_blas::{self as blas, reference as naive};

fn fill(n: usize, seed: u32) -> Vec<f64> {
    (0..n)
        .map(|i| {
            ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) as f64 / 16777216.0 - 0.5
        })
        .collect()
}
fn fill32(n: usize, seed: u32) -> Vec<f32> {
    fill(n, seed).into_iter().map(|v| v as f32).collect()
}

fn gemm(c: &mut Criterion) {
    for &n in &[64usize, 256] {
        let mut g = c.benchmark_group(format!("cgemm_{n}"));
        g.throughput(Throughput::Elements(8 * (n * n * n) as u64));
        let (ar, ai, br, bi) = (
            fill32(n * n, 1),
            fill32(n * n, 2),
            fill32(n * n, 3),
            fill32(n * n, 4),
        );
        let (mut cr, mut ci) = (vec![0.0f32; n * n], vec![0.0f32; n * n]);
        g.bench_function("c32/naive", |b| {
            b.iter(|| {
                naive::gemm_c32(
                    n,
                    n,
                    n,
                    [1.0, 0.0],
                    black_box(&ar),
                    black_box(&ai),
                    n,
                    black_box(&br),
                    black_box(&bi),
                    n,
                    [0.0, 0.0],
                    &mut cr,
                    &mut ci,
                    n,
                )
            })
        });
        g.bench_function("c32/tpt_split", |b| {
            b.iter(|| {
                blas::gemm_c32(
                    n,
                    n,
                    n,
                    [1.0, 0.0],
                    black_box(&ar),
                    black_box(&ai),
                    n,
                    black_box(&br),
                    black_box(&bi),
                    n,
                    [0.0, 0.0],
                    &mut cr,
                    &mut ci,
                    n,
                )
            })
        });
        let (a, bb): (Vec<[f32; 2]>, Vec<[f32; 2]>) = (
            ar.iter().zip(&ai).map(|(r, i)| [*r, *i]).collect(),
            br.iter().zip(&bi).map(|(r, i)| [*r, *i]).collect(),
        );
        let mut cc = vec![[0.0f32; 2]; n * n];
        g.bench_function("c32/tpt_interleaved", |b| {
            b.iter(|| {
                blas::gemm_il_c32(
                    n,
                    n,
                    n,
                    [1.0, 0.0],
                    black_box(&a),
                    n,
                    black_box(&bb),
                    n,
                    [0.0, 0.0],
                    &mut cc,
                    n,
                )
            })
        });
        let (ar, ai, br, bi) = (
            fill(n * n, 1),
            fill(n * n, 2),
            fill(n * n, 3),
            fill(n * n, 4),
        );
        let (mut cr, mut ci) = (vec![0.0f64; n * n], vec![0.0f64; n * n]);
        g.bench_function("c64/naive", |b| {
            b.iter(|| {
                naive::gemm_c64(
                    n,
                    n,
                    n,
                    [1.0, 0.0],
                    black_box(&ar),
                    black_box(&ai),
                    n,
                    black_box(&br),
                    black_box(&bi),
                    n,
                    [0.0, 0.0],
                    &mut cr,
                    &mut ci,
                    n,
                )
            })
        });
        g.bench_function("c64/tpt_split", |b| {
            b.iter(|| {
                blas::gemm_c64(
                    n,
                    n,
                    n,
                    [1.0, 0.0],
                    black_box(&ar),
                    black_box(&ai),
                    n,
                    black_box(&br),
                    black_box(&bi),
                    n,
                    [0.0, 0.0],
                    &mut cr,
                    &mut ci,
                    n,
                )
            })
        });
        g.bench_function("c64/tpt_split_alpha", |b| {
            b.iter(|| {
                blas::gemm_c64(
                    n,
                    n,
                    n,
                    [0.5, 0.25],
                    black_box(&ar),
                    black_box(&ai),
                    n,
                    black_box(&br),
                    black_box(&bi),
                    n,
                    [0.0, 0.0],
                    &mut cr,
                    &mut ci,
                    n,
                )
            })
        });
        g.finish();
    }
}

fn gemv(c: &mut Criterion) {
    let n = 1024usize;
    let mut g = c.benchmark_group("cgemv_1024");
    g.throughput(Throughput::Elements(8 * (n * n) as u64));
    let (ar, ai, xr, xi) = (
        fill32(n * n, 5),
        fill32(n * n, 6),
        fill32(n, 7),
        fill32(n, 8),
    );
    let (mut yr, mut yi) = (vec![0.0f32; n], vec![0.0f32; n]);
    g.bench_function("c32_N/naive", |b| {
        b.iter(|| {
            naive::gemv_c32(
                n,
                n,
                [1.0, 0.0],
                black_box(&ar),
                black_box(&ai),
                n,
                &xr,
                &xi,
                [0.0, 0.0],
                &mut yr,
                &mut yi,
            )
        })
    });
    g.bench_function("c32_N/tpt", |b| {
        b.iter(|| {
            blas::gemv_c32(
                n,
                n,
                [1.0, 0.0],
                black_box(&ar),
                black_box(&ai),
                n,
                &xr,
                &xi,
                [0.0, 0.0],
                &mut yr,
                &mut yi,
            )
        })
    });
    g.bench_function("c32_H/naive", |b| {
        b.iter(|| {
            naive::gemv_h_c32(
                n,
                n,
                [1.0, 0.0],
                black_box(&ar),
                black_box(&ai),
                n,
                &xr,
                &xi,
                [0.0, 0.0],
                &mut yr,
                &mut yi,
            )
        })
    });
    g.bench_function("c32_H/tpt", |b| {
        b.iter(|| {
            blas::gemv_h_c32(
                n,
                n,
                [1.0, 0.0],
                black_box(&ar),
                black_box(&ai),
                n,
                &xr,
                &xi,
                [0.0, 0.0],
                &mut yr,
                &mut yi,
            )
        })
    });
    let a: Vec<[f32; 2]> = ar.iter().zip(&ai).map(|(r, i)| [*r, *i]).collect();
    let x: Vec<[f32; 2]> = xr.iter().zip(&xi).map(|(r, i)| [*r, *i]).collect();
    let mut y = vec![[0.0f32; 2]; n];
    g.bench_function("c32_N/tpt_interleaved", |b| {
        b.iter(|| blas::gemv_il_c32(n, n, [1.0, 0.0], black_box(&a), n, &x, [0.0, 0.0], &mut y))
    });
    g.finish();
}

fn level1(c: &mut Criterion) {
    let n = 4096usize;
    let mut g = c.benchmark_group("clevel1_4096");
    g.throughput(Throughput::Elements(8 * n as u64));
    let (xr, xi, yr0, yi0) = (fill32(n, 1), fill32(n, 2), fill32(n, 3), fill32(n, 4));
    g.bench_function("dotc/naive", |b| {
        b.iter(|| naive::dotc_c32(black_box(&xr), black_box(&xi), &yr0, &yi0))
    });
    g.bench_function("dotc/tpt", |b| {
        b.iter(|| blas::dotc_c32(black_box(&xr), black_box(&xi), &yr0, &yi0))
    });
    let x: Vec<[f32; 2]> = xr.iter().zip(&xi).map(|(r, i)| [*r, *i]).collect();
    let y: Vec<[f32; 2]> = yr0.iter().zip(&yi0).map(|(r, i)| [*r, *i]).collect();
    g.bench_function("dotc/tpt_interleaved", |b| {
        b.iter(|| blas::dotc_il_c32(black_box(&x), &y))
    });
    g.bench_function("dotu/tpt", |b| {
        b.iter(|| blas::dotu_c32(black_box(&xr), black_box(&xi), &yr0, &yi0))
    });
    g.bench_function("nrm2/naive", |b| {
        b.iter(|| naive::nrm2_c32(black_box(&xr), black_box(&xi)))
    });
    g.bench_function("nrm2/tpt", |b| {
        b.iter(|| blas::nrm2_c32(black_box(&xr), black_box(&xi)))
    });
    g.bench_function("asum/naive", |b| {
        b.iter(|| naive::asum_c32(black_box(&xr), black_box(&xi)))
    });
    g.bench_function("asum/tpt", |b| {
        b.iter(|| blas::asum_c32(black_box(&xr), black_box(&xi)))
    });
    let (mut yr, mut yi) = (yr0.clone(), yi0.clone());
    g.bench_function("axpy/naive", |b| {
        b.iter(|| {
            naive::axpy_c32(
                [0.5, 0.25],
                black_box(&xr),
                black_box(&xi),
                &mut yr,
                &mut yi,
            )
        })
    });
    g.bench_function("axpy/tpt", |b| {
        b.iter(|| {
            blas::axpy_c32(
                [0.5, 0.25],
                black_box(&xr),
                black_box(&xi),
                &mut yr,
                &mut yi,
            )
        })
    });
    let mut yc = y.clone();
    g.bench_function("axpy/tpt_interleaved", |b| {
        b.iter(|| blas::axpy_il_c32([0.5, 0.25], black_box(&x), &mut yc))
    });
    g.finish();
}

criterion_group!(benches, gemm, gemv, level1);
criterion_main!(benches);
