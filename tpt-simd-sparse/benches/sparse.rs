//! SpMV and fused CG vector updates vs naive scalar references.
//!
//! Matrices: 2D Poisson 5-point stencil on a 256x256 grid (65 536 rows,
//! ~326k nnz, banded access to `x`) and a random sparse matrix with 32
//! nnz/row (65 536 x 65 536, random access to `x`). Row strategies are
//! compared head to head, including a hardware-gather variant built on
//! `tpt-simd-gather` (f32 only) that lives only in this file.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_core::{I32x8, Simd};
use tpt_simd_gather::gather_f32;
use tpt_simd_sparse::*;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn random_matrix(n: usize, per_row: usize) -> CsrMatrix<f32> {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut t = Vec::with_capacity(n * per_row);
    for r in 0..n {
        for _ in 0..per_row {
            let c = (rng.next() >> 16) as usize % n;
            let v = ((rng.next() >> 40) as f32 / 8_388_608.0) - 1.0;
            t.push((r, c, v));
        }
    }
    CsrMatrix::from_triplets(n, n, &t).unwrap()
}

/// Hardware-gather row kernel (f32): `vgatherdps` + vector multiply-add.
fn spmv_csr_gather(a: &CsrView<'_, f32>, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), a.ncols());
    assert_eq!(y.len(), a.nrows());
    for (r, w) in a.indptr().windows(2).enumerate() {
        let (idx, d) = (&a.indices()[w[0]..w[1]], &a.data()[w[0]..w[1]]);
        let mut acc = Simd::<f32, 8>::zero();
        let ic = idx.chunks_exact(8);
        let dc = d.chunks_exact(8);
        let (it, dt) = (ic.remainder(), dc.remainder());
        for (i, v) in ic.zip(dc) {
            let mut lanes = [0i32; 8];
            for j in 0..8 {
                lanes[j] = i[j] as i32;
            }
            // SAFETY: the view is validated, so every index < x.len().
            let g = unsafe { gather_f32(x.as_ptr(), I32x8::from_array(lanes)) };
            acc += Simd::from_slice(v) * g;
        }
        let l = acc.to_array();
        let mut t = ((l[0] + l[4]) + (l[2] + l[6])) + ((l[1] + l[5]) + (l[3] + l[7]));
        for (&i, &v) in it.iter().zip(dt) {
            t += v * x[i as usize];
        }
        y[r] = t;
    }
}

fn vec_data(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7919 % 2003) as f32 - 1000.0) * 0.001)
        .collect()
}

fn spmv_group(c: &mut Criterion, name: &str, a: &CsrMatrix<f32>) {
    let v = a.view();
    let n = a.ncols();
    let x = vec_data(n);
    let mut y = vec![0.0f32; a.nrows()];
    let csc = a.to_csc();
    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Elements(v.nnz() as u64));
    g.bench_function("reference", |b| {
        b.iter(|| reference::spmv_csr(1.0, &v, black_box(&x), 0.0, &mut y))
    });
    for (label, s) in [
        ("scalar-1acc", RowStrategy::Scalar),
        ("lanes4", RowStrategy::Lanes4),
        ("lanes8", RowStrategy::Lanes8),
        ("hybrid", RowStrategy::Hybrid),
    ] {
        g.bench_function(label, |b| {
            b.iter(|| spmv_csr_with(s, 1.0, &v, black_box(&x), 0.0, &mut y))
        });
    }
    g.bench_function("hw-gather", |b| {
        b.iter(|| spmv_csr_gather(&v, black_box(&x), &mut y))
    });
    g.bench_function("spmv_csr (default)", |b| {
        b.iter(|| spmv_csr(1.0, &v, black_box(&x), 0.0, &mut y))
    });
    g.bench_function("spmv_csr alpha/beta", |b| {
        b.iter(|| spmv_csr(0.5, &v, black_box(&x), 0.5, &mut y))
    });
    g.bench_function("spmv_csc (scatter)", |b| {
        b.iter(|| spmv_csc(1.0, &csc.view(), black_box(&x), 0.0, &mut y))
    });
    g.bench_function("reference csc", |b| {
        b.iter(|| reference::spmv_csc(1.0, &csc.view(), black_box(&x), 0.0, &mut y))
    });
    g.bench_function("spmv_csr_t (scatter)", |b| {
        b.iter(|| spmv_csr_t(1.0, &v, black_box(&x), 0.0, &mut y))
    });
    g.bench_function("reference csr_t", |b| {
        b.iter(|| reference::spmv_csr_t(1.0, &v, black_box(&x), 0.0, &mut y))
    });
    g.bench_function("spmv_csc_t (gather)", |b| {
        b.iter(|| spmv_csc_t(1.0, &csc.view(), black_box(&x), 0.0, &mut y))
    });
    g.finish();
}

fn bench_spmv(c: &mut Criterion) {
    let poisson = CsrMatrix::<f32>::poisson_2d(256);
    spmv_group(c, "spmv f32 poisson 256x256 (5 nnz/row)", &poisson);
    let random = random_matrix(65_536, 32);
    spmv_group(c, "spmv f32 random 65536 (32 nnz/row)", &random);

    // f64 on the same shapes.
    let mut g = c.benchmark_group("spmv f64 poisson 256x256");
    let p = CsrMatrix::<f64>::poisson_2d(256);
    let v = p.view();
    let x: Vec<f64> = vec_data(p.ncols()).iter().map(|&a| f64::from(a)).collect();
    let mut y = vec![0.0f64; p.nrows()];
    g.throughput(Throughput::Elements(v.nnz() as u64));
    g.bench_function("reference", |b| {
        b.iter(|| reference::spmv_csr(1.0, &v, black_box(&x), 0.0, &mut y))
    });
    g.bench_function("spmv_csr", |b| {
        b.iter(|| spmv_csr(1.0, &v, black_box(&x), 0.0, &mut y))
    });
    g.finish();
}

fn bench_vec(c: &mut Criterion) {
    // 65 536 fits L2; 4 Mi elements streams from memory.
    for n in [65_536usize, 4 << 20] {
        let p = vec_data(n);
        let q: Vec<f32> = p.iter().map(|v| v * 0.5 + 0.1).collect();
        let mut x = vec![0.0f32; n];
        let mut r = vec_data(n);
        let mut g = c.benchmark_group(format!("cg vector update f32 n={n}"));
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::new("separate passes (reference)", n), |b| {
            b.iter(|| {
                reference::axpy(1e-6, black_box(&p), &mut x);
                reference::axpy(-1e-6, black_box(&q), &mut r);
                reference::dot(&r, &r)
            })
        });
        g.bench_function(
            BenchmarkId::new("separate passes (axpy,axpy,sqnorm)", n),
            |b| {
                b.iter(|| {
                    axpy(1e-6, black_box(&p), &mut x);
                    axpy(-1e-6, black_box(&q), &mut r);
                    sqnorm(&r)
                })
            },
        );
        g.bench_function(BenchmarkId::new("fused cg_update", n), |b| {
            b.iter(|| cg_update(1e-6, black_box(&p), black_box(&q), &mut x, &mut r))
        });
        g.finish();

        let mut g = c.benchmark_group(format!("dot / axpy+dot f32 n={n}"));
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function("reference dot", |b| {
            b.iter(|| reference::dot(black_box(&p), black_box(&q)))
        });
        g.bench_function("dot", |b| b.iter(|| dot(black_box(&p), black_box(&q))));
        g.bench_function("axpy then dot", |b| {
            b.iter(|| {
                axpy(1e-6, black_box(&p), &mut r);
                dot(&r, black_box(&q))
            })
        });
        g.bench_function("fused axpy_dot", |b| {
            b.iter(|| axpy_dot(1e-6, black_box(&p), &mut r, black_box(&q)))
        });
        g.finish();
    }
}

criterion_group!(benches, bench_spmv, bench_vec);
criterion_main!(benches);
