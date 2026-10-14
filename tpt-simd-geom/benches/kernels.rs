//! SoA geometry kernels vs per-point loops over arrays of structs (`black_box` inputs).

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_geom::{Soa3, distance_sq};

fn bench(c: &mut Criterion) {
    let n = 1 << 14;
    let xs: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.5).collect();
    let ys: Vec<f32> = (0..n).map(|i| (i % 89) as f32 * 0.25).collect();
    let zs: Vec<f32> = (0..n).map(|i| (i % 83) as f32 * 0.125).collect();
    let aos: Vec<[f32; 3]> = (0..n).map(|i| [xs[i], ys[i], zs[i]]).collect();
    let q = [3.0f32, -2.0, 1.5];
    let mut out = vec![0f32; n];

    let mut group = c.benchmark_group("distance_sq");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("aos_naive", |b| {
        b.iter(|| {
            for (o, p) in out.iter_mut().zip(black_box(&aos)) {
                *o = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2);
            }
        })
    });
    group.bench_function("soa_tpt", |b| {
        b.iter(|| {
            distance_sq(
                Soa3::new(black_box(&xs), black_box(&ys), black_box(&zs)),
                q,
                &mut out,
            )
        })
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
