//! SoA geometry kernels vs per-point loops over arrays of structs (`black_box` inputs).

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_geom::{BoxesSoa, Soa3, distance_sq, ray_aabb};

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

    // Slab test of one ray against 16K boxes (min and max corners in SoA layout).
    let mins: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            [
                (i % 13) as f32 - 6.0,
                (i % 7) as f32 - 3.0,
                (i % 5) as f32 - 2.0,
            ]
        })
        .collect();
    let maxs: Vec<[f32; 3]> = mins
        .iter()
        .map(|m| [m[0] + 4.0, m[1] + 3.0, m[2] + 2.5])
        .collect();
    let mx: Vec<f32> = mins.iter().map(|m| m[0]).collect();
    let my: Vec<f32> = mins.iter().map(|m| m[1]).collect();
    let mz: Vec<f32> = mins.iter().map(|m| m[2]).collect();
    let xx: Vec<f32> = mx.iter().map(|v| v + 4.0).collect();
    let xy: Vec<f32> = my.iter().map(|v| v + 3.0).collect();
    let xz: Vec<f32> = mz.iter().map(|v| v + 2.5).collect();
    let origin = [0.5f32, 0.25, -0.75];
    let dir = [0.6f32, 0.8, -0.3];
    let inv = [1.0 / dir[0], 1.0 / dir[1], 1.0 / dir[2]];
    let mut group = c.benchmark_group("ray_aabb");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("aos_naive", |b| {
        b.iter(|| {
            // Same slab test per box, over arrays of structs: the full algorithm.
            for (o, (m, x)) in out
                .iter_mut()
                .zip(black_box(&mins).iter().zip(black_box(&maxs)))
            {
                let (mut tn, mut tf) = (f32::NEG_INFINITY, f32::INFINITY);
                for a in 0..3 {
                    let t1 = (m[a] - origin[a]) * inv[a];
                    let t2 = (x[a] - origin[a]) * inv[a];
                    let (lo, hi) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
                    if lo > tn {
                        tn = lo;
                    }
                    if hi < tf {
                        tf = hi;
                    }
                }
                *o = if tn <= tf && tf >= 0.0 && tn <= 100.0 {
                    if tn < 0.0 { 0.0 } else { tn }
                } else {
                    f32::INFINITY
                };
            }
        })
    });
    group.bench_function("soa_tpt", |b| {
        b.iter(|| {
            ray_aabb(
                BoxesSoa::new(
                    Soa3::new(black_box(&mx), black_box(&my), black_box(&mz)),
                    Soa3::new(black_box(&xx), black_box(&xy), black_box(&xz)),
                ),
                origin,
                dir,
                100.0,
                &mut out,
            )
        })
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
