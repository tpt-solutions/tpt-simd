//! Interpolation benches: 8-lane SIMD kernels vs eight scalar calls.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_core::F32x8;
use tpt_simd_interpolate::*;

fn bench(c: &mut Criterion) {
    let ramp = |o: f32| F32x8::from_fn(|i| i as f32 * 0.7 + o);
    let (a, b, cc, d, t) = (
        ramp(0.0),
        ramp(1.0),
        ramp(2.0),
        ramp(3.0),
        F32x8::splat(0.37),
    );

    c.bench_function("linear/simd", |bn| {
        bn.iter(|| interpolate_linear_simd_f32(black_box(a), black_box(b), black_box(t)))
    });
    c.bench_function("linear/scalar_x8", |bn| {
        bn.iter(|| {
            let (a, b, t) = (black_box(a), black_box(b), black_box(t));
            F32x8::from_fn(|i| interpolate_linear_f32(a[i], b[i], t[i]))
        })
    });
    c.bench_function("cubic/simd", |bn| {
        bn.iter(|| {
            interpolate_cubic_f32(
                black_box(a),
                black_box(b),
                black_box(cc),
                black_box(d),
                black_box(t),
            )
        })
    });
    c.bench_function("cubic/scalar_x8", |bn| {
        bn.iter(|| {
            let (a, b, c2, d, t) = (
                black_box(a),
                black_box(b),
                black_box(cc),
                black_box(d),
                black_box(t),
            );
            F32x8::from_fn(|i| interpolate_cubic_scalar_f32(a[i], b[i], c2[i], d[i], t[i]))
        })
    });
    let samples: Vec<F32x8> = (0..6).map(|i| ramp(i as f32)).collect();
    c.bench_function("lanczos3/simd", |bn| {
        bn.iter(|| interpolate_lanczos_f32(black_box(&samples), black_box(t), 3))
    });
    c.bench_function("lanczos3/scalar_x8", |bn| {
        bn.iter(|| {
            let t = black_box(t);
            F32x8::from_fn(|l| {
                let col: [f32; 6] = core::array::from_fn(|i| samples[i][l]);
                interpolate_lanczos_scalar_f32(&col, t[l], 3)
            })
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
