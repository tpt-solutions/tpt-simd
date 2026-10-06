//! Window benches: generation and application vs scalar baselines.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_core::F32x8;
use tpt_simd_window::*;

fn bench(c: &mut Criterion) {
    let n = 4096;
    let mut w = vec![0.0f32; n];
    c.bench_function("hamming_into/tpt_cos_approx", |b| {
        b.iter(|| hamming_window_into_f32(black_box(&mut w)))
    });
    c.bench_function("hamming/libm_cosf_scalar", |b| {
        b.iter(|| {
            let d = (black_box(n) - 1) as f32;
            for (i, o) in w.iter_mut().enumerate() {
                *o = 0.54 - 0.46 * libm::cosf(2.0 * core::f32::consts::PI * i as f32 / d);
            }
        })
    });
    c.bench_function("blackman_into", |b| {
        b.iter(|| blackman_window_into_f32(black_box(&mut w)))
    });
    c.bench_function("kaiser_into_beta8", |b| {
        b.iter(|| kaiser_window_into_f32(black_box(&mut w), 8.0))
    });
    c.bench_function("cos_approx_simd", |b| {
        let x = F32x8::from_fn(|i| i as f32 * 0.9);
        b.iter(|| cos_approx_simd_f32(black_box(x)))
    });
    c.bench_function("bessel_i0_f64", |b| {
        b.iter(|| bessel_i0_f64(black_box(8.0)))
    });

    let win = vec![0.5f32; n];
    let mut data = vec![1.0f32; n];
    c.bench_function("apply_window_f32/tpt", |b| {
        b.iter(|| apply_window_f32(black_box(&mut data), black_box(&win)))
    });
    c.bench_function("apply_window_f32/scalar", |b| {
        b.iter(|| apply_window_scalar_f32(black_box(&mut data), black_box(&win)))
    });
    let mut dv = vec![F32x8::splat(1.0); n / 8];
    let wv = vec![F32x8::splat(0.5); n / 8];
    c.bench_function("apply_window_simd_f32", |b| {
        b.iter(|| apply_window_simd_f32(black_box(&mut dv), black_box(&wv)))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
