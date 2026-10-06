//! Convolution / FIR benches: public (SIMD) path vs scalar reference.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_convolve::*;

fn bench(c: &mut Criterion) {
    let x: Vec<f32> = (0..4096).map(|i| (i % 97) as f32 * 0.01).collect();
    let k: Vec<f32> = (0..31).map(|i| 1.0 / (1 + i) as f32).collect();
    let mut out = vec![0.0f32; convolve_1d_output_len(x.len(), k.len())];
    c.bench_function("convolve_1d_f32/tpt", |b| {
        b.iter(|| convolve_1d_f32(black_box(&x), black_box(&k), &mut out))
    });
    c.bench_function("convolve_1d_f32/scalar", |b| {
        b.iter(|| convolve_1d_scalar_f32(black_box(&x), black_box(&k), &mut out))
    });

    let (w, h) = (128, 128);
    let img: Vec<f32> = (0..w * h).map(|i| (i % 255) as f32).collect();
    let kern = [0.25f32, 0.5, 0.25];
    let (mut o2, mut t2) = (vec![0.0f32; w * h], vec![0.0f32; w * h]);
    c.bench_function("convolve_2d_separable_f32/tpt", |b| {
        b.iter(|| convolve_2d_separable_f32(black_box(&img), w, h, &kern, &kern, &mut o2, &mut t2))
    });
    c.bench_function("convolve_2d_separable_f32/scalar", |b| {
        b.iter(|| {
            convolve_2d_separable_scalar_f32(black_box(&img), w, h, &kern, &kern, &mut o2, &mut t2)
        })
    });

    let xi: Vec<i16> = (0..4096).map(|i| (i * 37 % 20000 - 10000) as i16).collect();
    let ci: Vec<i16> = (0..32).map(|i| 1000 - i * 20).collect();
    let mut oi = vec![0i16; xi.len()];
    c.bench_function("fir_filter_i16/tpt", |b| {
        b.iter(|| fir_filter_i16(black_box(&xi), black_box(&ci), &mut oi))
    });
    c.bench_function("fir_filter_i16/scalar", |b| {
        b.iter(|| fir_filter_scalar_i16(black_box(&xi), black_box(&ci), &mut oi))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
