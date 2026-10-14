//! Audio kernels vs plain loops: FIR and mix (`black_box` inputs).

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_audio::{fir, mix_into};

fn bench(c: &mut Criterion) {
    let n = 1 << 16;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 13 % 201) as f32) / 100.0 - 1.0)
        .collect();
    let taps: Vec<f32> = (0..32).map(|k| 1.0 / (k as f32 + 1.0)).collect();
    let mut y = vec![0f32; n];

    let mut group = c.benchmark_group("fir_32_taps");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("naive", |b| {
        b.iter(|| {
            let (taps, x) = (black_box(&taps), black_box(&x));
            for (i, o) in y.iter_mut().enumerate() {
                let mut acc = 0f32;
                for (k, &t) in taps.iter().enumerate().take(i + 1) {
                    acc += t * x[i - k];
                }
                *o = acc;
            }
        })
    });
    group.bench_function("tpt", |b| {
        b.iter(|| fir(black_box(&taps), black_box(&x), &mut y))
    });
    group.finish();

    let src: Vec<f32> = x.iter().map(|v| v * 0.5).collect();
    let mut dst = x.clone();
    let mut group = c.benchmark_group("mix_into");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("naive", |b| {
        b.iter(|| {
            for (d, s) in dst.iter_mut().zip(black_box(&src)) {
                *d += 0.75 * s;
            }
        })
    });
    group.bench_function("tpt", |b| {
        b.iter(|| mix_into(&mut dst, black_box(&src), 0.75))
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
