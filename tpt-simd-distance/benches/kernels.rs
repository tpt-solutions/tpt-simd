//! Distance kernels vs plain iterator loops (`black_box` inputs).

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_distance::{inner_product_f32, l2_squared_f32};

fn bench(c: &mut Criterion) {
    let dim = 1024usize;
    let a: Vec<f32> = (0..dim).map(|i| (i % 31) as f32 * 0.1 - 1.5).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i % 29) as f32 * 0.2 - 2.0).collect();

    let mut group = c.benchmark_group("inner_product_f32_1024");
    group.throughput(Throughput::Elements(dim as u64));
    group.bench_function("iter", |bch| {
        bch.iter(|| {
            black_box(&a)
                .iter()
                .zip(black_box(&b))
                .map(|(x, y)| x * y)
                .sum::<f32>()
        })
    });
    group.bench_function("tpt", |bch| {
        bch.iter(|| inner_product_f32(black_box(&a), black_box(&b)))
    });
    group.finish();

    let mut group = c.benchmark_group("l2_squared_f32_1024");
    group.throughput(Throughput::Elements(dim as u64));
    group.bench_function("iter", |bch| {
        bch.iter(|| {
            black_box(&a)
                .iter()
                .zip(black_box(&b))
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
        })
    });
    group.bench_function("tpt", |bch| {
        bch.iter(|| l2_squared_f32(black_box(&a), black_box(&b)))
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
