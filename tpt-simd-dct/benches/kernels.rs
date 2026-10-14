//! DCT-II forward vs a direct matrix-vector product with the same cosine table (`black_box` inputs).

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_dct::Dct;

fn bench(c: &mut Criterion) {
    let n = 256usize;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 101) as f32) / 50.0 - 1.0)
        .collect();
    let table: Vec<f32> = (0..n * n)
        .map(|idx| {
            let (k, i) = (idx / n, idx % n);
            let s = if k == 0 {
                (1.0 / n as f64).sqrt()
            } else {
                (2.0 / n as f64).sqrt()
            };
            (s * (std::f64::consts::PI / n as f64 * (i as f64 + 0.5) * k as f64).cos()) as f32
        })
        .collect();
    let dct = Dct::new(n);
    let mut out = vec![0f32; n];

    let mut group = c.benchmark_group("dct2_forward_256");
    group.throughput(Throughput::Elements((n * n) as u64));
    group.bench_function("table_matvec", |b| {
        b.iter(|| {
            for (k, o) in out.iter_mut().enumerate() {
                let row = &table[k * n..(k + 1) * n];
                *o = black_box(row)
                    .iter()
                    .zip(black_box(&x))
                    .map(|(a, b)| a * b)
                    .sum();
            }
        })
    });
    group.bench_function("tpt", |b| b.iter(|| dct.forward(black_box(&x), &mut out)));
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
