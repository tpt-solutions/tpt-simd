//! f32 to f16 conversion vs the `half` crate's per-element call (`black_box` inputs).

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use half::f16;
use std::hint::black_box;
use tpt_simd_half::f32_to_f16_slice;

fn bench(c: &mut Criterion) {
    let n = 1 << 16;
    let src: Vec<f32> = (0..n).map(|i| (i as f32 - 32768.0) * 0.37).collect();
    let mut dst = vec![f16::ZERO; n];

    let mut group = c.benchmark_group("f32_to_f16");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("half_per_element", |b| {
        b.iter(|| {
            for (d, &s) in dst.iter_mut().zip(black_box(&src)) {
                *d = f16::from_f32(s);
            }
        })
    });
    group.bench_function("tpt", |b| {
        b.iter(|| f32_to_f16_slice(black_box(&src), black_box(&mut dst)))
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
