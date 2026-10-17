//! Bit kernels vs plain loops (`black_box` inputs).

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_bits::{popcount_u64, prefix_sum_u32};

fn bench(c: &mut Criterion) {
    let words: Vec<u64> = (0..4096u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    let mut group = c.benchmark_group("popcount_u64");
    group.throughput(Throughput::Bytes((words.len() * 8) as u64));
    group.bench_function("naive", |b| {
        b.iter(|| {
            black_box(&words)
                .iter()
                .map(|w| u64::from(w.count_ones()))
                .sum::<u64>()
        })
    });
    group.bench_function("tpt", |b| b.iter(|| popcount_u64(black_box(&words))));
    group.finish();

    let base: Vec<u32> = (0..4096u32).map(|i| i ^ 0x5A5A).collect();
    let mut group = c.benchmark_group("prefix_sum_u32");
    group.throughput(Throughput::Elements(base.len() as u64));
    // The copy is in the setup closure, so only the scan is timed.
    group.bench_function("naive", |b| {
        b.iter_batched(
            || black_box(base.clone()),
            |mut v| {
                let mut acc = 0u32;
                for x in v.iter_mut() {
                    acc = acc.wrapping_add(*x);
                    *x = acc;
                }
                v
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("tpt", |b| {
        b.iter_batched(
            || black_box(base.clone()),
            |mut v| {
                prefix_sum_u32(&mut v);
                v
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
