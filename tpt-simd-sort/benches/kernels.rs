//! Sorting kernels vs `std` (`black_box` inputs). Each iteration sorts a fresh copy.

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_sort::{sort_u32, sort_unstable};

fn input(n: usize) -> Vec<u32> {
    (0..n as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) ^ (i >> 3))
        .collect()
}

fn bench(c: &mut Criterion) {
    for n in [1_000usize, 100_000] {
        let base = input(n);
        let mut group = c.benchmark_group(format!("sort_u32/{n}"));
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function("std", |b| {
            b.iter_batched(
                || base.clone(),
                |mut v| {
                    v.sort_unstable();
                    black_box(v)
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function("tpt", |b| {
            b.iter_batched(
                || base.clone(),
                |mut v| {
                    let mut scratch = vec![0u32; v.len()];
                    sort_u32(&mut v, &mut scratch);
                    black_box(v)
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function("generic", |b| {
            b.iter_batched(
                || base.clone(),
                |mut v| {
                    sort_unstable(&mut v);
                    black_box(v)
                },
                BatchSize::SmallInput,
            )
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
