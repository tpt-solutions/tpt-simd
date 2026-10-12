//! tpt-simd-sort sort, select_nth and argsort vs std (sort_unstable, select_nth_unstable, sort_by).
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn random(n: usize) -> Vec<u32> {
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u32
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    for &n in &[1_000usize, 100_000, 1_000_000] {
        let v = random(n);
        let mut g = c.benchmark_group(format!("sort_unstable/{n}"));
        g.bench_function(BenchmarkId::new("tpt", n), |b| {
            b.iter_batched(
                || v.clone(),
                |mut w| tpt_simd_sort::sort_unstable(black_box(&mut w)),
                BatchSize::LargeInput,
            )
        });
        g.bench_function(BenchmarkId::new("std", n), |b| {
            b.iter_batched(
                || v.clone(),
                |mut w| black_box(&mut w).sort_unstable(),
                BatchSize::LargeInput,
            )
        });
        g.finish();

        let mut g = c.benchmark_group(format!("sort_u32_radix/{n}"));
        let mut scratch = vec![0u32; n];
        g.bench_function(BenchmarkId::new("tpt_radix", n), |b| {
            b.iter_batched(
                || v.clone(),
                |mut w| tpt_simd_sort::sort_u32(black_box(&mut w), &mut scratch),
                BatchSize::LargeInput,
            )
        });
        g.bench_function(BenchmarkId::new("std", n), |b| {
            b.iter_batched(
                || v.clone(),
                |mut w| black_box(&mut w).sort_unstable(),
                BatchSize::LargeInput,
            )
        });
        g.finish();

        let k = n / 10;
        let mut g = c.benchmark_group(format!("select_nth/{n}"));
        g.bench_function(BenchmarkId::new("tpt", n), |b| {
            b.iter_batched(
                || v.clone(),
                |mut w| tpt_simd_sort::select_nth(black_box(&mut w), k),
                BatchSize::LargeInput,
            )
        });
        g.bench_function(BenchmarkId::new("std", n), |b| {
            b.iter_batched(
                || v.clone(),
                |mut w| {
                    black_box(w.select_nth_unstable(k));
                },
                BatchSize::LargeInput,
            )
        });
        g.finish();

        let mut idx = vec![0usize; n];
        let mut g = c.benchmark_group(format!("argsort/{n}"));
        g.bench_function(BenchmarkId::new("tpt", n), |b| {
            b.iter(|| tpt_simd_sort::argsort(black_box(&v), &mut idx))
        });
        g.bench_function(BenchmarkId::new("std", n), |b| {
            b.iter_batched(
                || (0..n).collect::<Vec<usize>>(),
                |mut ix| black_box(&mut ix).sort_unstable_by_key(|&i| v[i]),
                BatchSize::LargeInput,
            )
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
