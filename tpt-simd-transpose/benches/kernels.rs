//! Matrix transpose vs a plain index loop (`black_box` inputs).

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_transpose::transpose;

fn naive(src: &[u32], rows: usize, cols: usize, dst: &mut [u32]) {
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

fn bench(c: &mut Criterion) {
    for n in [64usize, 512] {
        let src: Vec<u32> = (0..(n * n) as u32).collect();
        let mut dst = vec![0u32; n * n];
        let mut group = c.benchmark_group(format!("transpose_u32/{n}x{n}"));
        group.throughput(Throughput::Bytes((n * n * 4) as u64));
        group.bench_function("naive", |b| {
            b.iter(|| naive(black_box(&src), n, n, black_box(&mut dst)))
        });
        group.bench_function("tpt", |b| {
            b.iter(|| transpose(black_box(&src), n, n, black_box(&mut dst)))
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
