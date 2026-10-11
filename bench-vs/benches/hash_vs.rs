//! tpt-simd-hash CRC-32 and Adler-32 vs crc32fast and simd-adler32 (dev-dep oracles).
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn data(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 31 % 251) as u8).collect()
}

fn bench(c: &mut Criterion) {
    for &n in &[64usize, 4 << 10, 1 << 20] {
        let d = data(n);
        let mut g = c.benchmark_group(format!("crc32/{n}B"));
        g.throughput(Throughput::Bytes(n as u64));
        g.bench_function(BenchmarkId::new("tpt", n), |b| {
            b.iter(|| tpt_simd_hash::crc32(black_box(&d)))
        });
        g.bench_function(BenchmarkId::new("crc32fast", n), |b| {
            b.iter(|| crc32fast::hash(black_box(&d)))
        });
        g.finish();

        let mut g = c.benchmark_group(format!("adler32/{n}B"));
        g.throughput(Throughput::Bytes(n as u64));
        g.bench_function(BenchmarkId::new("tpt", n), |b| {
            b.iter(|| tpt_simd_hash::adler32(black_box(&d)))
        });
        g.bench_function(BenchmarkId::new("simd-adler32", n), |b| {
            b.iter(|| {
                let mut h = simd_adler32::Adler32::new();
                h.write(black_box(&d));
                h.finish()
            })
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
