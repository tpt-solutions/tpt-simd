//! tpt-simd-search byte and substring search vs memchr (dev-dep oracle).
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn haystack(n: usize) -> Vec<u8> {
    let mut h = vec![b'a'; n];
    // The needle byte only near the end, so the scan covers the whole buffer.
    if n > 8 {
        h[n - 5] = b'Z';
    }
    h
}

fn bench(c: &mut Criterion) {
    for &n in &[64usize, 4 << 10, 1 << 20] {
        let h = haystack(n);
        let mut g = c.benchmark_group(format!("find_byte/{n}B"));
        g.throughput(Throughput::Bytes(n as u64));
        g.bench_function(BenchmarkId::new("tpt", n), |b| {
            b.iter(|| tpt_simd_search::find_byte(black_box(&h), b'Z'))
        });
        g.bench_function(BenchmarkId::new("memchr", n), |b| {
            b.iter(|| memchr::memchr(black_box(b'Z'), black_box(&h)))
        });
        g.finish();

        let needle = b"aZ";
        let mut g = c.benchmark_group(format!("substring/{n}B"));
        g.throughput(Throughput::Bytes(n as u64));
        g.bench_function(BenchmarkId::new("tpt", n), |b| {
            b.iter(|| tpt_simd_search::find(black_box(&h), black_box(needle)))
        });
        g.bench_function(BenchmarkId::new("memmem", n), |b| {
            b.iter(|| memchr::memmem::find(black_box(&h), black_box(needle)))
        });
        g.finish();

        let mut g = c.benchmark_group(format!("count_byte/{n}B"));
        g.throughput(Throughput::Bytes(n as u64));
        g.bench_function(BenchmarkId::new("tpt", n), |b| {
            b.iter(|| tpt_simd_search::count_byte(black_box(&h), b'a'))
        });
        g.bench_function(BenchmarkId::new("memchr_iter", n), |b| {
            b.iter(|| memchr::memchr_iter(black_box(b'a'), black_box(&h)).count())
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
