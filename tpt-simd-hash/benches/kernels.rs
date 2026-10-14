//! Checksums vs plain byte loops (`black_box` inputs).

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_hash::{adler32, crc32};

fn naive_adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + u32::from(x)) % 65_521;
        b = (b + a) % 65_521;
    }
    (b << 16) | a
}

fn bench(c: &mut Criterion) {
    let data: Vec<u8> = (0..1 << 20).map(|i| (i * 13 % 251) as u8).collect();
    let mut group = c.benchmark_group("adler32_1MiB");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("naive", |b| b.iter(|| naive_adler32(black_box(&data))));
    group.bench_function("tpt", |b| b.iter(|| adler32(black_box(&data))));
    group.finish();

    let mut group = c.benchmark_group("crc32_1MiB");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("tpt", |b| b.iter(|| crc32(black_box(&data))));
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
