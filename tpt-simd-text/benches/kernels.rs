//! UTF-8 validation and hex encoding vs `std` (`black_box` inputs).

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_text::{hex_encode_lower, is_valid_utf8};

fn bench(c: &mut Criterion) {
    let text: String = "The quick brown fox jumps over the lazy dog. Grüße, 世界! ".repeat(20_000);
    let bytes = text.as_bytes();

    let mut group = c.benchmark_group("utf8_validate");
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("std", |b| {
        b.iter(|| std::str::from_utf8(black_box(bytes)).is_ok())
    });
    group.bench_function("tpt", |b| b.iter(|| is_valid_utf8(black_box(bytes))));
    group.finish();

    let raw: Vec<u8> = (0..1 << 18).map(|i| (i * 17 % 256) as u8).collect();
    let mut out = vec![0u8; raw.len() * 2];
    let mut group = c.benchmark_group("hex_encode_lower");
    group.throughput(Throughput::Bytes(raw.len() as u64));
    group.bench_function("format", |b| {
        b.iter(|| {
            let s: String = black_box(&raw).iter().map(|x| format!("{x:02x}")).collect();
            s.len()
        })
    });
    group.bench_function("tpt", |b| {
        b.iter(|| hex_encode_lower(black_box(&raw), black_box(&mut out)).expect("sized"))
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
