//! Byte and substring search vs iterator scans (`black_box` inputs).

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_search::{count_byte, find, find_byte};

fn bench(c: &mut Criterion) {
    // The needle sits near the end, so every call scans the whole haystack.
    let mut hay = vec![b'a'; 1 << 20];
    let last = hay.len() - 1;
    hay[last] = b'z';

    let mut group = c.benchmark_group("find_byte_1MiB");
    group.throughput(Throughput::Bytes(hay.len() as u64));
    group.bench_function("iter", |b| {
        b.iter(|| black_box(&hay).iter().position(|&x| x == b'z'))
    });
    group.bench_function("tpt", |b| b.iter(|| find_byte(black_box(&hay), b'z')));
    group.finish();

    let mut group = c.benchmark_group("count_byte_1MiB");
    group.throughput(Throughput::Bytes(hay.len() as u64));
    group.bench_function("iter", |b| {
        b.iter(|| black_box(&hay).iter().filter(|&&x| x == b'a').count())
    });
    group.bench_function("tpt", |b| b.iter(|| count_byte(black_box(&hay), b'a')));
    group.finish();

    let needle = b"azz";
    let mut group = c.benchmark_group("find_substring_1MiB");
    group.throughput(Throughput::Bytes(hay.len() as u64));
    group.bench_function("windows", |b| {
        b.iter(|| {
            black_box(&hay)
                .windows(needle.len())
                .position(|w| w == needle)
        })
    });
    group.bench_function("tpt", |b| {
        b.iter(|| find(black_box(&hay), black_box(needle)))
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
