//! tpt-simd-text hex and UTF-8 validation vs faster-hex and simdutf8 (dev-dep oracles).
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn bytes(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 131 % 256) as u8).collect()
}

fn text(n: usize) -> String {
    "héllo wörld 😀 ".repeat(n / 15 + 1)[..n].to_string()
}

fn bench(c: &mut Criterion) {
    for &n in &[64usize, 4 << 10, 256 << 10] {
        let src = bytes(n);
        let mut enc = vec![0u8; 2 * n];
        let mut g = c.benchmark_group(format!("hex_encode/{n}B"));
        g.throughput(Throughput::Bytes(n as u64));
        g.bench_function(BenchmarkId::new("tpt", n), |b| {
            b.iter(|| tpt_simd_text::hex_encode_lower(black_box(&src), &mut enc).unwrap())
        });
        g.bench_function(BenchmarkId::new("faster-hex", n), |b| {
            b.iter(|| {
                faster_hex::hex_encode(black_box(&src), &mut enc)
                    .unwrap()
                    .len()
            })
        });
        g.finish();

        let hex = {
            let mut h = vec![0u8; 2 * n];
            faster_hex::hex_encode(&src, &mut h).unwrap();
            h
        };
        let mut dec = vec![0u8; n];
        let mut g = c.benchmark_group(format!("hex_decode/{n}B"));
        g.throughput(Throughput::Bytes(n as u64));
        g.bench_function(BenchmarkId::new("tpt", n), |b| {
            b.iter(|| tpt_simd_text::hex_decode(black_box(&hex), &mut dec).unwrap())
        });
        g.bench_function(BenchmarkId::new("faster-hex", n), |b| {
            b.iter(|| {
                faster_hex::hex_decode(black_box(&hex), &mut dec)
                    .unwrap()
                    .len()
            })
        });
        g.finish();

        let t = text(n).into_bytes();
        let mut g = c.benchmark_group(format!("utf8_validate/{n}B"));
        g.throughput(Throughput::Bytes(t.len() as u64));
        g.bench_function(BenchmarkId::new("tpt-core", n), |b| {
            b.iter(|| tpt_simd_text::is_valid_utf8(black_box(&t)))
        });
        g.bench_function(BenchmarkId::new("simdutf8", n), |b| {
            b.iter(|| simdutf8::basic::from_utf8(black_box(&t)).is_ok())
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
