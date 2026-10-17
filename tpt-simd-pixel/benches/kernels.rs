//! Pixel conversions vs plain per-pixel loops (`black_box` inputs).

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_pixel::{histogram_u8, premultiply_rgba8, rgb8_to_rgba8};

fn bench(c: &mut Criterion) {
    let pixels = 1 << 18;
    let rgb: Vec<u8> = (0..pixels * 3).map(|i| (i * 31 % 256) as u8).collect();
    let mut rgba = vec![0u8; pixels * 4];

    let mut group = c.benchmark_group("rgb8_to_rgba8");
    group.throughput(Throughput::Elements(pixels as u64));
    group.bench_function("naive", |b| {
        b.iter(|| {
            for (i, px) in black_box(&rgb).chunks_exact(3).enumerate() {
                rgba[i * 4..i * 4 + 4].copy_from_slice(&[px[0], px[1], px[2], 255]);
            }
        })
    });
    group.bench_function("tpt", |b| {
        b.iter(|| rgb8_to_rgba8(black_box(&rgb), black_box(&mut rgba), 255))
    });
    group.finish();

    let rgba_src: Vec<u8> = (0..pixels * 4).map(|i| (i * 53 % 256) as u8).collect();
    let mut group = c.benchmark_group("premultiply_rgba8");
    group.throughput(Throughput::Elements(pixels as u64));
    group.bench_function("naive", |b| {
        b.iter_batched(
            || black_box(rgba_src.clone()),
            |mut buf| {
                for px in buf.chunks_exact_mut(4) {
                    let a = u32::from(px[3]);
                    for c in &mut px[..3] {
                        let t = u32::from(*c) * a + 128;
                        *c = ((t + (t >> 8)) >> 8) as u8;
                    }
                }
                buf
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("tpt", |b| {
        b.iter_batched(
            || black_box(rgba_src.clone()),
            |mut buf| {
                premultiply_rgba8(&mut buf);
                buf
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();

    let bytes: Vec<u8> = (0..pixels * 3).map(|i| (i * 7 % 256) as u8).collect();
    let mut bins = [0u32; 256];
    let mut group = c.benchmark_group("histogram_u8");
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("naive", |b| {
        b.iter(|| {
            let mut h = [0u32; 256];
            for &x in black_box(&bytes) {
                h[usize::from(x)] += 1;
            }
            h
        })
    });
    group.bench_function("tpt", |b| {
        b.iter(|| {
            histogram_u8(black_box(&bytes), &mut bins);
            bins[0]
        })
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
