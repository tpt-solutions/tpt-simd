//! Aligned vs unaligned load/store wrappers, and aligned allocation.
//!
//! * `copy_scale`: `out = in * 2` over 4096 `f32`: scalar iterator loop
//!   (auto-vectorised), unchecked `from_slice`/`copy_to_slice`, and the
//!   checked `load_aligned`/`store_aligned` wrappers on an aligned buffer.
//! * `alloc`: [`AlignedBuf`] vs `Vec` for zeroed/filled/from-slice buffers.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_aligned::{AlignedBuf, load_aligned, store_aligned, try_load_aligned};
use tpt_simd_core::F32x8;

fn bench(c: &mut Criterion) {
    let n = 4096usize;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let src = AlignedBuf::<f32, 32>::from_slice(&data);
    let mut dst = AlignedBuf::<f32, 32>::zeroed(n);
    let two = F32x8::splat(2.0);

    c.bench_function("copy_scale/scalar_iter", |b| {
        b.iter(|| {
            for (o, i) in dst.iter_mut().zip(black_box(&src).iter()) {
                *o = *i * 2.0;
            }
            black_box(&dst);
        })
    });
    c.bench_function("copy_scale/unaligned_from_slice", |b| {
        b.iter(|| {
            for (o, i) in dst.chunks_exact_mut(8).zip(black_box(&src).chunks_exact(8)) {
                (F32x8::from_slice(i) * two).copy_to_slice(o);
            }
            black_box(&dst);
        })
    });
    c.bench_function("copy_scale/load_store_aligned", |b| {
        b.iter(|| {
            for (o, i) in dst.chunks_exact_mut(8).zip(black_box(&src).chunks_exact(8)) {
                store_aligned(load_aligned::<f32, 8>(i) * two, o);
            }
            black_box(&dst);
        })
    });
    c.bench_function("copy_scale/try_load_aligned", |b| {
        b.iter(|| {
            for (o, i) in dst.chunks_exact_mut(8).zip(black_box(&src).chunks_exact(8)) {
                let v = try_load_aligned::<f32, 8>(i).unwrap();
                (v * two).copy_to_slice(o);
            }
            black_box(&dst);
        })
    });

    c.bench_function("alloc/vec_zeroed_4096", |b| {
        b.iter(|| black_box(vec![0.0f32; black_box(n)]))
    });
    c.bench_function("alloc/aligned_zeroed_4096", |b| {
        b.iter(|| black_box(AlignedBuf::<f32, 32>::zeroed(black_box(n))))
    });
    c.bench_function("alloc/vec_filled_4096", |b| {
        b.iter(|| black_box(vec![1.5f32; black_box(n)]))
    });
    c.bench_function("alloc/aligned_filled_4096", |b| {
        b.iter(|| black_box(AlignedBuf::<f32, 32>::filled(black_box(n), 1.5)))
    });
    c.bench_function("alloc/vec_from_slice_4096", |b| {
        b.iter(|| black_box(black_box(&data).to_vec()))
    });
    c.bench_function("alloc/aligned_from_slice_4096", |b| {
        b.iter(|| black_box(AlignedBuf::<f32, 32>::from_slice(black_box(&data))))
    });
    c.bench_function("alloc/vec_zeroed_8", |b| {
        b.iter(|| black_box(vec![0.0f32; black_box(8)]))
    });
    c.bench_function("alloc/aligned_zeroed_8", |b| {
        b.iter(|| black_box(AlignedBuf::<f32, 32>::zeroed(black_box(8))))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
