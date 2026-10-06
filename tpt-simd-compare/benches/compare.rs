//! Mask-producing compares vs scalar loops that count matches.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_compare::{cmp_gt_f32, cmp_gt_i32, mask_count};
use tpt_simd_core::{F32x8, I32x8};

fn bench(c: &mut Criterion) {
    let n = 4096;
    let ia: std::vec::Vec<i32> = (0..n).map(|i| (i * 2654435761u32 as i32) >> 7).collect();
    let fa: std::vec::Vec<f32> = ia.iter().map(|&x| x as f32).collect();

    c.bench_function("count_gt_i32/scalar", |b| {
        b.iter(|| black_box(&ia).iter().filter(|&&x| x > 0).count())
    });
    c.bench_function("count_gt_i32/tpt", |b| {
        b.iter(|| {
            let z = I32x8::splat(0);
            black_box(&ia)
                .chunks_exact(8)
                .map(|ch| mask_count(cmp_gt_i32(I32x8::from_slice(ch), z)))
                .sum::<usize>()
        })
    });
    c.bench_function("count_gt_f32/scalar", |b| {
        b.iter(|| black_box(&fa).iter().filter(|&&x| x > 0.0).count())
    });
    c.bench_function("count_gt_f32/tpt", |b| {
        b.iter(|| {
            let z = F32x8::splat(0.0);
            black_box(&fa)
                .chunks_exact(8)
                .map(|ch| mask_count(cmp_gt_f32(F32x8::from_slice(ch), z)))
                .sum::<usize>()
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
