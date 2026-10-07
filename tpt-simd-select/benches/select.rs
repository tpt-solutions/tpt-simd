//! Index-based lane selection vs scalar array indexing.
//!
//! 4096 elements are processed as 512 vectors of 8 lanes with pseudo-random
//! indices. The scalar baseline is the direct `v[idx & 7]` loop, which LLVM
//! may turn into permutes or leave scalar.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_core::{F32x8, I32x8};
use tpt_simd_select::{
    select_from_slice_i32, select_i32, select_lanes_f32, try_select_from_slice_i32,
};

fn bench(c: &mut Criterion) {
    let n = 4096usize;
    let mut s = 0x1234_5678u32;
    let idx: Vec<i32> = (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 16) as i32
        })
        .collect();
    let vi: Vec<i32> = (0..n).map(|i| i as i32 * 3).collect();
    let vf: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mut oi = vec![0i32; n];
    let mut of = vec![0.0f32; n];

    c.bench_function("select_i32/scalar_idx", |b| {
        b.iter(|| {
            let (v, ix) = (black_box(&vi), black_box(&idx));
            for blk in 0..n / 8 {
                for l in 0..8 {
                    oi[blk * 8 + l] = v[blk * 8 + (ix[blk * 8 + l] & 7) as usize];
                }
            }
            black_box(&oi);
        })
    });
    c.bench_function("select_i32/simd", |b| {
        b.iter(|| {
            for ((o, v), ix) in oi
                .chunks_exact_mut(8)
                .zip(black_box(&vi).chunks_exact(8))
                .zip(black_box(&idx).chunks_exact(8))
            {
                select_i32(I32x8::from_slice(v), I32x8::from_slice(ix)).copy_to_slice(o);
            }
            black_box(&oi);
        })
    });
    c.bench_function("select_lanes_f32/scalar_idx", |b| {
        b.iter(|| {
            let (v, ix) = (black_box(&vf), black_box(&idx));
            for blk in 0..n / 8 {
                for l in 0..8 {
                    of[blk * 8 + l] = v[blk * 8 + (ix[blk * 8 + l] & 7) as usize];
                }
            }
            black_box(&of);
        })
    });
    c.bench_function("select_lanes_f32/simd", |b| {
        b.iter(|| {
            for ((o, v), ix) in of
                .chunks_exact_mut(8)
                .zip(black_box(&vf).chunks_exact(8))
                .zip(black_box(&idx).chunks_exact(8))
            {
                select_lanes_f32(F32x8::from_slice(v), I32x8::from_slice(ix)).copy_to_slice(o);
            }
            black_box(&of);
        })
    });

    // Table lookup: indices in-bounds of a 256-entry table.
    let table: Vec<i32> = (0..256).map(|i| i * 7 - 100).collect();
    let tidx: Vec<i32> = idx.iter().map(|&i| i & 255).collect();
    c.bench_function("select_from_slice_i32/scalar_idx", |b| {
        b.iter(|| {
            let (t, ix) = (black_box(&table), black_box(&tidx));
            for i in 0..n {
                oi[i] = t[ix[i] as usize];
            }
            black_box(&oi);
        })
    });
    c.bench_function("select_from_slice_i32/simd_checked", |b| {
        b.iter(|| {
            for (o, ix) in oi.chunks_exact_mut(8).zip(black_box(&tidx).chunks_exact(8)) {
                select_from_slice_i32(black_box(&table), I32x8::from_slice(ix)).copy_to_slice(o);
            }
            black_box(&oi);
        })
    });
    c.bench_function("select_from_slice_i32/simd_try", |b| {
        b.iter(|| {
            for (o, ix) in oi.chunks_exact_mut(8).zip(black_box(&tidx).chunks_exact(8)) {
                try_select_from_slice_i32(black_box(&table), I32x8::from_slice(ix))
                    .unwrap()
                    .copy_to_slice(o);
            }
            black_box(&oi);
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
