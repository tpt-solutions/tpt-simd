//! Benchmarks: tpt-simd-permute vs plain scalar loops.
#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_permute::*;

const N: usize = 4096;

fn scalar_t8(m: &mut [[i16; 8]; 8]) {
    let src = *m;
    for i in 0..8 {
        for j in 0..8 {
            m[i][j] = src[j][i];
        }
    }
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("permute");

    let mut m: [[i16; 8]; 8] = std::array::from_fn(|r| std::array::from_fn(|c| (r * 8 + c) as i16));
    g.throughput(Throughput::Elements(64));
    g.bench_function("transpose_8x8_i16/scalar", |b| {
        b.iter(|| scalar_t8(black_box(&mut m)))
    });
    g.bench_function("transpose_8x8_i16/portable_swap", |b| {
        b.iter(|| transpose_8x8_i16_portable(black_box(&mut m)))
    });
    g.bench_function("transpose_8x8_i16/simd", |b| {
        b.iter(|| transpose_8x8_i16(black_box(&mut m)))
    });

    let mut f: [[f32; 4]; 4] = std::array::from_fn(|r| std::array::from_fn(|c| (r * 4 + c) as f32));
    g.throughput(Throughput::Elements(16));
    g.bench_function("transpose_4x4_f32/scalar", |b| {
        b.iter(|| transpose_4x4_f32_portable(black_box(&mut f)))
    });
    g.bench_function("transpose_4x4_f32/simd", |b| {
        b.iter(|| transpose_4x4_f32(black_box(&mut f)))
    });

    let l: Vec<i16> = (0..N as i32).map(|i| i as i16).collect();
    let r: Vec<i16> = l.iter().map(|x| !x).collect();
    let mut o = vec![0i16; 2 * N];
    g.throughput(Throughput::Elements(2 * N as u64));
    g.bench_function("interleave_stereo_i16/scalar_index", |b| {
        b.iter(|| {
            let (l, r) = (black_box(&l), black_box(&r));
            for i in 0..N {
                o[2 * i] = l[i];
                o[2 * i + 1] = r[i];
            }
            black_box(&o);
        })
    });
    g.bench_function("interleave_stereo_i16/lib", |b| {
        b.iter(|| {
            interleave_stereo_i16(black_box(&l), black_box(&r), &mut o);
            black_box(&o);
        })
    });
    let (mut l2, mut r2) = (vec![0i16; N], vec![0i16; N]);
    g.bench_function("deinterleave_stereo_i16/lib", |b| {
        b.iter(|| {
            deinterleave_stereo_i16(black_box(&o), &mut l2, &mut r2);
            black_box((&l2, &r2));
        })
    });

    let src: Vec<i8> = (0..N).map(|i| i as i8).collect();
    let mut dst = vec![0i16; N];
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("unpack_i8_to_i16/lib", |b| {
        b.iter(|| {
            unpack_i8_to_i16(black_box(&src), &mut dst);
            black_box(&dst);
        })
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
