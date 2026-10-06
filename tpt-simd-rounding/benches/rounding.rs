//! Benchmarks: tpt-simd rounding vs scalar `libm` loops.
#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_core::F32x8;
use tpt_simd_rounding::*;

const N: usize = 4096;

fn data() -> Vec<f32> {
    (0..N as i32)
        .map(|i| (i.wrapping_mul(7919) % 20001 - 10000) as f32 * 0.37)
        .collect()
}

fn bench(c: &mut Criterion) {
    let a = data();
    let mut out = vec![0.0f32; N];
    let mut outi = vec![0i32; N];
    let mut g = c.benchmark_group("rounding");
    g.throughput(Throughput::Elements(N as u64));

    macro_rules! pair {
        ($name:literal, $scalar:expr, $simd:ident) => {
            g.bench_function(concat!($name, "/scalar"), |b| {
                b.iter(|| {
                    for (o, &x) in out.iter_mut().zip(black_box(&a)) {
                        *o = $scalar(x);
                    }
                    black_box(&out);
                })
            });
            g.bench_function(concat!($name, "/simd"), |b| {
                b.iter(|| {
                    for (o, x) in out.chunks_exact_mut(8).zip(black_box(&a).chunks_exact(8)) {
                        $simd(F32x8::from_slice(x)).copy_to_slice(o);
                    }
                    black_box(&out);
                })
            });
        };
    }
    pair!("floor", libm::floorf, floor_f32);
    pair!("ceil", libm::ceilf, ceil_f32);
    pair!("trunc", libm::truncf, trunc_f32);
    pair!("round", libm::roundf, round_f32);
    pair!("round_ties_even", libm::rintf, round_ties_even_f32);

    g.bench_function("round_to_nearest_even_i32/scalar", |b| {
        b.iter(|| {
            for (o, &x) in outi.iter_mut().zip(black_box(&a)) {
                *o = libm::rintf(x) as i32;
            }
            black_box(&outi);
        })
    });
    g.bench_function("round_to_nearest_even_i32/simd", |b| {
        b.iter(|| {
            for (o, x) in outi.chunks_exact_mut(8).zip(black_box(&a).chunks_exact(8)) {
                round_to_nearest_even_i32(F32x8::from_slice(x)).copy_to_slice(o);
            }
            black_box(&outi);
        })
    });
    g.bench_function("round_with_bias_i32/simd", |b| {
        b.iter(|| {
            for (o, x) in outi.chunks_exact_mut(8).zip(black_box(&a).chunks_exact(8)) {
                round_with_bias_i32(F32x8::from_slice(x), 0.5).copy_to_slice(o);
            }
            black_box(&outi);
        })
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
