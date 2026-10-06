//! Dot products vs plain scalar loops (`black_box` inputs).
//!
//! `scalar` is a naive iterator/loop (LLVM may auto-vectorise it; for the
//! f32 case it cannot, because FP reassociation is not allowed);
//! `portable` is the crate's reference path; `tpt` is the public function
//! (AVX2 fast path when built with `-C target-cpu=native`).

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_core::{ComplexSimd, F32x8};
use tpt_simd_dot::*;

fn scalar_i16(a: &[i16], b: &[i16]) -> i32 {
    let mut s = 0i32;
    for i in 0..a.len() {
        s = s.wrapping_add(a[i] as i32 * b[i] as i32);
    }
    s
}
fn scalar_sat_i16(a: &[i16], b: &[i16]) -> i32 {
    let mut s = 0i32;
    for i in 0..a.len() {
        s = s.saturating_add(a[i] as i32 * b[i] as i32);
    }
    s
}
fn scalar_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}
fn scalar_complex(a: &[ComplexSimd<f32, 8>], b: &[ComplexSimd<f32, 8>]) -> ([f32; 8], [f32; 8]) {
    let mut re = [0.0f32; 8];
    let mut im = [0.0f32; 8];
    for i in 0..a.len() {
        for j in 0..8 {
            let (ar, ai) = (a[i].real[j], a[i].imag[j]);
            let (br, bi) = (b[i].real[j], b[i].imag[j]);
            re[j] += ar * br - ai * bi;
            im[j] += ar * bi + ai * br;
        }
    }
    (re, im)
}

fn bench(c: &mut Criterion) {
    for &n in &[256usize, 4096] {
        let a: Vec<i16> = (0..n).map(|i| ((i * 31) % 2001) as i16 - 1000).collect();
        let b: Vec<i16> = (0..n).map(|i| ((i * 17) % 1999) as i16 - 1000).collect();
        let mut g = c.benchmark_group(format!("i16_{n}"));
        g.bench_function("scalar", |t| {
            t.iter(|| scalar_i16(black_box(&a), black_box(&b)))
        });
        g.bench_function("portable", |t| {
            t.iter(|| portable::dot_product_i16(black_box(&a), black_box(&b)))
        });
        g.bench_function("tpt", |t| {
            t.iter(|| dot_product_i16(black_box(&a), black_box(&b)))
        });
        g.finish();

        let mut g = c.benchmark_group(format!("sat_i16_{n}"));
        g.bench_function("scalar", |t| {
            t.iter(|| scalar_sat_i16(black_box(&a), black_box(&b)))
        });
        g.bench_function("tpt", |t| {
            t.iter(|| dot_product_saturating_i16(black_box(&a), black_box(&b)))
        });
        g.finish();

        let fa: Vec<f32> = a.iter().map(|&x| x as f32 * 0.001).collect();
        let fb: Vec<f32> = b.iter().map(|&x| x as f32 * 0.001).collect();
        let mut g = c.benchmark_group(format!("f32_{n}"));
        g.bench_function("scalar", |t| {
            t.iter(|| scalar_f32(black_box(&fa), black_box(&fb)))
        });
        g.bench_function("portable", |t| {
            t.iter(|| portable::dot_product_f32(black_box(&fa), black_box(&fb)))
        });
        g.bench_function("tpt", |t| {
            t.iter(|| dot_product_f32(black_box(&fa), black_box(&fb)))
        });
        g.finish();
    }

    let m = 64;
    let ca: Vec<ComplexSimd<f32, 8>> = (0..m)
        .map(|i| ComplexSimd::new(F32x8::from_fn(|j| (i + j) as f32 * 0.01), F32x8::splat(0.5)))
        .collect();
    let cb: Vec<ComplexSimd<f32, 8>> = (0..m)
        .map(|i| {
            ComplexSimd::new(
                F32x8::splat(0.25),
                F32x8::from_fn(|j| (i * j) as f32 * 0.001),
            )
        })
        .collect();
    let mut g = c.benchmark_group("complex_64x8");
    g.bench_function("scalar", |t| {
        t.iter(|| scalar_complex(black_box(&ca), black_box(&cb)))
    });
    g.bench_function("tpt", |t| {
        t.iter(|| dot_product_complex_f32(black_box(&ca), black_box(&cb)))
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
