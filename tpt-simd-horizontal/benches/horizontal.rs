//! Horizontal reductions vs plain scalar loops.
//!
//! "single" benches reduce one vector (kept in a register via `black_box`);
//! "x1024" benches reduce 1024 independent vectors (the realistic case).

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_core::{F32x8, I16x16, I32x8};
use tpt_simd_horizontal::*;

fn scalar_sum_f32(a: &[f32; 8]) -> f32 {
    let mut s = 0.0;
    for &x in a {
        s += x;
    }
    s
}
fn scalar_sum_i32(a: &[i32; 8]) -> i32 {
    a.iter().fold(0i32, |s, &x| s.wrapping_add(x))
}
fn scalar_sum_i16(a: &[i16; 16]) -> i32 {
    a.iter().map(|&x| x as i32).sum()
}
fn scalar_max_i16(a: &[i16; 16]) -> i16 {
    a.iter().copied().fold(i16::MIN, i16::max)
}
fn scalar_min_f32(a: &[f32; 8]) -> f32 {
    a.iter().copied().fold(f32::NAN, f32::min)
}
fn scalar_prod_f32(a: &[f32; 8]) -> f32 {
    a.iter().product()
}

fn bench(c: &mut Criterion) {
    let f: [f32; 8] = core::array::from_fn(|i| i as f32 * 0.5 + 1.0);
    let i32a: [i32; 8] = core::array::from_fn(|i| i as i32 * 1000 - 3000);
    let i16a: [i16; 16] = core::array::from_fn(|i| i as i16 * 1000 - 8000);

    let mut g = c.benchmark_group("single");
    g.bench_function("sum_f32/scalar", |b| {
        b.iter(|| scalar_sum_f32(black_box(&f)))
    });
    g.bench_function("sum_f32/tpt", |b| {
        b.iter(|| horizontal_sum_f32(black_box(F32x8::from_array(f))))
    });
    g.bench_function("sum_i32/scalar", |b| {
        b.iter(|| scalar_sum_i32(black_box(&i32a)))
    });
    g.bench_function("sum_i32/tpt", |b| {
        b.iter(|| horizontal_sum_i32(black_box(I32x8::from_array(i32a))))
    });
    g.bench_function("sum_i16/scalar", |b| {
        b.iter(|| scalar_sum_i16(black_box(&i16a)))
    });
    g.bench_function("sum_i16/tpt", |b| {
        b.iter(|| horizontal_sum_i16(black_box(I16x16::from_array(i16a))))
    });
    g.bench_function("max_i16/scalar", |b| {
        b.iter(|| scalar_max_i16(black_box(&i16a)))
    });
    g.bench_function("max_i16/tpt", |b| {
        b.iter(|| horizontal_max_i16(black_box(I16x16::from_array(i16a))))
    });
    g.bench_function("min_f32/scalar", |b| {
        b.iter(|| scalar_min_f32(black_box(&f)))
    });
    g.bench_function("min_f32/tpt", |b| {
        b.iter(|| horizontal_min_f32(black_box(F32x8::from_array(f))))
    });
    g.bench_function("product_f32/scalar", |b| {
        b.iter(|| scalar_prod_f32(black_box(&f)))
    });
    g.bench_function("product_f32/tpt", |b| {
        b.iter(|| horizontal_product_f32(black_box(F32x8::from_array(f))))
    });
    g.finish();

    const M: usize = 1024;
    let fv: Vec<F32x8> = (0..M)
        .map(|k| F32x8::from_fn(|i| (k * 8 + i) as f32 * 0.01))
        .collect();
    let fa: Vec<[f32; 8]> = fv.iter().map(|v| v.to_array()).collect();
    let iv: Vec<I16x16> = (0..M)
        .map(|k| I16x16::from_fn(|i| ((k * 16 + i) % 2000) as i16 - 1000))
        .collect();
    let ia: Vec<[i16; 16]> = iv.iter().map(|v| v.to_array()).collect();

    let mut g = c.benchmark_group("x1024");
    g.bench_function("sum_f32/scalar", |b| {
        b.iter(|| {
            let mut out = 0.0f32;
            for a in black_box(&fa) {
                out += scalar_sum_f32(a);
            }
            out
        })
    });
    g.bench_function("sum_f32/tpt", |b| {
        b.iter(|| {
            let mut out = 0.0f32;
            for v in black_box(&fv) {
                out += horizontal_sum_f32(*v);
            }
            out
        })
    });
    g.bench_function("sum_i16/scalar", |b| {
        b.iter(|| {
            let mut out = 0i32;
            for a in black_box(&ia) {
                out = out.wrapping_add(scalar_sum_i16(a));
            }
            out
        })
    });
    g.bench_function("sum_i16/tpt", |b| {
        b.iter(|| {
            let mut out = 0i32;
            for v in black_box(&iv) {
                out = out.wrapping_add(horizontal_sum_i16(*v));
            }
            out
        })
    });
    g.bench_function("max_i16/scalar", |b| {
        b.iter(|| {
            let mut out = i16::MIN;
            for a in black_box(&ia) {
                out = out.max(scalar_max_i16(a));
            }
            out
        })
    });
    g.bench_function("max_i16/tpt", |b| {
        b.iter(|| {
            let mut out = i16::MIN;
            for v in black_box(&iv) {
                out = out.max(horizontal_max_i16(*v));
            }
            out
        })
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
