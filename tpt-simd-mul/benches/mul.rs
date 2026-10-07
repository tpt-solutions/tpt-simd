//! Multiply variants vs scalar and (where fair) zip-iterator baselines.
//!
//! Each SIMD loop processes a 4096-element slice in vector-sized chunks.
//! `*_scalar_idx` is an indexed loop; `*_scalar_iter` is a zip/iterator loop
//! that LLVM can auto-vectorise.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_core::{ComplexSimd, F32x8, I16x16, Simd};
use tpt_simd_mul::{
    complex_mul_f32, complex_mul_f32_portable, mul_add_sub_f32, mul_hi_i16, mul_lo_i32,
    mul_widen_i16,
};

fn bench(c: &mut Criterion) {
    let n = 4096usize;
    let a16: Vec<i16> = (0..n).map(|i| (i as i32 * 37 - 9000) as i16).collect();
    let b16: Vec<i16> = (0..n).map(|i| (i as i32 * 91 - 20000) as i16).collect();
    let mut o16 = vec![0i16; n];

    c.bench_function("mul_hi_i16/scalar_idx", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&a16), black_box(&b16));
            for i in 0..n {
                o16[i] = ((i32::from(a[i]) * i32::from(b[i])) >> 16) as i16;
            }
            black_box(&o16);
        })
    });
    c.bench_function("mul_hi_i16/scalar_iter", |bn| {
        bn.iter(|| {
            for ((o, &x), &y) in o16.iter_mut().zip(black_box(&a16)).zip(black_box(&b16)) {
                *o = ((i32::from(x) * i32::from(y)) >> 16) as i16;
            }
            black_box(&o16);
        })
    });
    c.bench_function("mul_hi_i16/simd", |bn| {
        bn.iter(|| {
            for ((o, x), y) in o16
                .chunks_exact_mut(16)
                .zip(black_box(&a16).chunks_exact(16))
                .zip(black_box(&b16).chunks_exact(16))
            {
                mul_hi_i16(I16x16::from_slice(x), I16x16::from_slice(y)).copy_to_slice(o);
            }
            black_box(&o16);
        })
    });

    let mut o32 = vec![0i32; n];
    c.bench_function("mul_widen_i16/scalar_iter", |bn| {
        bn.iter(|| {
            for ((o, &x), &y) in o32.iter_mut().zip(black_box(&a16)).zip(black_box(&b16)) {
                *o = i32::from(x) * i32::from(y);
            }
            black_box(&o32);
        })
    });
    c.bench_function("mul_widen_i16/simd", |bn| {
        bn.iter(|| {
            for ((o, x), y) in o32
                .chunks_exact_mut(8)
                .zip(black_box(&a16).chunks_exact(8))
                .zip(black_box(&b16).chunks_exact(8))
            {
                mul_widen_i16(Simd::<i16, 8>::from_slice(x), Simd::<i16, 8>::from_slice(y))
                    .copy_to_slice(o);
            }
            black_box(&o32);
        })
    });

    let a32: Vec<i32> = (0..n)
        .map(|i| (i as i32).wrapping_mul(0x9E37_79B1u32 as i32))
        .collect();
    let b32: Vec<i32> = (0..n)
        .map(|i| (i as i32).wrapping_mul(40503) ^ 0x5555)
        .collect();
    c.bench_function("mul_lo_i32/scalar_iter", |bn| {
        bn.iter(|| {
            for ((o, &x), &y) in o32.iter_mut().zip(black_box(&a32)).zip(black_box(&b32)) {
                *o = x.wrapping_mul(y);
            }
            black_box(&o32);
        })
    });
    c.bench_function("mul_lo_i32/simd", |bn| {
        bn.iter(|| {
            for ((o, x), y) in o32
                .chunks_exact_mut(8)
                .zip(black_box(&a32).chunks_exact(8))
                .zip(black_box(&b32).chunks_exact(8))
            {
                mul_lo_i32(Simd::<i32, 8>::from_slice(x), Simd::<i32, 8>::from_slice(y))
                    .copy_to_slice(o);
            }
            black_box(&o32);
        })
    });

    let af: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let bf: Vec<f32> = (0..n).map(|i| 1.0 - i as f32 * 0.001).collect();
    let cf: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let mut of = vec![0.0f32; n];
    c.bench_function("mul_add_sub_f32/scalar_idx", |bn| {
        bn.iter(|| {
            let (a, b, cc) = (black_box(&af), black_box(&bf), black_box(&cf));
            for i in 0..n {
                let s = if i % 2 == 0 { 1.0 } else { -1.0 };
                of[i] = a[i].mul_add(b[i], cc[i] * s);
            }
            black_box(&of);
        })
    });
    c.bench_function("mul_add_sub_f32/simd", |bn| {
        bn.iter(|| {
            for (((o, x), y), z) in of
                .chunks_exact_mut(8)
                .zip(black_box(&af).chunks_exact(8))
                .zip(black_box(&bf).chunks_exact(8))
                .zip(black_box(&cf).chunks_exact(8))
            {
                mul_add_sub_f32(
                    F32x8::from_slice(x),
                    F32x8::from_slice(y),
                    F32x8::from_slice(z),
                )
                .copy_to_slice(o);
            }
            black_box(&of);
        })
    });

    // Complex multiply, split layout: 4096 complex numbers.
    let (ar, ai) = (&af, &bf);
    let (br, bi) = (&cf, &af);
    let mut or_ = vec![0.0f32; n];
    let mut oi = vec![0.0f32; n];
    c.bench_function("complex_mul_f32/scalar_idx", |bn| {
        bn.iter(|| {
            let (ar, ai, br, bi) = (black_box(ar), black_box(ai), black_box(br), black_box(bi));
            for i in 0..n {
                or_[i] = ar[i] * br[i] - ai[i] * bi[i];
                oi[i] = ar[i] * bi[i] + ai[i] * br[i];
            }
            black_box((&or_, &oi));
        })
    });
    c.bench_function("complex_mul_f32/scalar_idx_fma", |bn| {
        bn.iter(|| {
            let (ar, ai, br, bi) = (black_box(ar), black_box(ai), black_box(br), black_box(bi));
            for i in 0..n {
                or_[i] = ar[i].mul_add(br[i], -(ai[i] * bi[i]));
                oi[i] = ar[i].mul_add(bi[i], ai[i] * br[i]);
            }
            black_box((&or_, &oi));
        })
    });
    let load = |p: &[f32], q: &[f32], k: usize| {
        ComplexSimd::new(
            F32x8::from_slice(&p[k..k + 8]),
            F32x8::from_slice(&q[k..k + 8]),
        )
    };
    c.bench_function("complex_mul_f32/simd", |bn| {
        bn.iter(|| {
            for k in (0..n).step_by(8) {
                let r = complex_mul_f32(black_box(load(ar, ai, k)), black_box(load(br, bi, k)));
                r.real.copy_to_slice(&mut or_[k..k + 8]);
                r.imag.copy_to_slice(&mut oi[k..k + 8]);
            }
            black_box((&or_, &oi));
        })
    });
    c.bench_function("complex_mul_f32/portable", |bn| {
        bn.iter(|| {
            for k in (0..n).step_by(8) {
                let r = complex_mul_f32_portable(
                    black_box(load(ar, ai, k)),
                    black_box(load(br, bi, k)),
                );
                r.real.copy_to_slice(&mut or_[k..k + 8]);
                r.imag.copy_to_slice(&mut oi[k..k + 8]);
            }
            black_box((&or_, &oi));
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
