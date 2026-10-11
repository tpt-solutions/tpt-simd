//! `Simd<T, N>` lane operations vs scalar loops over a 4096-element slice.
//!
//! `*_scalar_idx` is an indexed loop, `*_scalar_iter` a zip/iterator loop that
//! LLVM can auto-vectorise, and `*_simd` processes `Simd` chunks. The array
//! backend is expected to match the iterator loop, not beat it; this bench is
//! the evidence for that (see docs/benchmarks.md).

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_vector::Simd;

fn bench(c: &mut Criterion) {
    let n = 4096usize;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 - 300.0).collect();
    let b: Vec<f32> = (0..n).map(|i| 1.0 - i as f32 * 0.5).collect();
    let mut out = vec![0f32; n];

    c.bench_function("vector_f32_mul_add/scalar_idx", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&a), black_box(&b));
            for i in 0..n {
                out[i] = a[i] * b[i] + 2.0;
            }
            black_box(&out);
        })
    });
    c.bench_function("vector_f32_mul_add/scalar_iter", |bn| {
        bn.iter(|| {
            for ((o, &x), &y) in out.iter_mut().zip(black_box(&a)).zip(black_box(&b)) {
                *o = x * y + 2.0;
            }
            black_box(&out);
        })
    });
    c.bench_function("vector_f32_mul_add/simd8", |bn| {
        let two = Simd::<f32, 8>::splat(2.0);
        bn.iter(|| {
            let (a, b) = (black_box(&a), black_box(&b));
            for ((o, x), y) in out
                .chunks_exact_mut(8)
                .zip(a.chunks_exact(8))
                .zip(b.chunks_exact(8))
            {
                let v = Simd::<f32, 8>::from_slice(x) * Simd::from_slice(y) + two;
                v.copy_to_slice(o);
            }
            black_box(&out);
        })
    });

    let a16: Vec<i16> = (0..n).map(|i| (i as i32 * 37 - 9000) as i16).collect();
    let b16: Vec<i16> = (0..n).map(|i| (i as i32 * 91 - 20000) as i16).collect();
    let mut out16 = vec![0i16; n];

    c.bench_function("vector_i16_sat_add/scalar_idx", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&a16), black_box(&b16));
            for i in 0..n {
                out16[i] = a[i].saturating_add(b[i]);
            }
            black_box(&out16);
        })
    });
    c.bench_function("vector_i16_sat_add/scalar_iter", |bn| {
        bn.iter(|| {
            for ((o, &x), &y) in out16.iter_mut().zip(black_box(&a16)).zip(black_box(&b16)) {
                *o = x.saturating_add(y);
            }
            black_box(&out16);
        })
    });
    c.bench_function("vector_i16_sat_add/simd16", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&a16), black_box(&b16));
            for ((o, x), y) in out16
                .chunks_exact_mut(16)
                .zip(a.chunks_exact(16))
                .zip(b.chunks_exact(16))
            {
                Simd::<i16, 16>::from_slice(x)
                    .sat_add(Simd::from_slice(y))
                    .copy_to_slice(o);
            }
            black_box(&out16);
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
