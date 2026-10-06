//! Criterion benchmarks: tpt-simd-fixed vs manual scalar fixed-point loops.
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_core::Simd;
use tpt_simd_fixed::Fixed;

type Q = Fixed<i32, 16, 16, 8>;
const LEN: usize = 4096;
const MIN_R: i64 = i32::MIN as i64;
const MAX_R: i64 = i32::MAX as i64;

fn data(seed: u32) -> Vec<i32> {
    let mut s = seed;
    (0..LEN)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s as i32) >> 4 // keep magnitudes moderate so saturation is occasional
        })
        .collect()
}

#[inline(always)]
fn scalar_round(p: i64) -> i64 {
    // round to nearest, ties away from zero
    if p >= 0 {
        (p + (1 << 15)) >> 16
    } else {
        -((-p + (1 << 15)) >> 16)
    }
}

fn bench(c: &mut Criterion) {
    let a = data(1);
    let b = data(2);
    let mut out = vec![0i32; LEN];

    let mut g = c.benchmark_group("fixed_q16_16_4096");

    g.bench_function("mul_sat/scalar", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&a), black_box(&b));
            for i in 0..LEN {
                let r = scalar_round(a[i] as i64 * b[i] as i64);
                out[i] = r.clamp(MIN_R, MAX_R) as i32;
            }
            black_box(&out);
        })
    });
    g.bench_function("mul_sat/tpt-simd", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&a), black_box(&b));
            for ((o, x), y) in out
                .chunks_exact_mut(8)
                .zip(a.chunks_exact(8))
                .zip(b.chunks_exact(8))
            {
                let r = Q::from_raw(Simd::from_slice(x))
                    .saturating_mul(Q::from_raw(Simd::from_slice(y)));
                r.raw().copy_to_slice(o);
            }
            black_box(&out);
        })
    });

    g.bench_function("mul_wrap/scalar", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&a), black_box(&b));
            for i in 0..LEN {
                out[i] = scalar_round(a[i] as i64 * b[i] as i64) as i32;
            }
            black_box(&out);
        })
    });
    g.bench_function("mul_wrap/tpt-simd", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&a), black_box(&b));
            for ((o, x), y) in out
                .chunks_exact_mut(8)
                .zip(a.chunks_exact(8))
                .zip(b.chunks_exact(8))
            {
                let r = Q::from_raw(Simd::from_slice(x)) * Q::from_raw(Simd::from_slice(y));
                r.raw().copy_to_slice(o);
            }
            black_box(&out);
        })
    });

    g.bench_function("add_sat/scalar", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&a), black_box(&b));
            for i in 0..LEN {
                out[i] = a[i].saturating_add(b[i]);
            }
            black_box(&out);
        })
    });
    g.bench_function("add_sat/tpt-simd", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&a), black_box(&b));
            for ((o, x), y) in out
                .chunks_exact_mut(8)
                .zip(a.chunks_exact(8))
                .zip(b.chunks_exact(8))
            {
                let r = Q::from_raw(Simd::from_slice(x))
                    .saturating_add(Q::from_raw(Simd::from_slice(y)));
                r.raw().copy_to_slice(o);
            }
            black_box(&out);
        })
    });

    g.bench_function("from_f32/scalar", |bn| {
        let f: Vec<f32> = a.iter().map(|&v| v as f32 / 65536.0).collect();
        bn.iter(|| {
            let f = black_box(&f);
            for i in 0..LEN {
                out[i] = (f[i] * 65536.0).round_ties_even() as i32;
            }
            black_box(&out);
        })
    });
    g.bench_function("from_f32/tpt-simd", |bn| {
        let f: Vec<f32> = a.iter().map(|&v| v as f32 / 65536.0).collect();
        bn.iter(|| {
            let f = black_box(&f);
            for (o, x) in out.chunks_exact_mut(8).zip(f.chunks_exact(8)) {
                Q::from_f32(Simd::from_slice(x)).raw().copy_to_slice(o);
            }
            black_box(&out);
        })
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
