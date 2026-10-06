//! Vectorised `f32` math versus a scalar `libm` loop and a scalar `std`
//! loop (`f32::exp` etc.; `std` has no stable `erf`).
//!
//! Run natively: `RUSTFLAGS="-C target-cpu=native" cargo bench -p tpt-simd-math --bench math`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_math::*;

/// Deterministic inputs spread over `[lo, hi]`.
fn data(n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n)
        .map(|i| lo + (hi - lo) * ((i * 7919 % 4093) as f32 / 4092.0))
        .collect()
}

#[allow(clippy::type_complexity)]
fn bench(c: &mut Criterion) {
    type Slice = fn(&[f32], &mut [f32]);
    type Scalar = fn(f32) -> f32;
    // (name, input range, tpt slice kernel, libm scalar, std scalar)
    let cases: [(&str, f32, f32, Slice, Scalar, Option<Scalar>); 6] = [
        ("exp", -20.0, 20.0, exp_f32, libm::expf, Some(f32::exp)),
        ("ln", 1.0e-3, 1.0e3, ln_f32, libm::logf, Some(f32::ln)),
        ("sin", -10.0, 10.0, sin_f32, libm::sinf, Some(f32::sin)),
        ("cos", -10.0, 10.0, cos_f32, libm::cosf, Some(f32::cos)),
        ("tanh", -5.0, 5.0, tanh_f32, libm::tanhf, Some(f32::tanh)),
        ("erf", -3.0, 3.0, erf_f32, libm::erff, None),
    ];
    for &n in &[1024usize, 65536] {
        let mut g = c.benchmark_group(format!("math_f32_{n}"));
        g.throughput(Throughput::Elements(n as u64));
        for (name, lo, hi, tpt, lm, st) in cases {
            let x = data(n, lo, hi);
            let mut out = vec![0.0f32; n];
            g.bench_function(BenchmarkId::new(name, "libm"), |b| {
                b.iter(|| {
                    for (o, &v) in out.iter_mut().zip(black_box(&x)) {
                        *o = lm(v);
                    }
                    black_box(&out);
                })
            });
            if let Some(st) = st {
                g.bench_function(BenchmarkId::new(name, "std"), |b| {
                    b.iter(|| {
                        for (o, &v) in out.iter_mut().zip(black_box(&x)) {
                            *o = st(v);
                        }
                        black_box(&out);
                    })
                });
            }
            g.bench_function(BenchmarkId::new(name, "tpt"), |b| {
                b.iter(|| {
                    tpt(black_box(&x), &mut out);
                    black_box(&out);
                })
            });
        }
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
