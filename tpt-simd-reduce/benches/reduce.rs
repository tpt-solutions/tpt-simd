//! Slice reductions vs naive iterator / scalar loops (f32 unless noted).
//!
//! The naive baselines are the obvious single-accumulator loops. Float adds
//! are not reassociable, so LLVM cannot vectorise them; min/max folds and
//! integer sums it often can.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_reduce::*;

fn data32(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7919 % 2003) as f32 - 1000.0) * 0.37)
        .collect()
}
fn data64(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i * 7919 % 2003) as f64 - 1000.0) * 0.37)
        .collect()
}

fn naive_kahan(xs: &[f32]) -> f32 {
    let (mut s, mut c) = (0.0f32, 0.0f32);
    for &x in xs {
        let y = x - c;
        let t = s + y;
        c = (t - s) - y;
        s = t;
    }
    s
}
fn naive_mean(xs: &[f32]) -> Option<f32> {
    (!xs.is_empty()).then(|| xs.iter().sum::<f32>() / xs.len() as f32)
}
fn naive_variance(xs: &[f32]) -> Option<f32> {
    let m = naive_mean(xs)?;
    Some(xs.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / xs.len() as f32)
}
fn naive_cov(x: &[f32], y: &[f32]) -> Option<f32> {
    let (mx, my) = (naive_mean(x)?, naive_mean(y)?);
    Some(
        x.iter()
            .zip(y)
            .map(|(&a, &b)| (a - mx) * (b - my))
            .sum::<f32>()
            / x.len() as f32,
    )
}
fn naive_min(xs: &[f32]) -> Option<f32> {
    xs.iter().copied().fold(None, |a, b| match a {
        None => Some(b),
        Some(a) => Some(a.min(b)),
    })
}
fn naive_max(xs: &[f32]) -> Option<f32> {
    xs.iter().copied().fold(None, |a, b| match a {
        None => Some(b),
        Some(a) => Some(a.max(b)),
    })
}
fn naive_argmax(xs: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &x) in xs.iter().enumerate() {
        if !x.is_nan() && best.is_none_or(|(_, b)| x > b) {
            best = Some((i, x));
        }
    }
    best.map(|b| b.0)
}

fn bench(c: &mut Criterion) {
    for &n in &[1024usize, 65536] {
        let x = data32(n);
        let y: Vec<f32> = x.iter().rev().copied().collect();
        let mut g = c.benchmark_group(format!("f32_{n}"));
        g.throughput(Throughput::Elements(n as u64));
        macro_rules! pair {
            ($name:expr, $naive:expr, $tpt:expr) => {
                g.bench_function(BenchmarkId::new($name, "naive"), |b| b.iter(|| $naive));
                g.bench_function(BenchmarkId::new($name, "tpt"), |b| b.iter(|| $tpt));
            };
        }
        pair!(
            "sum",
            black_box(&x).iter().sum::<f32>(),
            sum_f32(black_box(&x))
        );
        pair!(
            "sum_pairwise",
            black_box(&x).iter().sum::<f32>(),
            pairwise_sum_f32(black_box(&x))
        );
        pair!(
            "sum_compensated(kahan)",
            naive_kahan(black_box(&x)),
            sum_compensated_f32(black_box(&x))
        );
        pair!("mean", naive_mean(black_box(&x)), mean_f32(black_box(&x)));
        pair!(
            "variance",
            naive_variance(black_box(&x)),
            variance_f32(black_box(&x))
        );
        pair!(
            "covariance",
            naive_cov(black_box(&x), black_box(&y)),
            covariance_f32(black_box(&x), black_box(&y))
        );
        pair!("min", naive_min(black_box(&x)), min_f32(black_box(&x)));
        pair!("max", naive_max(black_box(&x)), max_f32(black_box(&x)));
        pair!(
            "argmax",
            naive_argmax(black_box(&x)),
            argmax_f32(black_box(&x))
        );
        pair!(
            "sum_squares",
            black_box(&x).iter().map(|&v| v * v).sum::<f32>(),
            sum_squares_f32(black_box(&x))
        );
        pair!(
            "norm",
            black_box(&x).iter().map(|&v| v * v).sum::<f32>().sqrt(),
            norm_f32(black_box(&x))
        );
        g.finish();
    }

    let n = 1024;
    let x = data64(n);
    let xi: Vec<i32> = (0..n as i32)
        .map(|i| i.wrapping_mul(7919) % 2003 - 1000)
        .collect();
    let mut g = c.benchmark_group("other_1024");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("sum_f64/naive", |b| {
        b.iter(|| black_box(&x).iter().sum::<f64>())
    });
    g.bench_function("sum_f64/tpt", |b| b.iter(|| sum_f64(black_box(&x))));
    g.bench_function("max_f64/naive", |b| {
        b.iter(|| {
            black_box(&x)
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
        })
    });
    g.bench_function("max_f64/tpt", |b| b.iter(|| max_f64(black_box(&x))));
    g.bench_function("sum_i32/naive", |b| {
        b.iter(|| black_box(&xi).iter().fold(0i32, |a, &v| a.wrapping_add(v)))
    });
    g.bench_function("sum_i32/tpt", |b| b.iter(|| sum_i32(black_box(&xi))));
    g.bench_function("max_i32/naive", |b| {
        b.iter(|| black_box(&xi).iter().copied().max())
    });
    g.bench_function("max_i32/tpt", |b| b.iter(|| max_i32(black_box(&xi))));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
