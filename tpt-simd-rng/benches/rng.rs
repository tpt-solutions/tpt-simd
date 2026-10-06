//! Lane-parallel generators vs a scalar xoshiro256++ loop and a scalar
//! Box-Muller using libm. Throughput is reported per output element.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_rng::{Philox4x32X8, Rng8, Xoshiro256pp, Xoshiro256ppX8};

const SIZES: [usize; 2] = [4096, 1 << 15];

fn scalar_u64(g: &mut Xoshiro256pp, out: &mut [u64]) {
    for o in out {
        *o = g.next_u64();
    }
}

fn scalar_f32(g: &mut Xoshiro256pp, out: &mut [f32]) {
    for o in out {
        *o = (g.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0);
    }
}

fn scalar_f64(g: &mut Xoshiro256pp, out: &mut [f64]) {
    for o in out {
        *o = (g.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0);
    }
}

fn scalar_normal_f32(g: &mut Xoshiro256pp, out: &mut [f32]) {
    for pair in out.chunks_mut(2) {
        let u1 = 1.0 - (g.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0);
        let u2 = (g.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0);
        let r = libm::sqrtf(-2.0 * libm::logf(u1));
        let (s, c) = libm::sincosf(core::f32::consts::TAU * u2);
        pair[0] = r * c;
        if let Some(p) = pair.get_mut(1) {
            *p = r * s;
        }
    }
}

fn scalar_normal_f64(g: &mut Xoshiro256pp, out: &mut [f64]) {
    for pair in out.chunks_mut(2) {
        let u1 = 1.0 - (g.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0);
        let u2 = (g.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0);
        let r = libm::sqrt(-2.0 * libm::log(u1));
        let (s, c) = libm::sincos(core::f64::consts::TAU * u2);
        pair[0] = r * c;
        if let Some(p) = pair.get_mut(1) {
            *p = r * s;
        }
    }
}

macro_rules! bench_pair {
    ($c:expr, $name:expr, $t:ty, $zero:expr, $scalar:expr, $simd:ident) => {{
        let mut group = $c.benchmark_group($name);
        for &n in &SIZES {
            group.throughput(Throughput::Elements(n as u64));
            let mut buf: Vec<$t> = vec![$zero; n];
            let mut sg = Xoshiro256pp::from_seed(1);
            group.bench_with_input(BenchmarkId::new("scalar_xoshiro", n), &n, |b, _| {
                b.iter(|| {
                    $scalar(&mut sg, black_box(&mut buf));
                    black_box(&buf);
                })
            });
            let mut vg = Xoshiro256ppX8::from_seed(1);
            group.bench_with_input(BenchmarkId::new("x8", n), &n, |b, _| {
                b.iter(|| {
                    vg.$simd(black_box(&mut buf));
                    black_box(&buf);
                })
            });
        }
        group.finish();
    }};
}

fn benches(c: &mut Criterion) {
    bench_pair!(c, "fill_u64", u64, 0u64, scalar_u64, fill_u64);
    bench_pair!(c, "uniform_f32", f32, 0f32, scalar_f32, fill_f32);
    bench_pair!(c, "uniform_f64", f64, 0f64, scalar_f64, fill_f64);
    bench_pair!(
        c,
        "normal_f32",
        f32,
        0f32,
        scalar_normal_f32,
        fill_normal_f32
    );
    bench_pair!(
        c,
        "normal_f64",
        f64,
        0f64,
        scalar_normal_f64,
        fill_normal_f64
    );

    let mut group = c.benchmark_group("philox_x8");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let mut g = Philox4x32X8::from_seed(1);
        let mut u = vec![0u32; n];
        group.bench_with_input(BenchmarkId::new("fill_u32", n), &n, |b, _| {
            b.iter(|| {
                g.fill_u32(black_box(&mut u));
                black_box(&u);
            })
        });
        let mut f = vec![0f32; n];
        group.bench_with_input(BenchmarkId::new("normal_f32", n), &n, |b, _| {
            b.iter(|| {
                g.fill_normal_f32(black_box(&mut f));
                black_box(&f);
            })
        });
    }
    group.finish();

    // xoshiro u32 fill for comparison with Philox.
    let mut group = c.benchmark_group("fill_u32");
    for &n in &SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let mut g = Xoshiro256ppX8::from_seed(1);
        let mut u = vec![0u32; n];
        group.bench_with_input(BenchmarkId::new("x8", n), &n, |b, _| {
            b.iter(|| {
                g.fill_u32(black_box(&mut u));
                black_box(&u);
            })
        });
    }
    group.finish();
}

criterion_group!(rng, benches);
criterion_main!(rng);
