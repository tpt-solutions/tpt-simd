//! Benchmarks: tpt-simd butterflies vs plain scalar loops.
#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_butterfly::*;
use tpt_simd_core::{ComplexSimd, F32x8, I16x16};

const N: usize = 4096;

fn bench(c: &mut Criterion) {
    let mut a: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();
    let mut b: Vec<f32> = (0..N).map(|i| 3.0 - i as f32).collect();
    let mut ai: Vec<i16> = (0..N).map(|i| (i * 13) as i16).collect();
    let mut bi: Vec<i16> = (0..N).map(|i| (i * 31) as i16).collect();
    let mut g = c.benchmark_group("butterfly");
    g.throughput(Throughput::Elements(N as u64));

    g.bench_function("f32/scalar", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&mut a), black_box(&mut b));
            for i in 0..N {
                let (x, y) = (a[i], b[i]);
                a[i] = x + y;
                b[i] = x - y;
            }
        })
    });
    g.bench_function("f32/scalar_iter", |bn| {
        bn.iter(|| {
            for (x, y) in black_box(&mut a)
                .iter_mut()
                .zip(black_box(&mut b).iter_mut())
            {
                let (p, q) = (*x, *y);
                *x = p + q;
                *y = p - q;
            }
        })
    });
    g.bench_function("i16/scalar_iter", |bn| {
        bn.iter(|| {
            for (x, y) in black_box(&mut ai)
                .iter_mut()
                .zip(black_box(&mut bi).iter_mut())
            {
                let (p, q) = (*x, *y);
                *x = p.wrapping_add(q);
                *y = p.wrapping_sub(q);
            }
        })
    });
    g.bench_function("f32/tpt", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&mut a), black_box(&mut b));
            for (ca, cb) in a.chunks_exact_mut(8).zip(b.chunks_exact_mut(8)) {
                let (mut x, mut y) = (F32x8::from_slice(ca), F32x8::from_slice(cb));
                butterfly_f32(&mut x, &mut y);
                x.copy_to_slice(ca);
                y.copy_to_slice(cb);
            }
        })
    });
    g.bench_function("i16/scalar", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&mut ai), black_box(&mut bi));
            for i in 0..N {
                let (x, y) = (a[i], b[i]);
                a[i] = x.wrapping_add(y);
                b[i] = x.wrapping_sub(y);
            }
        })
    });
    g.bench_function("i16/tpt", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&mut ai), black_box(&mut bi));
            for (ca, cb) in a.chunks_exact_mut(16).zip(b.chunks_exact_mut(16)) {
                let (mut x, mut y) = (I16x16::from_slice(ca), I16x16::from_slice(cb));
                butterfly_i16(&mut x, &mut y);
                x.copy_to_slice(ca);
                y.copy_to_slice(cb);
            }
        })
    });

    // Complex DIT butterfly with twiddle on split arrays.
    let mut ar = a.clone();
    let mut aim = b.clone();
    let mut br = b.clone();
    let mut bim = a.clone();
    let wr: Vec<f32> = (0..N).map(|i| (i as f32 * 0.01).cos()).collect();
    let wi: Vec<f32> = (0..N).map(|i| (i as f32 * 0.01).sin()).collect();
    g.bench_function("twiddle_complex_f32/scalar", |bn| {
        bn.iter(|| {
            let (ar, aim, br, bim) = (
                black_box(&mut ar),
                black_box(&mut aim),
                black_box(&mut br),
                black_box(&mut bim),
            );
            for i in 0..N {
                let tr = br[i] * wr[i] - bim[i] * wi[i];
                let ti = br[i] * wi[i] + bim[i] * wr[i];
                let (xr, xi) = (ar[i], aim[i]);
                ar[i] = xr + tr;
                aim[i] = xi + ti;
                br[i] = xr - tr;
                bim[i] = xi - ti;
            }
        })
    });
    g.bench_function("twiddle_complex_f32/tpt", |bn| {
        bn.iter(|| {
            let (ar, aim, br, bim) = (
                black_box(&mut ar),
                black_box(&mut aim),
                black_box(&mut br),
                black_box(&mut bim),
            );
            for i in (0..N).step_by(8) {
                let mut x =
                    ComplexSimd::new(F32x8::from_slice(&ar[i..]), F32x8::from_slice(&aim[i..]));
                let mut y =
                    ComplexSimd::new(F32x8::from_slice(&br[i..]), F32x8::from_slice(&bim[i..]));
                let w = ComplexSimd::new(F32x8::from_slice(&wr[i..]), F32x8::from_slice(&wi[i..]));
                butterfly_with_twiddle_complex_f32(&mut x, &mut y, w);
                x.real.copy_to_slice(&mut ar[i..]);
                x.imag.copy_to_slice(&mut aim[i..]);
                y.real.copy_to_slice(&mut br[i..]);
                y.imag.copy_to_slice(&mut bim[i..]);
            }
        })
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
