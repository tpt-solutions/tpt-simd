//! Gather vs plain scalar loads (`black_box` inputs).
//!
//! `scalar` is a naive indexed loop; `portable` is the crate's scalar
//! reference; `tpt` is the public function (hardware gather when built with
//! `-C target-cpu=native`, otherwise identical to `portable`); `checked`
//! adds bounds checks. Hardware gather is frequently NOT faster than scalar
//! loads; compare the numbers on your CPU.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_core::I32x8;
use tpt_simd_gather::*;

fn scalar_i32(data: &[i32], idx: &[i32; 8]) -> [i32; 8] {
    let mut o = [0; 8];
    for k in 0..8 {
        o[k] = data[idx[k] as usize];
    }
    o
}

fn benches(c: &mut Criterion) {
    // L1-resident and larger-than-L2 tables with pseudo-random indices.
    for &n in &[256usize, 1 << 22] {
        let data: Vec<i32> = (0..n as i32).collect();
        let fdata: Vec<f32> = (0..n).map(|x| x as f32).collect();
        let mut s = 0x2545_F491u32;
        let idx_sets: Vec<[i32; 8]> = (0..1024)
            .map(|_| {
                let mut a = [0; 8];
                for v in &mut a {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    *v = (s as usize % n) as i32;
                }
                a
            })
            .collect();
        let vecs: Vec<I32x8> = idx_sets.iter().map(|a| I32x8::from_array(*a)).collect();
        let mut g = c.benchmark_group(format!("gather_i32_{n}"));
        g.bench_function("scalar", |t| {
            t.iter(|| {
                let mut acc = 0i32;
                for i in &idx_sets {
                    let r = scalar_i32(black_box(&data), black_box(i));
                    acc = acc.wrapping_add(r[0]).wrapping_add(r[7]);
                }
                acc
            })
        });
        g.bench_function("portable", |t| {
            t.iter(|| {
                let mut acc = 0i32;
                for v in &vecs {
                    // SAFETY: all indices are < n.
                    let r = unsafe { gather_i32_portable(black_box(data.as_ptr()), black_box(*v)) };
                    acc = acc.wrapping_add(r.0[0]).wrapping_add(r.0[7]);
                }
                acc
            })
        });
        g.bench_function("tpt", |t| {
            t.iter(|| {
                let mut acc = 0i32;
                for v in &vecs {
                    // SAFETY: all indices are < n.
                    let r = unsafe { gather_i32(black_box(data.as_ptr()), black_box(*v)) };
                    acc = acc.wrapping_add(r.0[0]).wrapping_add(r.0[7]);
                }
                acc
            })
        });
        g.bench_function("checked", |t| {
            t.iter(|| {
                let mut acc = 0i32;
                for v in &vecs {
                    let r = gather_checked_i32(black_box(&data), black_box(*v));
                    acc = acc.wrapping_add(r.0[0]).wrapping_add(r.0[7]);
                }
                acc
            })
        });
        g.bench_function("tpt_f32", |t| {
            t.iter(|| {
                let mut acc = 0.0f32;
                for v in &vecs {
                    // SAFETY: all indices are < n.
                    let r = unsafe { gather_f32(black_box(fdata.as_ptr()), black_box(*v)) };
                    acc += r.0[0] + r.0[7];
                }
                acc
            })
        });
        g.finish();
    }
}

criterion_group!(gather, benches);
criterion_main!(gather);
