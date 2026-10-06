//! Scatter vs plain scalar stores (`black_box` inputs).
//!
//! `scalar` is a naive indexed loop; `portable` is the crate's scalar
//! reference; `tpt` is the public function (`vpscatterdd` only when built
//! with avx512f+avx512vl, otherwise identical to `portable`); `checked` adds
//! bounds checks. Indices are a random permutation-free mix (duplicates
//! possible), so the stores are ordered.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_core::{F32x8, I32x8};
use tpt_simd_scatter::*;

fn scalar_i32(data: &mut [i32], idx: &[i32; 8], vals: &[i32; 8]) {
    for k in 0..8 {
        data[idx[k] as usize] = vals[k];
    }
}

fn benches(c: &mut Criterion) {
    for &n in &[256usize, 1 << 22] {
        let mut data = vec![0i32; n];
        let mut fdata = vec![0.0f32; n];
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
        let vals = [1, 2, 3, 4, 5, 6, 7, 8];
        let vv = I32x8::from_array(vals);
        let fv = F32x8::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut g = c.benchmark_group(format!("scatter_i32_{n}"));
        g.bench_function("scalar", |t| {
            t.iter(|| {
                for i in &idx_sets {
                    scalar_i32(black_box(&mut data), black_box(i), black_box(&vals));
                }
            })
        });
        g.bench_function("portable", |t| {
            t.iter(|| {
                for v in &vecs {
                    // SAFETY: all indices are < n.
                    unsafe {
                        scatter_i32_portable(
                            black_box(data.as_mut_ptr()),
                            black_box(*v),
                            black_box(vv),
                        )
                    }
                }
            })
        });
        g.bench_function("tpt", |t| {
            t.iter(|| {
                for v in &vecs {
                    // SAFETY: all indices are < n.
                    unsafe {
                        scatter_i32(black_box(data.as_mut_ptr()), black_box(*v), black_box(vv))
                    }
                }
            })
        });
        g.bench_function("checked", |t| {
            t.iter(|| {
                for v in &vecs {
                    scatter_checked_i32(black_box(&mut data), black_box(*v), black_box(vv));
                }
            })
        });
        g.bench_function("tpt_f32", |t| {
            t.iter(|| {
                for v in &vecs {
                    // SAFETY: all indices are < n.
                    unsafe {
                        scatter_f32(black_box(fdata.as_mut_ptr()), black_box(*v), black_box(fv))
                    }
                }
            })
        });
        g.finish();
    }
}

criterion_group!(scatter, benches);
criterion_main!(scatter);
