//! Small fixed-size matrix ops vs the crate's scalar references.
//!
//! Each iteration runs a batch of 256 independent matrices (per-matrix time
//! = reported time / 256) to amortise `black_box` overhead.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_matrix::{
    mat3x3_inverse_f32, mat4x4_mul_f32, mat4x4_mul_scalar_f32, mat4x4_transpose_f32,
    mat8x8_mul_i16, mat8x8_mul_scalar_i16, mat8x8_transpose_i16,
};

const B: usize = 256;

fn bench(c: &mut Criterion) {
    let m4: Vec<[[f32; 4]; 4]> = (0..B)
        .map(|k| {
            core::array::from_fn(|i| {
                core::array::from_fn(|j| ((k + i * 4 + j) % 13) as f32 * 0.25 - 1.0)
            })
        })
        .collect();
    let m8: Vec<[[i16; 8]; 8]> = (0..B)
        .map(|k| {
            core::array::from_fn(|i| {
                core::array::from_fn(|j| ((k * 31 + i * 8 + j) % 41) as i16 - 20)
            })
        })
        .collect();
    let m3: Vec<[[f32; 3]; 3]> = (0..B)
        .map(|k| {
            core::array::from_fn(|i| {
                core::array::from_fn(|j| {
                    ((k + i * 3 + j) % 7) as f32 + if i == j { 5.0 } else { 0.0 }
                })
            })
        })
        .collect();

    c.bench_function("mat4x4_mul_f32/scalar_x256", |b| {
        b.iter(|| {
            let mut s = 0.0f32;
            for m in black_box(&m4) {
                s += mat4x4_mul_scalar_f32(*m, *m).iter().flatten().sum::<f32>();
            }
            black_box(s)
        })
    });
    c.bench_function("mat4x4_mul_f32/simd_x256", |b| {
        b.iter(|| {
            let mut s = 0.0f32;
            for m in black_box(&m4) {
                s += mat4x4_mul_f32(*m, *m).iter().flatten().sum::<f32>();
            }
            black_box(s)
        })
    });

    c.bench_function("mat8x8_mul_i16/scalar_x256", |b| {
        b.iter(|| {
            let mut s = 0i32;
            for m in black_box(&m8) {
                s += mat8x8_mul_scalar_i16(*m, *m)
                    .iter()
                    .flatten()
                    .map(|&x| i32::from(x))
                    .sum::<i32>();
            }
            black_box(s)
        })
    });
    c.bench_function("mat8x8_mul_i16/simd_x256", |b| {
        b.iter(|| {
            let mut s = 0i32;
            for m in black_box(&m8) {
                s += mat8x8_mul_i16(*m, *m)
                    .iter()
                    .flatten()
                    .map(|&x| i32::from(x))
                    .sum::<i32>();
            }
            black_box(s)
        })
    });

    c.bench_function("mat4x4_transpose_f32/scalar_x256", |b| {
        let mut w = m4.clone();
        b.iter(|| {
            for m in black_box(&mut w).iter_mut() {
                let t = *m;
                for i in 0..4 {
                    for j in 0..4 {
                        m[i][j] = t[j][i];
                    }
                }
            }
        })
    });
    c.bench_function("mat4x4_transpose_f32/simd_x256", |b| {
        let mut w = m4.clone();
        b.iter(|| {
            for m in black_box(&mut w).iter_mut() {
                mat4x4_transpose_f32(m);
            }
        })
    });
    c.bench_function("mat8x8_transpose_i16/scalar_x256", |b| {
        let mut w = m8.clone();
        b.iter(|| {
            for m in black_box(&mut w).iter_mut() {
                let t = *m;
                for i in 0..8 {
                    for j in 0..8 {
                        m[i][j] = t[j][i];
                    }
                }
            }
        })
    });
    c.bench_function("mat8x8_transpose_i16/simd_x256", |b| {
        let mut w = m8.clone();
        b.iter(|| {
            for m in black_box(&mut w).iter_mut() {
                mat8x8_transpose_i16(m);
            }
        })
    });

    c.bench_function("mat3x3_inverse_f32/x256", |b| {
        b.iter(|| {
            let mut s = 0.0f32;
            for m in black_box(&m3) {
                if let Some(i) = mat3x3_inverse_f32(*m) {
                    s += i[0][0];
                }
            }
            black_box(s)
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
