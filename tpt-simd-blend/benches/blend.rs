//! Branch-free blend vs a branchy scalar loop on unpredictable data.
//!
//! The condition lanes are pseudo-random, so the scalar `if` mispredicts
//! about half the time unless LLVM if-converts it.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_blend::{blend_f32, select_f32};
use tpt_simd_core::F32x8;

fn branchy(cond: &[f32], a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        if cond[i] > 0.0 {
            out[i] = a[i];
        } else {
            out[i] = b[i];
        }
    }
}

fn bench(c: &mut Criterion) {
    let n = 4096usize;
    let mut s = 0x1234_5678u32;
    let cond: std::vec::Vec<f32> = (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            if s >> 31 == 0 { 1.0 } else { -1.0 }
        })
        .collect();
    let a: std::vec::Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: std::vec::Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
    let mut out = std::vec![0.0f32; n];

    c.bench_function("blend_f32/branchy_scalar", |bn| {
        bn.iter(|| {
            branchy(black_box(&cond), black_box(&a), black_box(&b), &mut out);
            black_box(&out);
        })
    });
    c.bench_function("blend_f32/mask", |bn| {
        bn.iter(|| {
            let z = F32x8::splat(0.0);
            for (((o, cc), aa), bb) in out
                .chunks_exact_mut(8)
                .zip(black_box(&cond).chunks_exact(8))
                .zip(black_box(&a).chunks_exact(8))
                .zip(black_box(&b).chunks_exact(8))
            {
                let m = F32x8::from_slice(cc).simd_gt(z);
                blend_f32(m, F32x8::from_slice(aa), F32x8::from_slice(bb)).copy_to_slice(o);
            }
            black_box(&out);
        })
    });
    c.bench_function("select_f32/sign_bit", |bn| {
        bn.iter(|| {
            for (((o, cc), aa), bb) in out
                .chunks_exact_mut(8)
                .zip(black_box(&cond).chunks_exact(8))
                .zip(black_box(&a).chunks_exact(8))
                .zip(black_box(&b).chunks_exact(8))
            {
                // cond > 0 -> 1.0 ; negate so the sign bit marks "take a".
                let c = -F32x8::from_slice(cc);
                select_f32(c, F32x8::from_slice(aa), F32x8::from_slice(bb)).copy_to_slice(o);
            }
            black_box(&out);
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
