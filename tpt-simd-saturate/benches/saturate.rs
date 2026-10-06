//! Benchmarks: tpt-simd saturating ops vs plain scalar loops (and, for the
//! pack, a manual AVX2 intrinsic version).
#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_core::{I8x32, I16x16};
use tpt_simd_saturate::*;

const N: usize = 4096;

fn data16(seed: i32) -> Vec<i16> {
    (0..N as i32)
        .map(|i| (i.wrapping_mul(7919).wrapping_add(seed) % 65536 - 32768) as i16)
        .collect()
}

fn bench(c: &mut Criterion) {
    let a = data16(1);
    let b = data16(77);
    let a8: Vec<i8> = a.iter().map(|&x| (x >> 8) as i8).collect();
    let b8: Vec<i8> = b.iter().map(|&x| (x >> 8) as i8).collect();
    let mut out16 = vec![0i16; N];
    let mut out8 = vec![0i8; N];
    let mut g = c.benchmark_group("saturate");
    g.throughput(Throughput::Elements(N as u64));

    g.bench_function("add_i16/scalar", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&a), black_box(&b));
            for i in 0..N {
                out16[i] = a[i].saturating_add(b[i]);
            }
            black_box(&out16);
        })
    });
    g.bench_function("add_i16/scalar_iter", |bn| {
        bn.iter(|| {
            for ((x, y), o) in black_box(&a)
                .iter()
                .zip(black_box(&b))
                .zip(out16.iter_mut())
            {
                *o = x.saturating_add(*y);
            }
            black_box(&out16);
        })
    });
    g.bench_function("mul_i16/scalar_iter", |bn| {
        bn.iter(|| {
            for ((x, y), o) in black_box(&a)
                .iter()
                .zip(black_box(&b))
                .zip(out16.iter_mut())
            {
                *o = (*x as i32 * *y as i32).clamp(-32768, 32767) as i16;
            }
            black_box(&out16);
        })
    });
    g.bench_function("pack_i16_to_i8/scalar_iter", |bn| {
        bn.iter(|| {
            for (x, o) in black_box(&a).iter().zip(out8.iter_mut()) {
                *o = (*x).clamp(-128, 127) as i8;
            }
            black_box(&out8);
        })
    });
    g.bench_function("add_i16/tpt", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&a), black_box(&b));
            for ((x, y), o) in a
                .chunks_exact(16)
                .zip(b.chunks_exact(16))
                .zip(out16.chunks_exact_mut(16))
            {
                saturating_add_i16(I16x16::from_slice(x), I16x16::from_slice(y)).copy_to_slice(o);
            }
            black_box(&out16);
        })
    });
    g.bench_function("add_i8/scalar", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&a8), black_box(&b8));
            for i in 0..N {
                out8[i] = a[i].saturating_add(b[i]);
            }
            black_box(&out8);
        })
    });
    g.bench_function("add_i8/tpt", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&a8), black_box(&b8));
            for ((x, y), o) in a
                .chunks_exact(32)
                .zip(b.chunks_exact(32))
                .zip(out8.chunks_exact_mut(32))
            {
                saturating_add_i8(I8x32::from_slice(x), I8x32::from_slice(y)).copy_to_slice(o);
            }
            black_box(&out8);
        })
    });
    g.bench_function("mul_i16/scalar", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&a), black_box(&b));
            for i in 0..N {
                out16[i] = (a[i] as i32 * b[i] as i32).clamp(-32768, 32767) as i16;
            }
            black_box(&out16);
        })
    });
    g.bench_function("mul_i16/tpt", |bn| {
        bn.iter(|| {
            let (a, b) = (black_box(&a), black_box(&b));
            for ((x, y), o) in a
                .chunks_exact(16)
                .zip(b.chunks_exact(16))
                .zip(out16.chunks_exact_mut(16))
            {
                saturating_mul_i16(I16x16::from_slice(x), I16x16::from_slice(y)).copy_to_slice(o);
            }
            black_box(&out16);
        })
    });
    g.bench_function("pack_i16_to_i8/scalar", |bn| {
        bn.iter(|| {
            let a = black_box(&a);
            for i in 0..N {
                out8[i] = a[i].clamp(-128, 127) as i8;
            }
            black_box(&out8);
        })
    });
    g.bench_function("pack_i16_to_i8/tpt_simd", |bn| {
        bn.iter(|| {
            let a = black_box(&a);
            for (x, o) in a.chunks_exact(16).zip(out8.chunks_exact_mut(16)) {
                pack_i16_to_i8(I16x16::from_slice(x)).copy_to_slice(o);
            }
            black_box(&out8);
        })
    });
    g.bench_function("pack_i16_to_i8/tpt_slice", |bn| {
        bn.iter(|| {
            pack_i16_to_i8_slice(black_box(&a), &mut out8);
            black_box(&out8);
        })
    });
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    g.bench_function("pack_i16_to_i8/manual_avx2", |bn| {
        use core::arch::x86_64::*;
        bn.iter(|| {
            let a = black_box(&a);
            for i in (0..N).step_by(32) {
                // SAFETY: AVX2 is enabled at compile time; the indices
                // `i..i+32` of `a` and `out8` are in bounds (N % 32 == 0).
                unsafe {
                    let x = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
                    let y = _mm256_loadu_si256(a.as_ptr().add(i + 16) as *const __m256i);
                    let p = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packs_epi16(x, y));
                    _mm256_storeu_si256(out8.as_mut_ptr().add(i) as *mut __m256i, p);
                }
            }
            black_box(&out8);
        })
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
