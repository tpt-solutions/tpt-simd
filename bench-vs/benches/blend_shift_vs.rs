//! tpt-simd-blend and tpt-simd-shift vs wide (dev-dep oracle), over 4096 lanes in 8-lane blocks.
//! Both sides load each block from an array and store the result, so the comparison covers
//! the same work in each implementation.
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_core::{Simd, SimdMask};

const LANES: usize = 4096;

fn blocks_f32(seed: u32) -> Vec<[f32; 8]> {
    (0..LANES / 8)
        .map(|b| std::array::from_fn(|i| ((b * 8 + i) as u32 ^ seed) as f32 * 0.25))
        .collect()
}

fn blocks_i32(seed: i32) -> Vec<[i32; 8]> {
    (0..LANES / 8)
        .map(|b| std::array::from_fn(|i| ((b * 8 + i) as i32).wrapping_mul(seed)))
        .collect()
}

fn masks() -> Vec<[bool; 8]> {
    (0..LANES / 8)
        .map(|b| std::array::from_fn(|i| (b * 8 + i) % 3 == 0))
        .collect()
}

fn bench(c: &mut Criterion) {
    let a = blocks_f32(0x5555);
    let b = blocks_f32(0xAAAA);
    let m = masks();
    let ai = blocks_i32(3);
    let shifts = 5u32;

    let mut g = c.benchmark_group("blend/f32/4096");
    g.throughput(Throughput::Elements(LANES as u64));
    g.bench_function(BenchmarkId::new("tpt", LANES), |bch| {
        bch.iter(|| {
            let mut sum = 0f32;
            for ((x, y), mk) in black_box(&a).iter().zip(&b).zip(&m) {
                let r = tpt_simd_blend::blend_f32(SimdMask::from_array(*mk), Simd(*x), Simd(*y));
                sum += r.0[0];
            }
            sum
        })
    });
    g.bench_function(BenchmarkId::new("wide", LANES), |bch| {
        bch.iter(|| {
            let mut sum = 0f32;
            for ((x, y), mk) in black_box(&a).iter().zip(&b).zip(&m) {
                let mask =
                    wide::f32x8::new(mk.map(|t| if t { f32::from_bits(u32::MAX) } else { 0.0 }));
                let r = mask.select(wide::f32x8::new(*x), wide::f32x8::new(*y));
                sum += r.to_array()[0];
            }
            sum
        })
    });
    g.finish();

    let mut g = c.benchmark_group("shift_left/i32/4096");
    g.throughput(Throughput::Elements(LANES as u64));
    g.bench_function(BenchmarkId::new("tpt", LANES), |bch| {
        bch.iter(|| {
            let mut sum = 0i32;
            for x in black_box(&ai) {
                let r = tpt_simd_shift::shift_left_i32(Simd(*x), shifts);
                sum = sum.wrapping_add(r.0[0]);
            }
            sum
        })
    });
    g.bench_function(BenchmarkId::new("wide", LANES), |bch| {
        bch.iter(|| {
            let mut sum = 0i32;
            for x in black_box(&ai) {
                let r = wide::i32x8::new(*x) << shifts;
                sum = sum.wrapping_add(r.to_array()[0]);
            }
            sum
        })
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
