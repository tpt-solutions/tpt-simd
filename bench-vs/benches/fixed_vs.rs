//! tpt-simd-fixed 8-lane Q16.16 vector ops vs the scalar `fixed` crate (dev-dep oracle).
//! `fixed` has no SIMD, so the incumbent does the same eight operations one lane at a time.
//! This measures whether the SIMD fixed-point vector is worth having over a scalar type.
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fixed::types::I16F16;
use std::hint::black_box;
use tpt_simd_core::Simd;
use tpt_simd_fixed::Fixed;

const LANES: usize = 4096;

type Q16 = Fixed<i32, 16, 16, 8>;

fn raw_blocks(seed: i32) -> Vec<[i32; 8]> {
    (0..LANES / 8)
        .map(|b| std::array::from_fn(|i| ((b * 8 + i) as i32).wrapping_mul(seed) ^ 0x1234))
        .collect()
}

fn bench(c: &mut Criterion) {
    let a = raw_blocks(40_503);
    let b = raw_blocks(65_521);

    let mut g = c.benchmark_group("fixed/saturating_mul/q16_16/4096");
    g.throughput(Throughput::Elements(LANES as u64));
    g.bench_function(BenchmarkId::new("tpt-simd-vector", LANES), |bch| {
        bch.iter(|| {
            let mut sum = 0i32;
            for (x, y) in black_box(&a).iter().zip(&b) {
                let r = Q16::from_raw(Simd(*x)).saturating_mul(Q16::from_raw(Simd(*y)));
                sum = sum.wrapping_add(r.raw().0[0]);
            }
            sum
        })
    });
    g.bench_function(BenchmarkId::new("fixed-scalar", LANES), |bch| {
        bch.iter(|| {
            let mut sum = 0i32;
            for (x, y) in black_box(&a).iter().zip(&b) {
                for lane in 0..8 {
                    let p = I16F16::from_bits(x[lane]).saturating_mul(I16F16::from_bits(y[lane]));
                    sum = sum.wrapping_add(p.to_bits());
                }
            }
            sum
        })
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
