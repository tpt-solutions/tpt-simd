//! Modular multiplication: Montgomery and Barrett vs the `%` operator (`black_box` inputs).

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_ntt::{Barrett, Montgomery};

const Q: u32 = 998_244_353;

fn bench(c: &mut Criterion) {
    let n = 4096usize;
    let a: Vec<u32> = (0..n as u32)
        .map(|i| i.wrapping_mul(747_796_405) % Q)
        .collect();
    let b: Vec<u32> = (0..n as u32)
        .map(|i| i.wrapping_mul(2_891_336_453) % Q)
        .collect();
    let m = Montgomery::new(Q);
    let bar = Barrett::new(u64::from(Q));

    let mut group = c.benchmark_group("modmul");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("rem", |bch| {
        bch.iter(|| {
            let mut acc = 0u64;
            for (x, y) in black_box(&a).iter().zip(black_box(&b)) {
                acc ^= (u64::from(*x) * u64::from(*y)) % u64::from(Q);
            }
            acc
        })
    });
    group.bench_function("barrett", |bch| {
        bch.iter(|| {
            let mut acc = 0u64;
            for (x, y) in black_box(&a).iter().zip(black_box(&b)) {
                acc ^= bar.reduce(u64::from(*x) * u64::from(*y));
            }
            acc
        })
    });
    group.bench_function("montgomery", |bch| {
        bch.iter(|| {
            let mut acc = 0u32;
            for (x, y) in black_box(&a).iter().zip(black_box(&b)) {
                acc ^= m.mul(*x, *y);
            }
            acc
        })
    });
    // Fixed multiplier in Montgomery form, as for twiddle factors: one reduction per product.
    let b_mont: Vec<u32> = b.iter().map(|&y| m.to_mont(y)).collect();
    group.bench_function("montgomery_fixed_multiplier", |bch| {
        bch.iter(|| {
            let mut acc = 0u32;
            for (x, y) in black_box(&a).iter().zip(black_box(&b_mont)) {
                acc ^= m.mul_mont(*x, *y);
            }
            acc
        })
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
