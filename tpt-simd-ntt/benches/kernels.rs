//! Modular multiplication: Montgomery and Barrett vs the `%` operator (`black_box` inputs).

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_ntt::{Barrett, Montgomery, NttPlan, bigint_mul};

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

    // Whole forward transform. The copy is in the setup closure, so only the transform is timed.
    let plan = NttPlan::new(2_013_265_921, 31, 4096);
    let base: Vec<u32> = (0..4096u32)
        .map(|i| i.wrapping_mul(747_796_405) % 2_013_265_921)
        .collect();
    let mut group = c.benchmark_group("ntt_forward_4096");
    group.throughput(Throughput::Elements(4096));
    group.bench_function("tpt", |bch| {
        bch.iter_batched(
            || black_box(base.clone()),
            |mut v| {
                plan.forward(&mut v);
                v
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn schoolbook(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = vec![0u32; a.len() + b.len()];
    for (i, &x) in a.iter().enumerate() {
        let mut carry = 0u64;
        for (j, &y) in b.iter().enumerate() {
            let t = u64::from(out[i + j]) + u64::from(x) * u64::from(y) + carry;
            out[i + j] = t as u32;
            carry = t >> 32;
        }
        out[i + b.len()] = carry as u32;
    }
    out
}

fn bench_bigint(c: &mut Criterion) {
    for n in [32usize, 256, 512, 1024, 2048] {
        let a: Vec<u32> = (0..n as u32)
            .map(|i| i.wrapping_mul(2_654_435_761) | 1)
            .collect();
        let b: Vec<u32> = (0..n as u32)
            .map(|i| i.wrapping_mul(40_503).wrapping_add(7))
            .collect();
        let mut group = c.benchmark_group(format!("bigint_mul/{n}_limbs"));
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function("schoolbook", |bch| {
            bch.iter(|| schoolbook(black_box(&a), black_box(&b)))
        });
        group.bench_function("ntt", |bch| {
            bch.iter(|| bigint_mul(black_box(&a), black_box(&b)))
        });
        group.finish();
    }
}

criterion_group!(benches, bench, bench_bigint);
criterion_main!(benches);
