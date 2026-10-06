//! Benchmarks: tpt-simd shifts vs scalar loops.
#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_core::{I32x8, U32x8};
use tpt_simd_shift::*;

const N: usize = 4096;

fn bench(c: &mut Criterion) {
    let a: Vec<i32> = (0..N as i32)
        .map(|i| i.wrapping_mul(2_654_435_761u32 as i32))
        .collect();
    let n: Vec<u32> = (0..N as u32).map(|i| i.wrapping_mul(7) % 36).collect();
    let mut out = vec![0i32; N];
    let mut g = c.benchmark_group("shift");
    g.throughput(Throughput::Elements(N as u64));

    macro_rules! uniform {
        ($name:literal, $scalar:expr, $simd:ident) => {
            g.bench_function(concat!($name, "/scalar"), |b| {
                b.iter(|| {
                    for (o, &x) in out.iter_mut().zip(black_box(&a)) {
                        *o = $scalar(x, black_box(5u32));
                    }
                    black_box(&out);
                })
            });
            g.bench_function(concat!($name, "/simd"), |b| {
                b.iter(|| {
                    for (o, x) in out.chunks_exact_mut(8).zip(black_box(&a).chunks_exact(8)) {
                        $simd(I32x8::from_slice(x), black_box(5u32)).copy_to_slice(o);
                    }
                    black_box(&out);
                })
            });
        };
    }
    uniform!("shift_left", |x: i32, n: u32| x << n, shift_left_i32);
    uniform!(
        "shift_right_logical",
        |x: i32, n: u32| ((x as u32) >> n) as i32,
        shift_right_logical_i32
    );
    uniform!(
        "shift_right_arithmetic",
        |x: i32, n: u32| x >> n,
        shift_right_arithmetic_i32
    );
    uniform!(
        "rotate_left",
        |x: i32, n: u32| x.rotate_left(n),
        rotate_left_i32
    );
    uniform!(
        "shift_with_rounding",
        |x: i32, n: u32| ((i64::from(x) + (1 << (n - 1))) >> n) as i32,
        shift_with_rounding_i32
    );

    macro_rules! var {
        ($name:literal, $scalar:expr, $simd:ident) => {
            g.bench_function(concat!($name, "/scalar"), |b| {
                b.iter(|| {
                    for ((o, &x), &s) in out.iter_mut().zip(black_box(&a)).zip(black_box(&n)) {
                        *o = $scalar(x, s);
                    }
                    black_box(&out);
                })
            });
            g.bench_function(concat!($name, "/simd"), |b| {
                b.iter(|| {
                    for ((o, x), s) in out
                        .chunks_exact_mut(8)
                        .zip(black_box(&a).chunks_exact(8))
                        .zip(black_box(&n).chunks_exact(8))
                    {
                        $simd(I32x8::from_slice(x), U32x8::from_slice(s)).copy_to_slice(o);
                    }
                    black_box(&out);
                })
            });
        };
    }
    var!(
        "shift_left_var",
        |x: i32, n: u32| if n >= 32 { 0 } else { x << n },
        shift_left_var_i32
    );
    var!(
        "shift_right_arithmetic_var",
        |x: i32, n: u32| x >> n.min(31),
        shift_right_arithmetic_var_i32
    );
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
