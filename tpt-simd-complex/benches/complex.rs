//! Complex multiply benchmarks: scalar reference vs tpt-simd.
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tpt_simd_complex::{ComplexSimd, ComplexSimdF32Ext, complex_mul_f32};
use tpt_simd_mul::complex_mul_f32_portable;

const N: usize = 1024; // complex numbers per iteration (128 vectors)

fn data() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let f = |s: f32| {
        (0..N)
            .map(|i| ((i as f32) * s).sin() + 1.5)
            .collect::<Vec<_>>()
    };
    (f(0.1), f(0.2), f(0.3), f(0.4))
}

fn bench(c: &mut Criterion) {
    let (ar, ai, br, bi) = data();
    let mut g = c.benchmark_group("complex_mul_1024");
    g.bench_function("scalar", |b| {
        let (mut or, mut oi) = (vec![0.0f32; N], vec![0.0f32; N]);
        b.iter(|| {
            let (ar, ai, br, bi) = (
                black_box(&ar),
                black_box(&ai),
                black_box(&br),
                black_box(&bi),
            );
            for i in 0..N {
                or[i] = ar[i] * br[i] - ai[i] * bi[i];
                oi[i] = ar[i] * bi[i] + ai[i] * br[i];
            }
            black_box((&or, &oi));
        })
    });
    // Scalar with black_box per element: 8 genuinely separate complex muls.
    g.bench_function("scalar_noautovec", |b| {
        let (mut or, mut oi) = (vec![0.0f32; N], vec![0.0f32; N]);
        b.iter(|| {
            for i in 0..N {
                let (x, y, u, v) = (
                    black_box(ar[i]),
                    black_box(ai[i]),
                    black_box(br[i]),
                    black_box(bi[i]),
                );
                or[i] = x * u - y * v;
                oi[i] = x * v + y * u;
            }
            black_box((&or, &oi));
        })
    });
    // Iterator/zip loops: no bounds checks, so LLVM auto-vectorises them.
    // `zip_unfused` is the fair "auto-vectorised scalar" baseline (not
    // bit-identical to tpt: mul+sub rounds twice); `zip_fma` uses the same
    // fused rule as tpt (bit-identical) and also auto-vectorises under
    // `-C target-cpu=native`.
    g.bench_function("scalar_zip_unfused", |b| {
        let (mut or, mut oi) = (vec![0.0f32; N], vec![0.0f32; N]);
        b.iter(|| {
            let (ar, ai, br, bi) = (
                black_box(&ar),
                black_box(&ai),
                black_box(&br),
                black_box(&bi),
            );
            for ((((o_r, o_i), (&x, &y)), &u), &v) in or
                .iter_mut()
                .zip(oi.iter_mut())
                .zip(ar.iter().zip(ai))
                .zip(br)
                .zip(bi)
            {
                *o_r = x * u - y * v;
                *o_i = x * v + y * u;
            }
            black_box((&or, &oi));
        })
    });
    g.bench_function("scalar_zip_fma", |b| {
        let (mut or, mut oi) = (vec![0.0f32; N], vec![0.0f32; N]);
        b.iter(|| {
            let (ar, ai, br, bi) = (
                black_box(&ar),
                black_box(&ai),
                black_box(&br),
                black_box(&bi),
            );
            for ((((o_r, o_i), (&x, &y)), &u), &v) in or
                .iter_mut()
                .zip(oi.iter_mut())
                .zip(ar.iter().zip(ai))
                .zip(br)
                .zip(bi)
            {
                *o_r = x.mul_add(u, -(y * v));
                *o_i = x.mul_add(v, y * u);
            }
            black_box((&or, &oi));
        })
    });
    // tpt over whole slices (chunks, no per-vector `black_box`).
    g.bench_function("tpt_complex_mul_f32_chunks", |b| {
        let (mut or, mut oi) = (vec![0.0f32; N], vec![0.0f32; N]);
        b.iter(|| {
            let (ar, ai, br, bi) = (
                black_box(&ar),
                black_box(&ai),
                black_box(&br),
                black_box(&bi),
            );
            for ((((ca, cb), cc), cd), (co, ci)) in ar
                .chunks_exact(8)
                .zip(ai.chunks_exact(8))
                .zip(br.chunks_exact(8))
                .zip(bi.chunks_exact(8))
                .zip(or.chunks_exact_mut(8).zip(oi.chunks_exact_mut(8)))
            {
                let x = ComplexSimd::new(
                    tpt_simd_core::Simd::from_slice(ca),
                    tpt_simd_core::Simd::from_slice(cb),
                );
                let y = ComplexSimd::new(
                    tpt_simd_core::Simd::from_slice(cc),
                    tpt_simd_core::Simd::from_slice(cd),
                );
                let r = complex_mul_f32(x, y);
                r.real.copy_to_slice(co);
                r.imag.copy_to_slice(ci);
            }
            black_box((&or, &oi));
        })
    });
    let run =
        |b: &mut criterion::Bencher,
         f: fn(ComplexSimd<f32, 8>, ComplexSimd<f32, 8>) -> ComplexSimd<f32, 8>| {
            let (mut or, mut oi) = (vec![0.0f32; N], vec![0.0f32; N]);
            b.iter(|| {
                for k in (0..N).step_by(8) {
                    let a = ComplexSimd::new(
                        tpt_simd_core::Simd::from_slice(black_box(&ar[k..k + 8])),
                        tpt_simd_core::Simd::from_slice(black_box(&ai[k..k + 8])),
                    );
                    let bb = ComplexSimd::new(
                        tpt_simd_core::Simd::from_slice(black_box(&br[k..k + 8])),
                        tpt_simd_core::Simd::from_slice(black_box(&bi[k..k + 8])),
                    );
                    let r = f(a, bb);
                    r.real.copy_to_slice(&mut or[k..k + 8]);
                    r.imag.copy_to_slice(&mut oi[k..k + 8]);
                }
                black_box((&or, &oi));
            })
        };
    g.bench_function("tpt_complex_mul_f32", |b| run(b, complex_mul_f32));
    g.bench_function("tpt_portable_fused", |b| run(b, complex_mul_f32_portable));
    g.bench_function("tpt_core_unfused_mul", |b| run(b, |a, b| a * b));
    g.bench_function("tpt_twiddle_mul_fma", |b| {
        run(b, |a, b| a.twiddle_mul_fma(b))
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
