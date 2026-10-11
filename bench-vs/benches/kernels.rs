//! Absolute timings for tpt-simd kernel crates that have no direct incumbent in
//! this suite. Where a simple scalar reference exists, it is timed alongside.
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn f32s(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.37 + seed).sin()).collect()
}

fn bench_distance(c: &mut Criterion) {
    for &n in &[128usize, 4096] {
        let a = f32s(n, 1.0);
        let b = f32s(n, 2.0);
        let mut g = c.benchmark_group(format!("distance/inner_product_f32/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function("tpt", |bn| {
            bn.iter(|| tpt_simd_distance::inner_product_f32(black_box(&a), black_box(&b)))
        });
        g.bench_function("scalar", |bn| {
            bn.iter(|| {
                black_box(&a)
                    .iter()
                    .zip(black_box(&b))
                    .map(|(x, y)| x * y)
                    .sum::<f32>()
            })
        });
        g.finish();
    }
}

fn bench_bits(c: &mut Criterion) {
    let words: Vec<u64> = (0..16_384u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    let mut g = c.benchmark_group("bits/popcount/128KiB");
    g.throughput(Throughput::Bytes((words.len() * 8) as u64));
    g.bench_function("tpt", |b| {
        b.iter(|| tpt_simd_bits::popcount_u64(black_box(&words)))
    });
    g.bench_function("scalar", |b| {
        b.iter(|| {
            black_box(&words)
                .iter()
                .map(|w| u64::from(w.count_ones()))
                .sum::<u64>()
        })
    });
    g.finish();

    let rs = tpt_simd_bits::RankSelect::new(&words, words.len() * 64);
    let mut g = c.benchmark_group("bits/rank1");
    g.bench_function("tpt", |b| {
        b.iter(|| rs.rank1(black_box(words.len() * 32 + 17)))
    });
    g.finish();

    let mut g = c.benchmark_group("bits/prefix_sum_u32/64K");
    let base: Vec<u32> = (0..65_536u32).collect();
    g.throughput(Throughput::Elements(65_536));
    g.bench_function("tpt", |b| {
        b.iter_batched(
            || base.clone(),
            |mut v| tpt_simd_bits::prefix_sum_u32(black_box(&mut v)),
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

fn bench_geom(c: &mut Criterion) {
    let n = 100_000;
    let (x, y, z) = (f32s(n, 0.1), f32s(n, 0.2), f32s(n, 0.3));
    let mut out = vec![0.0f32; n];
    let mut g = c.benchmark_group("geom/distance_sq/100K");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("tpt", |b| {
        b.iter(|| {
            tpt_simd_geom::distance_sq(
                tpt_simd_geom::Soa3::new(black_box(&x), black_box(&y), black_box(&z)),
                [0.5, -0.5, 0.25],
                &mut out,
            )
        })
    });
    g.bench_function("scalar", |b| {
        b.iter(|| {
            for i in 0..n {
                let (dx, dy, dz) = (x[i] - 0.5, y[i] + 0.5, z[i] - 0.25);
                out[i] = dx * dx + dy * dy + dz * dz;
            }
        })
    });
    g.finish();
}

fn bench_dct(c: &mut Criterion) {
    let n = 256;
    let dct = tpt_simd_dct::Dct::new(n);
    let x = f32s(n, 0.5);
    let mut out = vec![0.0f32; n];
    let mut g = c.benchmark_group("dct/forward/256");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("tpt", |b| b.iter(|| dct.forward(black_box(&x), &mut out)));
    g.finish();
}

fn bench_transpose(c: &mut Criterion) {
    let n = 1024;
    let src: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
    let mut dst = vec![0.0f32; n * n];
    let mut g = c.benchmark_group("transpose/1024x1024_f32");
    g.throughput(Throughput::Bytes((n * n * 4) as u64));
    g.bench_function("tpt_blocked", |b| {
        b.iter(|| tpt_simd_transpose::transpose(black_box(&src), n, n, &mut dst))
    });
    g.bench_function("naive", |b| {
        b.iter(|| {
            let s = black_box(&src);
            for r in 0..n {
                for c in 0..n {
                    dst[c * n + r] = s[r * n + c];
                }
            }
        })
    });
    g.finish();
}

fn bench_ntt(c: &mut Criterion) {
    let q = 998_244_353u32;
    let plan = tpt_simd_ntt::NttPlan::new(q, 3, 4096);
    let base: Vec<u32> = (0..4096u32).map(|i| (i * 7919) % q).collect();
    let mut g = c.benchmark_group("ntt/forward/4096");
    g.throughput(Throughput::Elements(4096));
    g.bench_function("tpt", |b| {
        b.iter_batched(
            || base.clone(),
            |mut v| plan.forward(black_box(&mut v)),
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

fn bench_half(c: &mut Criterion) {
    let n = 65_536;
    let src = f32s(n, 0.9);
    let mut h = vec![tpt_simd_half::f16::ZERO; n];
    let mut g = c.benchmark_group("half/f32_to_f16/64K");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("tpt", |b| {
        b.iter(|| tpt_simd_half::f32_to_f16_slice(black_box(&src), &mut h))
    });
    g.finish();

    let mut q = vec![
        tpt_simd_half::BlockQ8_0 {
            scale: tpt_simd_half::f16::ZERO,
            q: [0; 32],
        };
        n / 32
    ];
    let mut g = c.benchmark_group("half/quantize_q8_0/64K");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("tpt", |b| {
        b.iter(|| tpt_simd_half::quantize_q8_0(black_box(&src), &mut q))
    });
    g.finish();
}

fn bench_pixel(c: &mut Criterion) {
    let px = 1_000_000;
    let rgb: Vec<u8> = (0..px * 3).map(|i| (i % 251) as u8).collect();
    let mut rgba = vec![0u8; px * 4];
    let mut g = c.benchmark_group("pixel/rgb8_to_rgba8/1MP");
    g.throughput(Throughput::Elements(px as u64));
    g.bench_function("tpt", |b| {
        b.iter(|| tpt_simd_pixel::rgb8_to_rgba8(black_box(&rgb), &mut rgba, 255))
    });
    g.finish();

    let mut buf: Vec<u8> = (0..px * 4).map(|i| (i % 253) as u8).collect();
    let mut g = c.benchmark_group("pixel/premultiply_rgba8/1MP");
    g.throughput(Throughput::Elements(px as u64));
    g.bench_function("tpt", |b| {
        b.iter(|| tpt_simd_pixel::premultiply_rgba8(black_box(&mut buf)))
    });
    g.finish();
}

fn bench_audio(c: &mut Criterion) {
    let n = 65_536;
    let src = f32s(n, 0.3);
    let mut dst = f32s(n, 0.7);
    let mut g = c.benchmark_group("audio/mix_into/64K");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("tpt", |b| {
        b.iter(|| tpt_simd_audio::mix_into(&mut dst, black_box(&src), 0.5))
    });
    g.finish();

    let mut g = c.benchmark_group("audio/biquad_lowpass/64K");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("tpt", |b| {
        b.iter_batched(
            || {
                (
                    tpt_simd_audio::Biquad::lowpass(48_000.0, 2_000.0, 0.707),
                    src.clone(),
                )
            },
            |(mut f, mut x)| f.process(black_box(&mut x)),
            BatchSize::SmallInput,
        )
    });
    g.finish();

    let taps = f32s(256, 0.2);
    let mut y = vec![0.0f32; n];
    let mut g = c.benchmark_group("audio/fir_256taps/64K");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("tpt", |b| {
        b.iter(|| tpt_simd_audio::fir(black_box(&taps), black_box(&src), &mut y))
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_distance,
    bench_bits,
    bench_geom,
    bench_dct,
    bench_transpose,
    bench_ntt,
    bench_half,
    bench_pixel,
    bench_audio
);
criterion_main!(benches);
