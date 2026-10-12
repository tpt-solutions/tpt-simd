//! tpt-simd-window cosine-sum windows vs apodize (dev-dep oracle), both writing into a
//! preallocated f32 buffer so the timing excludes allocation.
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn bench(c: &mut Criterion) {
    // The two implementations must agree before their speeds are compared.
    for &n in &[64usize, 1024, 16384] {
        let mut ours = vec![0f32; n];
        tpt_simd_window::hamming_window_into_f32(&mut ours);
        let theirs: Vec<f32> = apodize::hamming_iter(n).map(|w| w as f32).collect();
        let worst = ours
            .iter()
            .zip(&theirs)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst < 1e-5,
            "hamming n={n} disagrees with apodize by {worst}"
        );
    }

    for &n in &[64usize, 1024, 16384] {
        let mut out = vec![0f32; n];

        let mut g = c.benchmark_group(format!("window/hamming/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::new("tpt", n), |b| {
            b.iter(|| tpt_simd_window::hamming_window_into_f32(black_box(&mut out)))
        });
        g.bench_function(BenchmarkId::new("apodize", n), |b| {
            b.iter(|| {
                for (o, w) in black_box(&mut out).iter_mut().zip(apodize::hamming_iter(n)) {
                    *o = w as f32;
                }
            })
        });
        g.finish();

        let mut g = c.benchmark_group(format!("window/hanning/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::new("tpt", n), |b| {
            b.iter(|| tpt_simd_window::hanning_window_into_f32(black_box(&mut out)))
        });
        g.bench_function(BenchmarkId::new("apodize", n), |b| {
            b.iter(|| {
                for (o, w) in black_box(&mut out).iter_mut().zip(apodize::hanning_iter(n)) {
                    *o = w as f32;
                }
            })
        });
        g.finish();

        let mut g = c.benchmark_group(format!("window/blackman/{n}"));
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(BenchmarkId::new("tpt", n), |b| {
            b.iter(|| tpt_simd_window::blackman_window_into_f32(black_box(&mut out)))
        });
        g.bench_function(BenchmarkId::new("apodize", n), |b| {
            b.iter(|| {
                for (o, w) in black_box(&mut out)
                    .iter_mut()
                    .zip(apodize::blackman_iter(n))
                {
                    *o = w as f32;
                }
            })
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
