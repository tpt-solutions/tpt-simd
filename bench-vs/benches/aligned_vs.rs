//! tpt-simd-aligned buffers vs aligned-vec (dev-dep oracle): allocate and fill a 32-byte aligned
//! f32 buffer, and copy one from a slice. Both sides return the buffer, so the timing includes
//! the allocation and the copy, and the buffer is dropped inside the timed loop.
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn bench(c: &mut Criterion) {
    for &n in &[64usize, 4096, 262144] {
        let src: Vec<f32> = (0..n).map(|i| i as f32).collect();

        let mut g = c.benchmark_group(format!("aligned/zeroed/{n}"));
        g.throughput(Throughput::Bytes((n * 4) as u64));
        g.bench_function(BenchmarkId::new("tpt", n), |b| {
            b.iter(|| tpt_simd_aligned::AlignedBuf::<f32, 32>::zeroed(black_box(n)))
        });
        g.bench_function(BenchmarkId::new("aligned-vec", n), |b| {
            b.iter(|| {
                aligned_vec::AVec::<f32, aligned_vec::ConstAlign<32>>::__from_elem(
                    32,
                    0.0,
                    black_box(n),
                )
            })
        });
        g.finish();

        let mut g = c.benchmark_group(format!("aligned/from_slice/{n}"));
        g.throughput(Throughput::Bytes((n * 4) as u64));
        g.bench_function(BenchmarkId::new("tpt", n), |b| {
            b.iter(|| tpt_simd_aligned::AlignedBuf::<f32, 32>::from_slice(black_box(&src)))
        });
        g.bench_function(BenchmarkId::new("aligned-vec", n), |b| {
            b.iter(|| {
                aligned_vec::AVec::<f32, aligned_vec::ConstAlign<32>>::from_slice(
                    32,
                    black_box(&src),
                )
            })
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
