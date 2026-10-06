//! Interleaved best-of-N timing of the SpMV row strategies (robust to a
//! noisy machine, unlike a single criterion run): `cargo run --release
//! --example strategies` (use `RUSTFLAGS="-C target-cpu=native"`).
//!
//! Reports the minimum and median time per stored entry over many rounds
//! in which the variants are run back to back.

use std::hint::black_box;
use std::time::Instant;
use tpt_simd_core::{I32x8, Simd};
use tpt_simd_gather::gather_f32;
use tpt_simd_sparse::*;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn random_matrix(n: usize, per_row: usize) -> CsrMatrix<f32> {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut t = Vec::with_capacity(n * per_row);
    for r in 0..n {
        for _ in 0..per_row {
            let c = (rng.next() >> 16) as usize % n;
            let v = ((rng.next() >> 40) as f32 / 8_388_608.0) - 1.0;
            t.push((r, c, v));
        }
    }
    CsrMatrix::from_triplets(n, n, &t).unwrap()
}

/// Hardware-gather row kernel (f32): `vgatherdps` + vector multiply-add.
fn spmv_csr_gather(a: &CsrView<'_, f32>, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), a.ncols());
    assert_eq!(y.len(), a.nrows());
    for (r, w) in a.indptr().windows(2).enumerate() {
        let (idx, d) = (&a.indices()[w[0]..w[1]], &a.data()[w[0]..w[1]]);
        let mut acc = Simd::<f32, 8>::zero();
        let ic = idx.chunks_exact(8);
        let dc = d.chunks_exact(8);
        let (it, dt) = (ic.remainder(), dc.remainder());
        for (i, v) in ic.zip(dc) {
            let mut lanes = [0i32; 8];
            for j in 0..8 {
                lanes[j] = i[j] as i32;
            }
            // SAFETY: the view is validated, so every index < x.len().
            let g = unsafe { gather_f32(x.as_ptr(), I32x8::from_array(lanes)) };
            acc += Simd::from_slice(v) * g;
        }
        let l = acc.to_array();
        let mut t = ((l[0] + l[4]) + (l[2] + l[6])) + ((l[1] + l[5]) + (l[3] + l[7]));
        for (&i, &v) in it.iter().zip(dt) {
            t += v * x[i as usize];
        }
        y[r] = t;
    }
}

fn time_variants(title: &str, a: &CsrMatrix<f32>) {
    let v = a.view();
    let x: Vec<f32> = (0..a.ncols())
        .map(|i| ((i * 7919 % 2003) as f32 - 1000.0) * 0.001)
        .collect();
    let mut y = vec![0.0f32; a.nrows()];
    let csc = a.to_csc();
    type Run<'a> = Box<dyn FnMut(&mut [f32]) + 'a>;
    let mut variants: Vec<(&str, Run<'_>)> = vec![
        (
            "reference (safe idx)",
            Box::new(|y| reference::spmv_csr(1.0, &v, black_box(&x), 0.0, y)),
        ),
        (
            "scalar-1acc",
            Box::new(|y| spmv_csr_with(RowStrategy::Scalar, 1.0, &v, black_box(&x), 0.0, y)),
        ),
        (
            "lanes4",
            Box::new(|y| spmv_csr_with(RowStrategy::Lanes4, 1.0, &v, black_box(&x), 0.0, y)),
        ),
        (
            "lanes8",
            Box::new(|y| spmv_csr_with(RowStrategy::Lanes8, 1.0, &v, black_box(&x), 0.0, y)),
        ),
        (
            "hybrid(8)",
            Box::new(|y| spmv_csr_with(RowStrategy::Hybrid, 1.0, &v, black_box(&x), 0.0, y)),
        ),
        (
            "hw-gather",
            Box::new(|y| spmv_csr_gather(&v, black_box(&x), y)),
        ),
        (
            "csr_t scatter",
            Box::new(|y| spmv_csr_t(1.0, &v, black_box(&x), 0.0, y)),
        ),
        (
            "reference csr_t",
            Box::new(|y| reference::spmv_csr_t(1.0, &v, black_box(&x), 0.0, y)),
        ),
        (
            "csc scatter",
            Box::new(|y| spmv_csc(1.0, &csc.view(), black_box(&x), 0.0, y)),
        ),
        (
            "csc_t gather",
            Box::new(|y| spmv_csc_t(1.0, &csc.view(), black_box(&x), 0.0, y)),
        ),
    ];
    let rounds = 150;
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
    for round in 0..rounds + 10 {
        for (k, (_, f)) in variants.iter_mut().enumerate() {
            let t = Instant::now();
            f(&mut y);
            black_box(&y);
            let dt = t.elapsed().as_secs_f64();
            if round >= 10 {
                times[k].push(dt);
            }
        }
    }
    println!("{title}  ({} rows, {} nnz)", a.nrows(), v.nnz());
    for ((name, _), t) in variants.iter().zip(&mut times) {
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (min, med) = (t[0], t[t.len() / 2]);
        let n = v.nnz() as f64;
        println!(
            "  {name:<22} min {:>7.3} ns/nnz  median {:>7.3} ns/nnz  ({:.2} Gnnz/s best)",
            min / n * 1e9,
            med / n * 1e9,
            n / min / 1e9
        );
    }
}

fn main() {
    time_variants("poisson 256x256", &CsrMatrix::<f32>::poisson_2d(256));
    time_variants("random 65536, 32 nnz/row", &random_matrix(65_536, 32));
    time_variants("random 65536, 12 nnz/row", &random_matrix(65_536, 12));
    time_variants("random 8192, 128 nnz/row", &random_matrix(8_192, 128));
}
