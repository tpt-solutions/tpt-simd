//! `Dct::forward` and `Mdct::forward` must return the same bits as the scalar
//! row sums. The reference rebuilds the cosine tables with the crate's formulas
//! and sums each row with `Iterator::sum`, so the summation order and the
//! sign of zero are the scalar code's.

use core::f64::consts::PI;
use tpt_simd_dct::{Dct, Mdct};

fn dct_table(n: usize) -> Vec<f32> {
    let mut table = vec![0.0f32; n * n];
    for k in 0..n {
        let s = if k == 0 {
            (1.0 / n as f64).sqrt()
        } else {
            (2.0 / n as f64).sqrt()
        };
        for i in 0..n {
            let angle = PI / n as f64 * (i as f64 + 0.5) * k as f64;
            table[k * n + i] = (s * angle.cos()) as f32;
        }
    }
    table
}

fn mdct_table(n: usize) -> Vec<f32> {
    let len = 2 * n;
    let mut table = vec![0.0f32; n * len];
    let half = n as f64 / 2.0;
    for k in 0..n {
        for i in 0..len {
            let angle = PI / n as f64 * (i as f64 + 0.5 + half) * (k as f64 + 0.5);
            table[k * len + i] = angle.cos() as f32;
        }
    }
    table
}

fn reference(table: &[f32], m: usize, n_out: usize, x: &[f32]) -> Vec<f32> {
    (0..n_out)
        .map(|k| {
            table[k * m..(k + 1) * m]
                .iter()
                .zip(x)
                .map(|(&c, &v)| c * v)
                .sum()
        })
        .collect()
}

fn signal(len: usize, seed: &mut u64) -> Vec<f32> {
    (0..len)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed % 2001) as f32 - 1000.0) / 250.0
        })
        .collect()
}

#[test]
fn dct_forward_bits_match_scalar() {
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    for n in 1..=40usize {
        let dct = Dct::new(n);
        let table = dct_table(n);
        let mut inputs = vec![signal(n, &mut seed), vec![0.0f32; n]];
        inputs.push(vec![-0.0f32; n]);
        for x in &inputs {
            let mut got = vec![0.0f32; n];
            dct.forward(x, &mut got);
            let want = reference(&table, n, n, x);
            for k in 0..n {
                assert_eq!(got[k].to_bits(), want[k].to_bits(), "dct n={n} k={k}");
            }
        }
    }
}

#[test]
fn mdct_forward_bits_match_scalar() {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for n in [1usize, 2, 5, 8, 9, 16, 17, 31, 32, 40] {
        let mdct = Mdct::new(n);
        let table = mdct_table(n);
        for _ in 0..4 {
            let x = signal(2 * n, &mut seed);
            let mut got = vec![0.0f32; n];
            mdct.forward(&x, &mut got);
            let want = reference(&table, 2 * n, n, &x);
            for k in 0..n {
                assert_eq!(got[k].to_bits(), want[k].to_bits(), "mdct n={n} k={k}");
            }
        }
        let zeros = vec![0.0f32; 2 * n];
        let mut got = vec![0.0f32; n];
        mdct.forward(&zeros, &mut got);
        let want = reference(&table, 2 * n, n, &zeros);
        for k in 0..n {
            assert_eq!(got[k].to_bits(), want[k].to_bits(), "mdct zero n={n} k={k}");
        }
    }
}
