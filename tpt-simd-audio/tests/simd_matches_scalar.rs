//! `peak` and `fir` must match their scalar definitions bit for bit. The
//! references are written out in full here. The inputs include NaN, which
//! `f32::max` ignores, plus empty and short buffers and every tap count up to 20.

use tpt_simd_audio::{fir, peak};

fn peak_ref(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |m, &s| m.max(s.abs()))
}

fn fir_ref(taps: &[f32], x: &[f32]) -> Vec<f32> {
    (0..x.len())
        .map(|n| {
            let mut acc = 0.0f32;
            for (k, &t) in taps.iter().enumerate().take(n + 1) {
                acc += t * x[n - k];
            }
            acc
        })
        .collect()
}

fn samples(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            if i % 37 == 5 {
                f32::NAN
            } else {
                ((*seed % 4001) as f32 - 2000.0) / 333.0
            }
        })
        .collect()
}

#[test]
fn peak_matches_reference() {
    let mut seed = 0x7F4A_7C15_9E37_79B9u64;
    for n in 0..=130 {
        let x = samples(n, &mut seed);
        assert_eq!(peak(&x).to_bits(), peak_ref(&x).to_bits(), "n={n}");
    }
    assert_eq!(
        peak(&[f32::NAN, -0.0, f32::NAN]).to_bits(),
        0.0f32.to_bits()
    );
    assert_eq!(
        peak(&[-7.5, 2.0, -9.25, 0.0, 1.0, 1.0, 1.0, -0.0, 3.0]),
        9.25
    );
}

#[test]
fn fir_matches_reference() {
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    for taps_len in 0..=20usize {
        let taps: Vec<f32> = (0..taps_len)
            .map(|k| ((k as f32) * 0.37 - 2.0).sin())
            .collect();
        for n in [0usize, 1, 5, 8, 9, 16, 23, 40, 64, 100] {
            let x = samples(n, &mut seed);
            let mut y = vec![0.0f32; n];
            fir(&taps, &x, &mut y);
            let want = fir_ref(&taps, &x);
            for i in 0..n {
                assert_eq!(
                    y[i].to_bits(),
                    want[i].to_bits(),
                    "taps={taps_len} n={n} i={i}"
                );
            }
        }
    }
}
