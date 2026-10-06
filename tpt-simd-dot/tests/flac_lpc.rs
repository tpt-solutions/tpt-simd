//! Realistic FLAC-style LPC residual computation using `dot_product_i16`.
//!
//! FLAC (subset) predicts `x[n]` from the previous `order` samples with
//! quantised integer coefficients: `pred = (sum_j c[j] * x[n-1-j]) >> shift`,
//! `residual = x[n] - pred`. The decoder does the inverse. With the
//! coefficients stored reversed, the inner sum is one dot product over a
//! window of the signal.

use tpt_simd_dot::dot_product_i16;

const ORDER: usize = 32;
const SHIFT: u32 = 12;

/// Deterministic pseudo-audio: sum of two integer "sines" plus an LCG noise.
fn signal(n: usize) -> Vec<i16> {
    let mut state = 0x1234_5678u32;
    (0..n)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = ((state >> 24) as i32) - 128;
            let tri = ((i * 37) % 2000) as i32 - 1000;
            let saw = ((i * 5) % 600) as i32 - 300;
            (tri * 8 + saw * 8 + noise).clamp(-32768, 32767) as i16
        })
        .collect()
}

/// Quantised coefficients (q12): small, stable-ish, alternating sign.
fn coefficients() -> [i16; ORDER] {
    core::array::from_fn(|j| {
        let base = 4096 / (j as i32 + 2);
        (if j % 2 == 0 { base } else { -base / 2 }) as i16
    })
}

fn residual_scalar(x: &[i16], c: &[i16; ORDER]) -> Vec<i32> {
    (ORDER..x.len())
        .map(|n| {
            let mut s = 0i64;
            for j in 0..ORDER {
                s += c[j] as i64 * x[n - 1 - j] as i64;
            }
            x[n] as i32 - (s >> SHIFT) as i32
        })
        .collect()
}

fn residual_simd(x: &[i16], c: &[i16; ORDER]) -> Vec<i32> {
    // Reverse coefficients so the window x[n-ORDER..n] lines up with them.
    let mut rev = *c;
    rev.reverse();
    (ORDER..x.len())
        .map(|n| {
            let sum = dot_product_i16(&rev, &x[n - ORDER..n]);
            x[n] as i32 - (sum >> SHIFT)
        })
        .collect()
}

#[test]
fn order32_residual_matches_scalar() {
    let x = signal(4096);
    let c = coefficients();
    let a = residual_simd(&x, &c);
    let b = residual_scalar(&x, &c);
    assert_eq!(a.len(), 4096 - ORDER);
    assert_eq!(a, b);
}

#[test]
fn decoder_reconstructs_signal() {
    let x = signal(1024);
    let c = coefficients();
    let res = residual_simd(&x, &c);
    // Decode: warm-up samples are verbatim, the rest are pred + residual.
    let mut rev = c;
    rev.reverse();
    let mut y: Vec<i16> = x[..ORDER].to_vec();
    for (k, &r) in res.iter().enumerate() {
        let n = ORDER + k;
        let sum = dot_product_i16(&rev, &y[n - ORDER..n]);
        y.push((r + (sum >> SHIFT)) as i16);
    }
    assert_eq!(y, x);
}

#[test]
fn appendix_b3_example() {
    let a: Vec<i16> = (0..256).collect();
    let b: Vec<i16> = (0..256).collect();
    let result = dot_product_i16(&a, &b);
    assert_eq!(result, 5_559_680);
}
