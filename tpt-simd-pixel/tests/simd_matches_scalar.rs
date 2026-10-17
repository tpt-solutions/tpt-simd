//! The public pixel kernels must match the scalar formulas exactly. The
//! premultiply check is exhaustive over every (channel, alpha) pair, and the
//! f32 check compares bit patterns, so it runs against whichever path this
//! build compiled.

use proptest::prelude::*;
use tpt_simd_pixel::{premultiply_rgba8, u8_to_unit_f32};

fn scalar_premul(c: u8, a: u8) -> u8 {
    let t = u32::from(c) * u32::from(a) + 128;
    ((t + (t >> 8)) >> 8) as u8
}

#[test]
fn premultiply_matches_formula_for_every_channel_and_alpha() {
    // Each pixel is (c, c, c, a), so the same channel value goes through each
    // lane and alpha is the second input to the formula.
    let mut buf = Vec::with_capacity(256 * 256 * 4);
    for a in 0..=255u8 {
        for c in 0..=255u8 {
            buf.extend_from_slice(&[c, c, c, a]);
        }
    }
    let mut got = buf.clone();
    premultiply_rgba8(&mut got);
    for (i, px) in got.chunks_exact(4).enumerate() {
        let (c, a) = (buf[i * 4], buf[i * 4 + 3]);
        assert_eq!(px[0], scalar_premul(c, a), "c={c} a={a}");
        assert_eq!(px[1], scalar_premul(c, a), "c={c} a={a}");
        assert_eq!(px[2], scalar_premul(c, a), "c={c} a={a}");
        assert_eq!(px[3], a, "alpha must pass through");
    }
}

#[test]
fn premultiply_handles_every_pixel_count_tail() {
    for n in 0..=17usize {
        let src: Vec<u8> = (0..n * 4).map(|i| (i * 37 + 11) as u8).collect();
        let mut got = src.clone();
        premultiply_rgba8(&mut got);
        let mut want = src.clone();
        for px in want.chunks_exact_mut(4) {
            let a = px[3];
            for c in &mut px[..3] {
                *c = scalar_premul(*c, a);
            }
        }
        assert_eq!(got, want, "n={n} pixels");
    }
}

proptest! {
    #[test]
    fn u8_to_unit_bits_match_scalar(src in prop::collection::vec(any::<u8>(), 0..300)) {
        let mut got = vec![0.0f32; src.len()];
        u8_to_unit_f32(&src, &mut got);
        for (&s, &g) in src.iter().zip(&got) {
            prop_assert_eq!(g.to_bits(), (f32::from(s) / 255.0).to_bits());
        }
    }

    #[test]
    fn premultiply_matches_scalar_random(buf in prop::collection::vec(any::<u8>(), 0..400)) {
        let n = buf.len() / 4 * 4;
        let src = &buf[..n];
        let mut got = src.to_vec();
        premultiply_rgba8(&mut got);
        let mut want = src.to_vec();
        for px in want.chunks_exact_mut(4) {
            let a = px[3];
            for c in &mut px[..3] {
                *c = scalar_premul(*c, a);
            }
        }
        prop_assert_eq!(got, want);
    }
}
