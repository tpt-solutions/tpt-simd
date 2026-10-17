//! The slice conversions must return the same bits as `half`'s per-element
//! conversions, NaN payloads included. This covers every f16 pattern and a
//! large sample of f32 patterns, with lengths that exercise the 8-wide body
//! and the tail. It runs against whichever path the build compiled: F16C
//! (compile-time or `runtime-dispatch`) or the software `half` loop.

use tpt_simd_half::{f16, f16_to_f32_slice, f32_to_f16_slice};

#[test]
fn f16_to_f32_matches_half_for_every_pattern() {
    let src: Vec<f16> = (0..=u16::MAX).map(f16::from_bits).collect();
    let mut got = vec![0.0f32; src.len()];
    f16_to_f32_slice(&src, &mut got);
    for (i, (&h, &g)) in src.iter().zip(&got).enumerate() {
        assert_eq!(g.to_bits(), h.to_f32().to_bits(), "f16 pattern {i:#06x}");
    }
}

#[test]
fn f32_to_f16_matches_half_on_many_patterns() {
    // A fixed-seed xorshift walk over all 32-bit patterns (NaNs and subnormals
    // included), plus the boundary values around the f16 range.
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut src: Vec<f32> = (0..200_000)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            f32::from_bits(seed as u32)
        })
        .collect();
    src.extend([
        0.0,
        -0.0,
        65504.0,
        65520.0,
        65519.0,
        1.0e-8,
        5.960_464_5e-8,
        2.980_232_2e-8,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
    ]);
    let mut got = vec![f16::ZERO; src.len()];
    f32_to_f16_slice(&src, &mut got);
    for (i, (&s, &g)) in src.iter().zip(&got).enumerate() {
        assert_eq!(
            g.to_bits(),
            f16::from_f32(s).to_bits(),
            "index {i}, input {:#010x}",
            s.to_bits()
        );
    }
}

#[test]
fn every_length_up_to_twenty_matches_half() {
    // Lengths 0..=20 cover an empty slice, pure tails, and bodies with tails.
    for n in 0..=20usize {
        let src: Vec<f32> = (0..n).map(|i| (i as f32 - 7.5) * 1.37).collect();
        let mut h = vec![f16::ZERO; n];
        f32_to_f16_slice(&src, &mut h);
        let mut back = vec![0.0f32; n];
        f16_to_f32_slice(&h, &mut back);
        for i in 0..n {
            assert_eq!(
                h[i].to_bits(),
                f16::from_f32(src[i]).to_bits(),
                "n={n} i={i}"
            );
            assert_eq!(back[i].to_bits(), h[i].to_f32().to_bits(), "n={n} i={i}");
        }
    }
}
