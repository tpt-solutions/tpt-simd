//! Property tests: half-precision conversions against the `half` crate, and quantisation bounds.

use half::{bf16, f16};
use proptest::prelude::*;
use tpt_simd_half::{
    BlockQ4_0, BlockQ8_0, bf16_to_f32_slice, dequantize_i8, dequantize_q4_0, dequantize_q8_0,
    f16_to_f32_slice, f32_to_bf16_slice, f32_to_f16_slice, quantize_i8, quantize_q4_0,
    quantize_q8_0,
};

proptest! {
    #[test]
    fn f16_conversion_matches_half_crate(src in prop::collection::vec(any::<f32>(), 0..300)) {
        let mut dst = vec![f16::ZERO; src.len()];
        f32_to_f16_slice(&src, &mut dst);
        for (&x, &h) in src.iter().zip(&dst) {
            let want = f16::from_f32(x);
            prop_assert!(h.to_bits() == want.to_bits() || (h.is_nan() && want.is_nan()));
        }
        let mut back = vec![0f32; src.len()];
        f16_to_f32_slice(&dst, &mut back);
        for (&h, &x) in dst.iter().zip(&back) {
            let want = h.to_f32();
            prop_assert!(x.to_bits() == want.to_bits() || (x.is_nan() && want.is_nan()));
        }
    }

    #[test]
    fn bf16_conversion_matches_half_crate(src in prop::collection::vec(any::<f32>(), 0..300)) {
        let mut dst = vec![bf16::ZERO; src.len()];
        f32_to_bf16_slice(&src, &mut dst);
        for (&x, &b) in src.iter().zip(&dst) {
            let want = bf16::from_f32(x);
            prop_assert!(b.to_bits() == want.to_bits() || (b.is_nan() && want.is_nan()));
        }
        let mut back = vec![0f32; src.len()];
        bf16_to_f32_slice(&dst, &mut back);
        for (&b, &x) in dst.iter().zip(&back) {
            let want = b.to_f32();
            prop_assert!(x.to_bits() == want.to_bits() || (x.is_nan() && want.is_nan()));
        }
    }

    #[test]
    fn int8_quantisation_error_is_half_a_step(
        scale in 0.01f32..10.0,
        src in prop::collection::vec(-1.0f32..1.0, 0..300),
    ) {
        // Inputs in [-1, 1] stay inside the +-127 * scale range for scale >= 1/127.
        let mut q = vec![0i8; src.len()];
        quantize_i8(&src, scale, &mut q);
        let mut back = vec![0f32; src.len()];
        dequantize_i8(&q, scale, &mut back);
        for (&x, &y) in src.iter().zip(&back) {
            prop_assert!((x - y).abs() <= 0.5 * scale + 1e-6);
        }
    }

    #[test]
    fn q8_0_round_trip_error_is_half_a_step(src in prop::collection::vec(-100.0f32..100.0, 32..=256).prop_map(|mut v| { v.truncate(v.len() / 32 * 32); v })) {
        let blocks = src.len() / 32;
        let mut packed = vec![BlockQ8_0 { scale: f16::ZERO, q: [0; 32] }; blocks];
        quantize_q8_0(&src, &mut packed);
        let mut back = vec![0f32; src.len()];
        dequantize_q8_0(&packed, &mut back);
        for b in 0..blocks {
            let s = packed[b].scale.to_f32();
            for i in b * 32..(b + 1) * 32 {
                // f16 rounding of the scale adds a relative error on top of the half step.
                prop_assert!((src[i] - back[i]).abs() <= 0.5 * s.abs() + 1e-3 * src[i].abs().max(1.0));
            }
        }
    }

    #[test]
    fn q4_0_round_trip_error_is_bounded(src in prop::collection::vec(-100.0f32..100.0, 32..=256).prop_map(|mut v| { v.truncate(v.len() / 32 * 32); v })) {
        let blocks = src.len() / 32;
        let mut packed = vec![BlockQ4_0 { scale: f16::ZERO, qs: [0; 16] }; blocks];
        quantize_q4_0(&src, &mut packed);
        let mut back = vec![0f32; src.len()];
        dequantize_q4_0(&packed, &mut back);
        for b in 0..blocks {
            let s = packed[b].scale.to_f32();
            for i in b * 32..(b + 1) * 32 {
                // 4-bit levels are coarse, and the top level clamps, so the bound is 1.5 steps.
                prop_assert!((src[i] - back[i]).abs() <= 1.5 * s.abs() + 1e-3 * src[i].abs().max(1.0));
            }
        }
    }
}
