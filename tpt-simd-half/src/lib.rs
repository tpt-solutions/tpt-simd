//! Half-precision conversions and low-bit quantisation.
//!
//! | item | source | notes |
//! |---|---|---|
//! | `f16`, `bf16` types | re-exported from `half` (MIT OR Apache-2.0) | |
//! | [`f32_to_f16_slice`], [`f16_to_f32_slice`], [`f32_to_bf16_slice`], [`bf16_to_f32_slice`] | built here | length-checked slice conversion |
//! | [`quantize_i8`], [`dequantize_i8`] | built here | symmetric per-tensor int8 |
//! | [`BlockQ8_0`], [`quantize_q8_0`], [`dequantize_q8_0`] | built here | 32-element blocks, f16 scale |
//! | [`BlockQ4_0`], [`quantize_q4_0`], [`dequantize_q4_0`] | built here | 32-element blocks, 4-bit, f16 scale |
//!
//! The block layouts follow the ggml `Q8_0` / `Q4_0` conventions, so blocks
//! written by this crate can be read by tools that use the same layout.
//!
//! ## Length policy
//!
//! Paired slices must match in length, and block functions require the value
//! count to be a multiple of 32; otherwise the function **panics**.
//!
//! ## Example
//!
//! ```
//! use tpt_simd_half::{f16, f32_to_f16_slice};
//!
//! let src = [1.0f32, 0.5, -2.0];
//! let mut dst = [f16::ZERO; 3];
//! f32_to_f16_slice(&src, &mut dst);
//! assert_eq!(dst[2].to_f32(), -2.0);
//! ```
#![no_std]
#![deny(unsafe_code)]
#![warn(missing_docs)]

pub use half::{bf16, f16};

/// Block size shared by the Q8_0 and Q4_0 layouts.
pub const BLOCK: usize = 32;

fn check_len(func: &str, a: usize, b: usize) {
    if a != b {
        panic!("{func}: lengths differ ({a} vs {b})");
    }
}

/// Converts `src` to `f16` (round to nearest even) into `dst`.
///
/// # Panics
///
/// Panics if the slices have different lengths.
pub fn f32_to_f16_slice(src: &[f32], dst: &mut [f16]) {
    check_len("f32_to_f16_slice", src.len(), dst.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = f16::from_f32(s);
    }
}

/// Converts `src` from `f16` into `dst`. Exact.
///
/// # Panics
///
/// Panics if the slices have different lengths.
pub fn f16_to_f32_slice(src: &[f16], dst: &mut [f32]) {
    check_len("f16_to_f32_slice", src.len(), dst.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d = s.to_f32();
    }
}

/// Converts `src` to `bf16` (round to nearest even) into `dst`.
///
/// # Panics
///
/// Panics if the slices have different lengths.
pub fn f32_to_bf16_slice(src: &[f32], dst: &mut [bf16]) {
    check_len("f32_to_bf16_slice", src.len(), dst.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = bf16::from_f32(s);
    }
}

/// Converts `src` from `bf16` into `dst`. Exact.
///
/// # Panics
///
/// Panics if the slices have different lengths.
pub fn bf16_to_f32_slice(src: &[bf16], dst: &mut [f32]) {
    check_len("bf16_to_f32_slice", src.len(), dst.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d = s.to_f32();
    }
}

/// Symmetric int8 quantisation: `q = clamp(round(x / scale), -127, 127)`.
///
/// # Panics
///
/// Panics if the slices have different lengths or `scale` is not positive and
/// finite.
pub fn quantize_i8(src: &[f32], scale: f32, dst: &mut [i8]) {
    check_len("quantize_i8", src.len(), dst.len());
    assert!(
        scale > 0.0 && scale.is_finite(),
        "quantize_i8: scale must be positive and finite"
    );
    let inv = 1.0 / scale;
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = libm::roundf(s * inv).clamp(-127.0, 127.0) as i8;
    }
}

/// Inverse of [`quantize_i8`]: `x = q * scale`.
///
/// # Panics
///
/// Panics if the slices have different lengths.
pub fn dequantize_i8(src: &[i8], scale: f32, dst: &mut [f32]) {
    check_len("dequantize_i8", src.len(), dst.len());
    for (d, &q) in dst.iter_mut().zip(src) {
        *d = f32::from(q) * scale;
    }
}

/// 32 int8 values with one `f16` scale: `x = q * scale`.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct BlockQ8_0 {
    /// Scale, `max|x| / 127` rounded to `f16`.
    pub scale: f16,
    /// Quantised values in `-127..=127`.
    pub q: [i8; BLOCK],
}

/// 32 4-bit values with one `f16` scale: `x = (q - 8) * scale`.
///
/// Element `i` is in the low nibble of `qs[i]` for `i < 16`, and element
/// `i + 16` is in the high nibble of `qs[i]`.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct BlockQ4_0 {
    /// Scale, `v / -8` where `v` is the signed value of largest magnitude, rounded to `f16`.
    pub scale: f16,
    /// Packed 4-bit values.
    pub qs: [u8; BLOCK / 2],
}

/// Quantises `src` into Q8_0 blocks. `src.len()` must be `32 * dst.len()`.
///
/// # Panics
///
/// Panics if the length relation does not hold.
pub fn quantize_q8_0(src: &[f32], dst: &mut [BlockQ8_0]) {
    check_len("quantize_q8_0", src.len(), dst.len() * BLOCK);
    for (block, chunk) in dst.iter_mut().zip(src.chunks_exact(BLOCK)) {
        let amax = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let scale = f16::from_f32(amax / 127.0);
        let d = scale.to_f32();
        block.scale = scale;
        if d == 0.0 {
            block.q = [0; BLOCK];
            continue;
        }
        let inv = 1.0 / d;
        for (q, &x) in block.q.iter_mut().zip(chunk) {
            *q = libm::roundf(x * inv).clamp(-127.0, 127.0) as i8;
        }
    }
}

/// Dequantises Q8_0 blocks into `dst`. `dst.len()` must be `32 * src.len()`.
///
/// # Panics
///
/// Panics if the length relation does not hold.
pub fn dequantize_q8_0(src: &[BlockQ8_0], dst: &mut [f32]) {
    check_len("dequantize_q8_0", dst.len(), src.len() * BLOCK);
    for (block, out) in src.iter().zip(dst.chunks_exact_mut(BLOCK)) {
        let d = block.scale.to_f32();
        for (o, &q) in out.iter_mut().zip(&block.q) {
            *o = f32::from(q) * d;
        }
    }
}

/// Quantises `src` into Q4_0 blocks. `src.len()` must be `32 * dst.len()`.
///
/// # Panics
///
/// Panics if the length relation does not hold.
pub fn quantize_q4_0(src: &[f32], dst: &mut [BlockQ4_0]) {
    check_len("quantize_q4_0", src.len(), dst.len() * BLOCK);
    for (block, chunk) in dst.iter_mut().zip(src.chunks_exact(BLOCK)) {
        // Signed value with the largest magnitude; the scale maps it to -8.
        let mut max = 0.0f32;
        for &x in chunk {
            if x.abs() > max.abs() {
                max = x;
            }
        }
        let scale = f16::from_f32(max / -8.0);
        let d = scale.to_f32();
        block.scale = scale;
        let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
        let mut q = [0u8; BLOCK];
        for (qi, &x) in q.iter_mut().zip(chunk) {
            // Truncating (x * inv + 8.5) rounds to nearest; the clamp covers x * inv == -8.
            *qi = ((x * inv + 8.5) as i32).clamp(0, 15) as u8;
        }
        for i in 0..BLOCK / 2 {
            block.qs[i] = q[i] | (q[i + BLOCK / 2] << 4);
        }
    }
}

/// Dequantises Q4_0 blocks into `dst`. `dst.len()` must be `32 * src.len()`.
///
/// # Panics
///
/// Panics if the length relation does not hold.
pub fn dequantize_q4_0(src: &[BlockQ4_0], dst: &mut [f32]) {
    check_len("dequantize_q4_0", dst.len(), src.len() * BLOCK);
    for (block, out) in src.iter().zip(dst.chunks_exact_mut(BLOCK)) {
        let d = block.scale.to_f32();
        for i in 0..BLOCK / 2 {
            let byte = block.qs[i];
            out[i] = (f32::from(byte & 0x0F) - 8.0) * d;
            out[i + BLOCK / 2] = (f32::from(byte >> 4) - 8.0) * d;
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec;
    use std::vec::Vec;

    fn data(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| libm::sinf(i as f32 * 0.7) * 3.0 + ((i % 5) as f32 - 2.0) * 0.1)
            .collect()
    }

    #[test]
    fn f16_and_bf16_slices_round_trip_exactly_representable_values() {
        let src = [0.0f32, 1.0, -2.5, 0.5, 65504.0];
        let mut h = [f16::ZERO; 5];
        f32_to_f16_slice(&src, &mut h);
        let mut back = [0.0f32; 5];
        f16_to_f32_slice(&h, &mut back);
        assert_eq!(back, src);
        // bf16 has 8 mantissa bits, so 65504 rounds to 65536; use values it holds exactly.
        let small = [0.0f32, 1.0, -2.5, 0.5, 256.0];
        let mut b = [bf16::ZERO; 5];
        f32_to_bf16_slice(&small, &mut b);
        bf16_to_f32_slice(&b, &mut back);
        assert_eq!(back, small);
        f32_to_bf16_slice(&src, &mut b);
        bf16_to_f32_slice(&b, &mut back);
        assert_eq!(back[4], 65536.0);
    }

    #[test]
    fn f16_rounding_error_is_bounded() {
        let src = data(64);
        let mut h = vec![f16::ZERO; 64];
        f32_to_f16_slice(&src, &mut h);
        for (x, y) in src.iter().zip(&h) {
            let e = (x - y.to_f32()).abs();
            assert!(e <= x.abs() * 0.001 + 1e-4, "x={x} err={e}");
        }
    }

    #[test]
    fn i8_quantisation_round_trip() {
        let src = [0.0f32, 1.0, -1.0, 100.0, -100.0, 0.49];
        let scale = 100.0 / 127.0;
        let mut q = [0i8; 6];
        quantize_i8(&src, scale, &mut q);
        // 0.49 / 0.787 = 0.62, which rounds to 1.
        assert_eq!(q, [0, 1, -1, 127, -127, 1]);
        let mut back = [0.0f32; 6];
        dequantize_i8(&q, scale, &mut back);
        for (x, y) in src.iter().zip(&back) {
            assert!((x - y).abs() <= scale / 2.0 + 1e-4);
        }
    }

    #[test]
    fn q8_0_round_trip_within_half_step() {
        let src = data(96);
        let mut blocks = vec![
            BlockQ8_0 {
                scale: f16::ZERO,
                q: [0; BLOCK]
            };
            3
        ];
        quantize_q8_0(&src, &mut blocks);
        let mut back = vec![0.0f32; 96];
        dequantize_q8_0(&blocks, &mut back);
        for (b, chunk) in src.chunks(BLOCK).enumerate() {
            let d = blocks[b].scale.to_f32();
            let amax = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            for (i, &x) in chunk.iter().enumerate() {
                let e = (x - back[b * BLOCK + i]).abs();
                assert!(
                    e <= 0.5 * d + amax * 2e-3,
                    "block {b} i {i}: err {e}, step {d}"
                );
            }
        }
    }

    #[test]
    fn q8_0_zero_block_is_zero() {
        let src = [0.0f32; BLOCK];
        let mut blocks = [BlockQ8_0 {
            scale: f16::ZERO,
            q: [1; BLOCK],
        }];
        quantize_q8_0(&src, &mut blocks);
        assert_eq!(blocks[0].q, [0; BLOCK]);
        assert_eq!(blocks[0].scale.to_f32(), 0.0);
    }

    #[test]
    fn q4_0_round_trip_within_one_step() {
        let src = data(64);
        let mut blocks = vec![
            BlockQ4_0 {
                scale: f16::ZERO,
                qs: [0; 16]
            };
            2
        ];
        quantize_q4_0(&src, &mut blocks);
        let mut back = vec![0.0f32; 64];
        dequantize_q4_0(&blocks, &mut back);
        for (b, chunk) in src.chunks(BLOCK).enumerate() {
            let d = blocks[b].scale.to_f32().abs();
            for (i, &x) in chunk.iter().enumerate() {
                let e = (x - back[b * BLOCK + i]).abs();
                assert!(e <= d + 1e-3, "block {b} i {i}: err {e}, step {d}");
            }
        }
    }

    #[test]
    fn q4_0_known_layout() {
        // The largest-magnitude value sets the scale so it maps to nibble 0.
        let mut src = [0.0f32; BLOCK];
        src[0] = 8.0; // largest magnitude, so d = 8 / -8 = -1
        src[16] = -8.0;
        let mut blocks = [BlockQ4_0 {
            scale: f16::ZERO,
            qs: [0; 16],
        }];
        quantize_q4_0(&src, &mut blocks);
        assert_eq!(blocks[0].scale.to_f32(), -1.0);
        assert_eq!(blocks[0].qs[0] & 0x0F, 0); // element 0 -> q = 0 (x = -8 d)
        // Element 16 is -8 * d = 8, which is past the top of the range: it saturates to 15.
        assert_eq!(blocks[0].qs[0] >> 4, 15);
        let mut back = [0.0f32; BLOCK];
        dequantize_q4_0(&blocks, &mut back);
        assert_eq!(back[0], 8.0);
        assert_eq!(back[16], -7.0);
    }

    #[test]
    #[should_panic(expected = "quantize_q8_0: lengths differ (33 vs 32)")]
    fn q8_0_requires_whole_blocks() {
        let mut blocks = [BlockQ8_0 {
            scale: f16::ZERO,
            q: [0; BLOCK],
        }];
        quantize_q8_0(&[0.0; 33], &mut blocks);
    }
}
