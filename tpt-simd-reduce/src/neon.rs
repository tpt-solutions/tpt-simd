//! aarch64 NEON block sums, feeding the same accumulator layout as the AVX-512 hooks.
//!
//! Each accumulator lane sees the same elements in the same order as the portable loop
//! (separate `vaddq`, no fusing), so the blocks are bit-identical to the portable result.

use core::arch::aarch64::*;
use tpt_simd_core::Simd;

/// Accumulators for the full 32-element blocks of `xs` (f32). Accumulator `k` holds elements
/// `8k..8k+8` of each block, held as two 4-lane registers. Returns them with the number of
/// elements consumed.
///
/// # Safety
/// NEON is part of the aarch64 baseline.
pub(crate) unsafe fn f32_blocks(xs: &[f32]) -> ([Simd<f32, 8>; 4], usize) {
    let blocks = xs.len() / 32;
    let p = xs.as_ptr();
    let mut out = [[0.0f32; 8]; 4];
    // SAFETY: NEON is enabled on aarch64. For block `i`, offsets `32i + 4j` with j < 8 read
    // four floats each, and `32i + 32 <= 32 * blocks <= xs.len()`. The stores write into the
    // 8-element `out[k]` arrays, two 4-lane halves each.
    unsafe {
        // acc[j] holds elements 4j..4j+4 of the block (j = 0..8).
        let mut acc = [vdupq_n_f32(0.0); 8];
        for i in 0..blocks {
            for (j, a) in acc.iter_mut().enumerate() {
                *a = vaddq_f32(*a, vld1q_f32(p.add(32 * i + 4 * j)));
            }
        }
        for (k, lanes) in out.iter_mut().enumerate() {
            vst1q_f32(lanes.as_mut_ptr(), acc[2 * k]);
            vst1q_f32(lanes.as_mut_ptr().add(4), acc[2 * k + 1]);
        }
    }
    (out.map(Simd::from_array), blocks * 32)
}

/// f64 counterpart: 16-element blocks, accumulator `k` holds elements `4k..4k+4`, held as two
/// 2-lane registers. See [`f32_blocks`].
///
/// # Safety
/// NEON is part of the aarch64 baseline.
pub(crate) unsafe fn f64_blocks(xs: &[f64]) -> ([Simd<f64, 4>; 4], usize) {
    let blocks = xs.len() / 16;
    let p = xs.as_ptr();
    let mut out = [[0.0f64; 4]; 4];
    // SAFETY: NEON is enabled on aarch64. For block `i`, offsets `16i + 2j` with j < 8 read two
    // doubles each, and `16i + 16 <= 16 * blocks <= xs.len()`. The stores write into the
    // 4-element `out[k]` arrays, two 2-lane halves each.
    unsafe {
        // acc[j] holds elements 2j..2j+2 of the block (j = 0..8).
        let mut acc = [vdupq_n_f64(0.0); 8];
        for i in 0..blocks {
            for (j, a) in acc.iter_mut().enumerate() {
                *a = vaddq_f64(*a, vld1q_f64(p.add(16 * i + 2 * j)));
            }
        }
        for (k, lanes) in out.iter_mut().enumerate() {
            vst1q_f64(lanes.as_mut_ptr(), acc[2 * k]);
            vst1q_f64(lanes.as_mut_ptr().add(2), acc[2 * k + 1]);
        }
    }
    (out.map(Simd::from_array), blocks * 16)
}
