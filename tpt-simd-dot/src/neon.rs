//! aarch64 NEON fast paths. Bit-identical to [`crate::portable`].
//!
//! * `i16`: exact widening products and wrapping `i32` sums, so any summation order gives the
//!   portable result.
//! * `f32`: the same 32-element blocks and four `F32x8` accumulators as the portable code. Each
//!   accumulator lane is a separate multiply then add (`vmulq` then `vaddq`, never fused), so
//!   every lane sees the same rounding sequence and the blocks match bit for bit.

use core::arch::aarch64::*;
use tpt_simd_core::F32x8;

/// NEON `i16` dot product with wrapping `i32` accumulation.
///
/// # Safety
/// `a.len() == b.len()`. NEON is part of the aarch64 baseline.
pub(crate) unsafe fn dot_i16(a: &[i16], b: &[i16]) -> i32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut i = 0;
    // SAFETY: NEON is enabled on aarch64. Each 8-lane load reads indices `i..i + 8` (or
    // `i + 8..i + 16`) with `i + 16 <= n`, so it is in bounds for both slices.
    let mut sum = unsafe {
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        while i + 16 <= n {
            let x0 = vld1q_s16(pa.add(i));
            let y0 = vld1q_s16(pb.add(i));
            let x1 = vld1q_s16(pa.add(i + 8));
            let y1 = vld1q_s16(pb.add(i + 8));
            acc0 = vaddq_s32(acc0, vmull_s16(vget_low_s16(x0), vget_low_s16(y0)));
            acc0 = vaddq_s32(acc0, vmull_high_s16(x0, y0));
            acc1 = vaddq_s32(acc1, vmull_s16(vget_low_s16(x1), vget_low_s16(y1)));
            acc1 = vaddq_s32(acc1, vmull_high_s16(x1, y1));
            i += 16;
        }
        vaddvq_s32(vaddq_s32(acc0, acc1))
    };
    while i < n {
        sum = sum.wrapping_add(i32::from(a[i]) * i32::from(b[i]));
        i += 1;
    }
    sum
}

/// NEON accumulators for every full 32-element block of `a` and `b`, in the layout the
/// portable code uses: accumulator `k` holds elements `8k..8k+8` of each block.
///
/// Returns the accumulators; the caller adds the trailing elements with
/// `portable::dot_f32_tail`. Only the first `len - len % 32` elements are read.
///
/// # Safety
/// `a.len() == b.len()`. NEON is part of the aarch64 baseline.
pub(crate) unsafe fn dot_f32_blocks(a: &[f32], b: &[f32]) -> [F32x8; 4] {
    let full = a.len() - a.len() % 32;
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut out = [F32x8::from_array([0.0; 8]); 4];
    // SAFETY: NEON is enabled on aarch64. Every load reads `off..off + 4` with
    // `off + 4 <= i + 32 <= full <= len`, so it is in bounds for both slices. The stores write
    // into `lanes`, which has 8 elements.
    unsafe {
        // acc[k][h]: lanes 0..4 (h = 0) and 4..8 (h = 1) of accumulator k.
        let mut acc = [[vdupq_n_f32(0.0); 2]; 4];
        let mut i = 0;
        while i < full {
            for (k, pair) in acc.iter_mut().enumerate() {
                for (h, lane) in pair.iter_mut().enumerate() {
                    let off = i + 8 * k + 4 * h;
                    let x = vld1q_f32(pa.add(off));
                    let y = vld1q_f32(pb.add(off));
                    // Separate multiply and add, as in `acc[k] += xv * yv` on the portable path.
                    *lane = vaddq_f32(*lane, vmulq_f32(x, y));
                }
            }
            i += 32;
        }
        for (k, pair) in acc.iter().enumerate() {
            let mut lanes = [0.0f32; 8];
            vst1q_f32(lanes.as_mut_ptr(), pair[0]);
            vst1q_f32(lanes.as_mut_ptr().add(4), pair[1]);
            out[k] = F32x8::from_array(lanes);
        }
    }
    out
}
