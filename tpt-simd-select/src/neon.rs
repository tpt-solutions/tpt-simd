//! aarch64 NEON lane permutation (`out[i] = v[idx[i] & 7]`), for `i32` and `f32` lanes.
//!
//! The 8-lane vector is a 32-byte table, and each output lane's four bytes are looked up by
//! `vqtbl2q_u8`. Byte indices are `4 * lane + k`, built from the lane indices with integer
//! operations that never carry between bytes, because `4 * lane <= 28`.

use core::arch::aarch64::*;

/// `out[i] = v[idx[i] & 7]` on the raw 32-bit lanes.
pub(crate) fn permute8(v: &[u32; 8], idx: &[i32; 8]) -> [u32; 8] {
    let mut out = [0u32; 8];
    // SAFETY: NEON is part of the aarch64 baseline. The loads read 4 lanes at offsets 0, 4, and
    // the stores write 4 lanes at offsets 0 and 4 of 8-element arrays.
    unsafe {
        let table = uint8x16x2_t(
            vreinterpretq_u8_u32(vld1q_u32(v.as_ptr())),
            vreinterpretq_u8_u32(vld1q_u32(v.as_ptr().add(4))),
        );
        for h in [0usize, 4] {
            let lanes = vandq_s32(vld1q_s32(idx.as_ptr().add(h)), vdupq_n_s32(7));
            // 4 * lane in every byte of each 32-bit lane, then bytes 4*lane + (0, 1, 2, 3).
            let base = vshlq_n_u32(vreinterpretq_u32_s32(lanes), 2);
            let replicated = vmulq_u32(base, vdupq_n_u32(0x0101_0101));
            let bytes = vaddq_u32(replicated, vdupq_n_u32(0x0302_0100));
            let picked = vqtbl2q_u8(table, vreinterpretq_u8_u32(bytes));
            vst1q_u32(out.as_mut_ptr().add(h), vreinterpretq_u32_u8(picked));
        }
    }
    out
}
