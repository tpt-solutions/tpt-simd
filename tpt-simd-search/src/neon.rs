//! aarch64 NEON byte search, mirroring the SSE2 kernels: 16-byte compares, a movemask built from
//! weighted lanes, and counts flushed before the byte lanes can overflow.

use core::arch::aarch64::*;

/// Bit `i` of the result is the top bit of byte `i` of `v` (the bytes are 0 or 0xFF).
#[inline(always)]
fn movemask16(v: uint8x16_t) -> u32 {
    const WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    // SAFETY: NEON is part of the aarch64 baseline; register-only operations.
    unsafe {
        let picked = vandq_u8(v, vld1q_u8(WEIGHTS.as_ptr()));
        let lo = vaddv_u8(vget_low_u8(picked)) as u32;
        let hi = vaddv_u8(vget_high_u8(picked)) as u32;
        lo | (hi << 8)
    }
}

/// Index of the first `needle` byte, scanning 64 bytes per iteration.
pub(crate) fn find_byte(haystack: &[u8], needle: u8) -> Option<usize> {
    let mut i = 0;
    // SAFETY: NEON is part of the aarch64 baseline. Every load reads 16 bytes at `i + 16k` with
    // `i + 64 <= len`, so all reads are inside `haystack`.
    unsafe {
        let v = vdupq_n_u8(needle);
        let p = haystack.as_ptr();
        while i + 64 <= haystack.len() {
            let c0 = vceqq_u8(vld1q_u8(p.add(i)), v);
            let c1 = vceqq_u8(vld1q_u8(p.add(i + 16)), v);
            let c2 = vceqq_u8(vld1q_u8(p.add(i + 32)), v);
            let c3 = vceqq_u8(vld1q_u8(p.add(i + 48)), v);
            let any = vorrq_u8(vorrq_u8(c0, c1), vorrq_u8(c2, c3));
            if vmaxvq_u8(any) != 0 {
                for (k, c) in [c0, c1, c2, c3].into_iter().enumerate() {
                    let m = movemask16(c);
                    if m != 0 {
                        return Some(i + 16 * k + m.trailing_zeros() as usize);
                    }
                }
            }
            i += 64;
        }
        while i + 16 <= haystack.len() {
            let m = movemask16(vceqq_u8(vld1q_u8(p.add(i)), v));
            if m != 0 {
                return Some(i + m.trailing_zeros() as usize);
            }
            i += 16;
        }
    }
    haystack[i..]
        .iter()
        .position(|&b| b == needle)
        .map(|p| i + p)
}

/// Number of `needle` bytes in `haystack`. Each match is a lane of 0xFF, so subtracting the
/// compare masks counts matches per byte lane. A lane holds at most 255 counts, so the lanes are
/// summed every 255 blocks (4080 bytes).
pub(crate) fn count_byte(haystack: &[u8], needle: u8) -> usize {
    const BLOCK: usize = 16;
    const MAX_BLOCKS: usize = 255;
    let mut i = 0;
    let mut total = 0usize;
    // SAFETY: NEON is part of the aarch64 baseline. Every load is 16 bytes at `i + 16k`, inside
    // `haystack` by the loop bounds.
    unsafe {
        let v = vdupq_n_u8(needle);
        let p = haystack.as_ptr();
        while i + BLOCK * MAX_BLOCKS <= haystack.len() {
            let mut acc = vdupq_n_u8(0);
            for k in 0..MAX_BLOCKS {
                acc = vsubq_u8(acc, vceqq_u8(vld1q_u8(p.add(i + BLOCK * k)), v));
            }
            total += vaddlvq_u8(acc) as usize;
            i += BLOCK * MAX_BLOCKS;
        }
        let mut acc = vdupq_n_u8(0);
        while i + BLOCK <= haystack.len() {
            acc = vsubq_u8(acc, vceqq_u8(vld1q_u8(p.add(i)), v));
            i += BLOCK;
        }
        total += vaddlvq_u8(acc) as usize;
    }
    total + haystack[i..].iter().filter(|&&b| b == needle).count()
}
