//! aarch64 NEON per-lane variable shifts.
//!
//! `vshlq` takes a signed count per lane and uses only its low byte. Amounts are clamped to 64
//! first, so an unsigned amount of 32 or more keeps its meaning after the cast to a signed count.
//! A left shift by 32 or more gives 0, a logical right shift gives 0, and an arithmetic right
//! shift gives the sign fill, as the portable definitions require.

use core::arch::aarch64::*;

/// Clamp each amount to 64 and read it as a signed shift count.
#[inline(always)]
fn count(n: uint32x4_t) -> int32x4_t {
    // SAFETY: NEON is part of the aarch64 baseline; register-only operations.
    unsafe { vreinterpretq_s32_u32(vminq_u32(n, vdupq_n_u32(64))) }
}

/// Per-lane `a << n`, with `n >= 32` giving 0.
pub(crate) fn shift_left(a: [i32; 8], n: [u32; 8]) -> [i32; 8] {
    let mut out = [0i32; 8];
    // SAFETY: NEON is enabled on aarch64. Each access covers 4 lanes at offsets 0 and 4 of
    // 8-element arrays.
    unsafe {
        for h in [0usize, 4] {
            let va = vld1q_s32(a.as_ptr().add(h));
            let vn = vld1q_u32(n.as_ptr().add(h));
            vst1q_s32(out.as_mut_ptr().add(h), vshlq_s32(va, count(vn)));
        }
    }
    out
}

/// Per-lane logical `a >> n` (as unsigned), with `n >= 32` giving 0.
pub(crate) fn shift_right_logical(a: [i32; 8], n: [u32; 8]) -> [i32; 8] {
    let mut out = [0i32; 8];
    // SAFETY: as for `shift_left`.
    unsafe {
        for h in [0usize, 4] {
            let va = vreinterpretq_u32_s32(vld1q_s32(a.as_ptr().add(h)));
            let neg = vnegq_s32(count(vld1q_u32(n.as_ptr().add(h))));
            let r = vshlq_u32(va, neg);
            vst1q_s32(out.as_mut_ptr().add(h), vreinterpretq_s32_u32(r));
        }
    }
    out
}

/// Per-lane arithmetic `a >> n`, with `n >= 32` giving the sign fill.
pub(crate) fn shift_right_arithmetic(a: [i32; 8], n: [u32; 8]) -> [i32; 8] {
    let mut out = [0i32; 8];
    // SAFETY: as for `shift_left`.
    unsafe {
        for h in [0usize, 4] {
            let va = vld1q_s32(a.as_ptr().add(h));
            let neg = vnegq_s32(count(vld1q_u32(n.as_ptr().add(h))));
            vst1q_s32(out.as_mut_ptr().add(h), vshlq_s32(va, neg));
        }
    }
    out
}
