//! aarch64 NEON gemm microkernels, the same tiles and per-lane arithmetic as [`crate::x86`].
//!
//! The f32 tile is 16 rows by 6 columns and the f64 tile is 8 by 6, as on AVX2. Each tile is
//! computed in two halves of 8 (f32) or 4 (f64) rows, so every half uses 12 accumulators.
//! Every output element receives the same fused multiply-adds, in the same order over `p`, as
//! the AVX2 kernel, so the results are bit-identical to it.

use core::arch::aarch64::*;

/// `out[j*16 + i] = sum_p a[p*16 + i] * b[p*6 + j]` for a 16x6 f32 tile.
///
/// # Safety
/// `a` is valid for `16*kc` reads, `b` for `6*kc` reads and `out` for 96 writes.
pub(crate) unsafe fn kernel_f32(kc: usize, a: *const f32, b: *const f32, out: *mut f32) {
    for half in 0..2 {
        // SAFETY: NEON is part of the aarch64 baseline.
        let mut acc = unsafe { [[vdupq_n_f32(0.0); 2]; 6] };
        for p in 0..kc {
            // SAFETY: p < kc and half * 8 + 8 <= 16, so the reads stay within `16*kc` and `6*kc`.
            unsafe {
                let ap = a.add(p * 16 + half * 8);
                let bp = b.add(p * 6);
                let a0 = vld1q_f32(ap);
                let a1 = vld1q_f32(ap.add(4));
                for (j, cj) in acc.iter_mut().enumerate() {
                    let bj = vdupq_n_f32(*bp.add(j));
                    cj[0] = vfmaq_f32(cj[0], a0, bj);
                    cj[1] = vfmaq_f32(cj[1], a1, bj);
                }
            }
        }
        // SAFETY: the 96 writes are `j*16 + half*8 + 0..8` for j < 6, within `out`.
        unsafe {
            for (j, cj) in acc.iter().enumerate() {
                vst1q_f32(out.add(j * 16 + half * 8), cj[0]);
                vst1q_f32(out.add(j * 16 + half * 8 + 4), cj[1]);
            }
        }
    }
}

/// Full 16x6 f32 tile that accumulates into C: `c[j*ldc + i] += alpha * sum_p ...`, one fused
/// multiply-add per vector, as in `x86::kernel_f32_store`.
///
/// # Safety
/// `a` is valid for `16*kc` reads, `b` for `6*kc` reads, and `c` for writes to a 16x6 tile with
/// column stride `ldc`.
pub(crate) unsafe fn kernel_f32_store(
    kc: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
    alpha: f32,
) {
    // SAFETY: NEON is part of the aarch64 baseline; the tile reads and writes are in bounds by
    // the safety contract.
    unsafe {
        let av = vdupq_n_f32(alpha);
        for half in 0..2 {
            let mut acc = [[vdupq_n_f32(0.0); 2]; 6];
            for p in 0..kc {
                let ap = a.add(p * 16 + half * 8);
                let bp = b.add(p * 6);
                let a0 = vld1q_f32(ap);
                let a1 = vld1q_f32(ap.add(4));
                for (j, cj) in acc.iter_mut().enumerate() {
                    let bj = vdupq_n_f32(*bp.add(j));
                    cj[0] = vfmaq_f32(cj[0], a0, bj);
                    cj[1] = vfmaq_f32(cj[1], a1, bj);
                }
            }
            for (j, cj) in acc.iter().enumerate() {
                let col = c.add(j * ldc + half * 8);
                let c0 = vld1q_f32(col);
                let c1 = vld1q_f32(col.add(4));
                vst1q_f32(col, vfmaq_f32(c0, av, cj[0]));
                vst1q_f32(col.add(4), vfmaq_f32(c1, av, cj[1]));
            }
        }
    }
}

/// `out[j*8 + i] = sum_p a[p*8 + i] * b[p*6 + j]` for an 8x6 f64 tile.
///
/// # Safety
/// `a` is valid for `8*kc` reads, `b` for `6*kc` reads and `out` for 48 writes.
pub(crate) unsafe fn kernel_f64(kc: usize, a: *const f64, b: *const f64, out: *mut f64) {
    for half in 0..2 {
        // SAFETY: NEON is part of the aarch64 baseline.
        let mut acc = unsafe { [[vdupq_n_f64(0.0); 2]; 6] };
        for p in 0..kc {
            // SAFETY: p < kc and half * 4 + 4 <= 8, so the reads stay within `8*kc` and `6*kc`.
            unsafe {
                let ap = a.add(p * 8 + half * 4);
                let bp = b.add(p * 6);
                let a0 = vld1q_f64(ap);
                let a1 = vld1q_f64(ap.add(2));
                for (j, cj) in acc.iter_mut().enumerate() {
                    let bj = vdupq_n_f64(*bp.add(j));
                    cj[0] = vfmaq_f64(cj[0], a0, bj);
                    cj[1] = vfmaq_f64(cj[1], a1, bj);
                }
            }
        }
        // SAFETY: the 48 writes are `j*8 + half*4 + 0..4` for j < 6, within `out`.
        unsafe {
            for (j, cj) in acc.iter().enumerate() {
                vst1q_f64(out.add(j * 8 + half * 4), cj[0]);
                vst1q_f64(out.add(j * 8 + half * 4 + 2), cj[1]);
            }
        }
    }
}

/// Full 8x6 f64 tile that accumulates into C; see [`kernel_f32_store`].
///
/// # Safety
/// `a` is valid for `8*kc` reads, `b` for `6*kc` reads, and `c` for writes to an 8x6 tile with
/// column stride `ldc`.
pub(crate) unsafe fn kernel_f64_store(
    kc: usize,
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    ldc: usize,
    alpha: f64,
) {
    // SAFETY: NEON is part of the aarch64 baseline; the tile accesses are in bounds by the
    // safety contract.
    unsafe {
        let av = vdupq_n_f64(alpha);
        for half in 0..2 {
            let mut acc = [[vdupq_n_f64(0.0); 2]; 6];
            for p in 0..kc {
                let ap = a.add(p * 8 + half * 4);
                let bp = b.add(p * 6);
                let a0 = vld1q_f64(ap);
                let a1 = vld1q_f64(ap.add(2));
                for (j, cj) in acc.iter_mut().enumerate() {
                    let bj = vdupq_n_f64(*bp.add(j));
                    cj[0] = vfmaq_f64(cj[0], a0, bj);
                    cj[1] = vfmaq_f64(cj[1], a1, bj);
                }
            }
            for (j, cj) in acc.iter().enumerate() {
                let col = c.add(j * ldc + half * 4);
                let c0 = vld1q_f64(col);
                let c1 = vld1q_f64(col.add(2));
                vst1q_f64(col, vfmaq_f64(c0, av, cj[0]));
                vst1q_f64(col.add(2), vfmaq_f64(c1, av, cj[1]));
            }
        }
    }
}
