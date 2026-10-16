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

/// NEON level-1 block kernel for `f32`, mirroring `x86::l1_f32`. `OP` selects the per-element
/// product: 0 = `x*y`, 1 = `x*x` (nrm2), 2 = `|x|` (asum). Returns the 32 lane accumulators (lane
/// `j` sums element `32*i + j` over the blocks `i`) and the number of elements consumed (a
/// multiple of 32). Multiply and add are separate roundings, as in the portable loop, so the
/// result is bit-identical to it; `vfmaq` is not used.
///
/// # Safety
/// `y.len() == x.len()` (for `OP != 0`, `y` is not read and may be `x`).
pub(crate) unsafe fn l1_f32<const OP: u8>(x: &[f32], y: &[f32]) -> ([f32; 32], usize) {
    let blocks = x.len() / 32;
    let (px, py) = (x.as_ptr(), y.as_ptr());
    // SAFETY: NEON is part of the aarch64 baseline.
    let mut acc = [unsafe { vdupq_n_f32(0.0) }; 8];
    for i in 0..blocks {
        for (k, a) in acc.iter_mut().enumerate() {
            // SAFETY: 32*i + 4*k + 4 <= 32*blocks <= len for both slices.
            unsafe {
                let xv = vld1q_f32(px.add(32 * i + 4 * k));
                let p = match OP {
                    0 => vmulq_f32(xv, vld1q_f32(py.add(32 * i + 4 * k))),
                    1 => vmulq_f32(xv, xv),
                    _ => vabsq_f32(xv),
                };
                *a = vaddq_f32(*a, p);
            }
        }
    }
    let mut out = [0.0f32; 32];
    for (k, a) in acc.iter().enumerate() {
        // SAFETY: `out` has 32 elements and each store writes 4 at `4*k`, k < 8.
        unsafe { vst1q_f32(out.as_mut_ptr().add(4 * k), *a) };
    }
    (out, blocks * 32)
}

/// NEON level-1 block kernel for `f64`; see [`l1_f32`]. 16 lane accumulators in eight 2-lane
/// registers.
///
/// # Safety
/// As for [`l1_f32`].
pub(crate) unsafe fn l1_f64<const OP: u8>(x: &[f64], y: &[f64]) -> ([f64; 16], usize) {
    let blocks = x.len() / 16;
    let (px, py) = (x.as_ptr(), y.as_ptr());
    // SAFETY: NEON is part of the aarch64 baseline.
    let mut acc = [unsafe { vdupq_n_f64(0.0) }; 8];
    for i in 0..blocks {
        for (k, a) in acc.iter_mut().enumerate() {
            // SAFETY: 16*i + 2*k + 2 <= 16*blocks <= len for both slices.
            unsafe {
                let xv = vld1q_f64(px.add(16 * i + 2 * k));
                let p = match OP {
                    0 => vmulq_f64(xv, vld1q_f64(py.add(16 * i + 2 * k))),
                    1 => vmulq_f64(xv, xv),
                    _ => vabsq_f64(xv),
                };
                *a = vaddq_f64(*a, p);
            }
        }
    }
    let mut out = [0.0f64; 16];
    for (k, a) in acc.iter().enumerate() {
        // SAFETY: `out` has 16 elements and each store writes 2 at `2*k`, k < 8.
        unsafe { vst1q_f64(out.as_mut_ptr().add(2 * k), *a) };
    }
    (out, blocks * 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar model of the block kernel: lane `j` sums `op(x[32*i + j])` over the blocks, with
    /// separate multiply and add roundings. Used to check the NEON lane mapping bit for bit.
    fn model_f32(op: u8, x: &[f32], y: &[f32]) -> ([f32; 32], usize) {
        let blocks = x.len() / 32;
        let mut acc = [0.0f32; 32];
        for i in 0..blocks {
            for j in 0..32 {
                let e = 32 * i + j;
                let p = match op {
                    0 => x[e] * y[e],
                    1 => x[e] * x[e],
                    _ => x[e].abs(),
                };
                acc[j] += p;
            }
        }
        (acc, blocks * 32)
    }

    #[test]
    fn l1_f32_matches_lane_model_bitwise() {
        for len in [0usize, 1, 31, 32, 33, 64, 100, 257] {
            let x: std::vec::Vec<f32> = (0..len)
                .map(|i| (i as f32 * 0.37 - 11.0).sin() * 1e3)
                .collect();
            let y: std::vec::Vec<f32> = (0..len)
                .map(|i| (i as f32 * 0.91 + 2.0).cos() * 7.5)
                .collect();
            // SAFETY: equal lengths; NEON is the aarch64 baseline.
            let got = [
                unsafe { l1_f32::<0>(&x, &y) },
                unsafe { l1_f32::<1>(&x, &x) },
                unsafe { l1_f32::<2>(&x, &x) },
            ];
            let want = [
                model_f32(0, &x, &y),
                model_f32(1, &x, &x),
                model_f32(2, &x, &x),
            ];
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.1, w.1);
                for (a, b) in g.0.iter().zip(&w.0) {
                    assert_eq!(a.to_bits(), b.to_bits(), "len {len}");
                }
            }
        }
    }
}
