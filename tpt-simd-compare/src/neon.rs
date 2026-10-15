//! aarch64 NEON comparisons, masks, bitmasks and counts.
//!
//! Same names and signatures as the AVX2 `x86` module, which `lib.rs` aliases to this module
//! on aarch64. Comparisons use the NEON ordered forms, so they match Rust's scalar operators:
//! `ne` is the negation of an ordered `eq`, which makes NaN compare unequal, as `!=` does.

use core::arch::aarch64::*;
use tpt_simd_core::{MaskLane, Simd, SimdElement, SimdMask};

/// The byte view of an `i8` comparison result, which is already `uint8x16_t`.
#[inline(always)]
fn u8_identity(v: uint8x16_t) -> uint8x16_t {
    v
}

/// Stores two 16-byte halves of raw mask lanes into a 32-byte mask array.
#[inline(always)]
fn wrap<T: SimdElement, const N: usize>(lo: uint8x16_t, hi: uint8x16_t) -> SimdMask<T, N> {
    assert_eq!(core::mem::size_of::<[T::MaskLane; N]>(), 32);
    let mut out = [T::MaskLane::CLEAR; N];
    // SAFETY: `out` is exactly 32 writable bytes (asserted above); the stores are unaligned.
    unsafe {
        vst1q_u8(out.as_mut_ptr().cast(), lo);
        vst1q_u8(out.as_mut_ptr().cast::<u8>().add(16), hi);
    }
    SimdMask::from_raw(out)
}

macro_rules! int_family {
    (
        $t:ty, $n:literal, $l:literal, $ld:ident, $cvt:ident,
        $gt:ident, $eq:ident, $fgt:ident, $flt:ident, $feq:ident, $fne:ident, $fge:ident, $fle:ident
    ) => {
        #[inline(always)]
        pub fn $fgt(a: Simd<$t, $n>, b: Simd<$t, $n>) -> SimdMask<$t, $n> {
            let (pa, pb) = (a.0.as_ptr(), b.0.as_ptr());
            // SAFETY: NEON is part of the aarch64 baseline; the loads stay inside the arrays.
            let lo = unsafe { $cvt($gt($ld(pa), $ld(pb))) };
            // SAFETY: as above, for the second half of the lanes.
            let hi = unsafe { $cvt($gt($ld(pa.add($l)), $ld(pb.add($l)))) };
            wrap::<$t, $n>(lo, hi)
        }
        #[inline(always)]
        pub fn $flt(a: Simd<$t, $n>, b: Simd<$t, $n>) -> SimdMask<$t, $n> {
            let (pa, pb) = (a.0.as_ptr(), b.0.as_ptr());
            // SAFETY: as for the greater-than case.
            let lo = unsafe { $cvt($gt($ld(pb), $ld(pa))) };
            // SAFETY: as above, for the second half.
            let hi = unsafe { $cvt($gt($ld(pb.add($l)), $ld(pa.add($l)))) };
            wrap::<$t, $n>(lo, hi)
        }
        #[inline(always)]
        pub fn $feq(a: Simd<$t, $n>, b: Simd<$t, $n>) -> SimdMask<$t, $n> {
            let (pa, pb) = (a.0.as_ptr(), b.0.as_ptr());
            // SAFETY: as for the greater-than case.
            let lo = unsafe { $cvt($eq($ld(pa), $ld(pb))) };
            // SAFETY: as above, for the second half.
            let hi = unsafe { $cvt($eq($ld(pa.add($l)), $ld(pb.add($l)))) };
            wrap::<$t, $n>(lo, hi)
        }
        #[inline(always)]
        pub fn $fne(a: Simd<$t, $n>, b: Simd<$t, $n>) -> SimdMask<$t, $n> {
            let (pa, pb) = (a.0.as_ptr(), b.0.as_ptr());
            // SAFETY: as for the greater-than case.
            let lo = unsafe { vmvnq_u8($cvt($eq($ld(pa), $ld(pb)))) };
            // SAFETY: as above, for the second half.
            let hi = unsafe { vmvnq_u8($cvt($eq($ld(pa.add($l)), $ld(pb.add($l))))) };
            wrap::<$t, $n>(lo, hi)
        }
        #[inline(always)]
        pub fn $fge(a: Simd<$t, $n>, b: Simd<$t, $n>) -> SimdMask<$t, $n> {
            let (pa, pb) = (a.0.as_ptr(), b.0.as_ptr());
            // SAFETY: as for the greater-than case. `a >= b` is `!(b > a)` for integers.
            let lo = unsafe { vmvnq_u8($cvt($gt($ld(pb), $ld(pa)))) };
            // SAFETY: as above, for the second half.
            let hi = unsafe { vmvnq_u8($cvt($gt($ld(pb.add($l)), $ld(pa.add($l))))) };
            wrap::<$t, $n>(lo, hi)
        }
        #[inline(always)]
        pub fn $fle(a: Simd<$t, $n>, b: Simd<$t, $n>) -> SimdMask<$t, $n> {
            let (pa, pb) = (a.0.as_ptr(), b.0.as_ptr());
            // SAFETY: as for the greater-than case.
            let lo = unsafe { vmvnq_u8($cvt($gt($ld(pa), $ld(pb)))) };
            // SAFETY: as above, for the second half.
            let hi = unsafe { vmvnq_u8($cvt($gt($ld(pa.add($l)), $ld(pb.add($l))))) };
            wrap::<$t, $n>(lo, hi)
        }
    };
}

int_family!(
    i32,
    8,
    4,
    vld1q_s32,
    vreinterpretq_u8_u32,
    vcgtq_s32,
    vceqq_s32,
    cmp_gt_i32,
    cmp_lt_i32,
    cmp_eq_i32,
    cmp_ne_i32,
    cmp_ge_i32,
    cmp_le_i32
);
int_family!(
    i16,
    16,
    8,
    vld1q_s16,
    vreinterpretq_u8_u16,
    vcgtq_s16,
    vceqq_s16,
    cmp_gt_i16,
    cmp_lt_i16,
    cmp_eq_i16,
    cmp_ne_i16,
    cmp_ge_i16,
    cmp_le_i16
);
int_family!(
    i8,
    32,
    16,
    vld1q_s8,
    u8_identity,
    vcgtq_s8,
    vceqq_s8,
    cmp_gt_i8,
    cmp_lt_i8,
    cmp_eq_i8,
    cmp_ne_i8,
    cmp_ge_i8,
    cmp_le_i8
);

/// `f32` comparisons, ordered like Rust's operators (`ne` is true for NaN).
macro_rules! f32_cmp {
    ($name:ident, |$x:ident, $y:ident| $body:expr) => {
        #[inline(always)]
        pub fn $name(a: Simd<f32, 8>, b: Simd<f32, 8>) -> SimdMask<f32, 8> {
            let (pa, pb) = (a.0.as_ptr(), b.0.as_ptr());
            let half = |k: usize| {
                // SAFETY: NEON is part of the aarch64 baseline; half `k` reads lanes `4k..4k+4`
                // of the 8-element arrays.
                unsafe {
                    let $x = vld1q_f32(pa.add(4 * k));
                    let $y = vld1q_f32(pb.add(4 * k));
                    vreinterpretq_u8_u32($body)
                }
            };
            wrap::<f32, 8>(half(0), half(1))
        }
    };
}

f32_cmp!(cmp_gt_f32, |x, y| vcgtq_f32(x, y));
f32_cmp!(cmp_lt_f32, |x, y| vcltq_f32(x, y));
f32_cmp!(cmp_eq_f32, |x, y| vceqq_f32(x, y));
f32_cmp!(cmp_ne_f32, |x, y| vmvnq_u32(vceqq_f32(x, y)));
f32_cmp!(cmp_ge_f32, |x, y| vcgeq_f32(x, y));
f32_cmp!(cmp_le_f32, |x, y| vcleq_f32(x, y));
// Unordered: either operand is NaN (a NaN is not equal to itself).
f32_cmp!(cmp_unord_f32, |x, y| vmvnq_u32(vandq_u32(
    vceqq_f32(x, x),
    vceqq_f32(y, y)
)));

/// Packs raw mask lanes into a bitmask, lane 0 in bit 0. One- and four-byte lanes use
/// NEON; other widths and the tail use the scalar `bit()`.
#[inline(always)]
pub fn bitmask<L: MaskLane, const N: usize>(raw: &[L; N]) -> u64 {
    assert!(N <= 64, "to_bitmask supports at most 64 lanes");
    let p = raw.as_ptr().cast::<u8>();
    let w = core::mem::size_of::<L>();
    let mut m = 0u64;
    let mut i = 0;
    match w {
        1 => {
            // SAFETY: NEON is part of the aarch64 baseline; `i + 16 <= N` keeps the 16-byte load
            // inside `raw`, which has `N` bytes of one-byte lanes.
            unsafe {
                let weights = [1u8, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
                let wv = vld1q_u8(weights.as_ptr());
                while i + 16 <= N {
                    let v = vandq_u8(vld1q_u8(p.add(i)), wv);
                    let lo = vaddv_u8(vget_low_u8(v)) as u64;
                    let hi = vaddv_u8(vget_high_u8(v)) as u64;
                    m |= (lo | (hi << 8)) << i;
                    i += 16;
                }
            }
        }
        4 => {
            // SAFETY: as above, with four-byte lanes and `i + 4 <= N`.
            unsafe {
                let weights = [1u32, 2, 4, 8];
                let wv = vld1q_u32(weights.as_ptr());
                while i + 4 <= N {
                    let v = vandq_u32(vld1q_u32(p.cast::<u32>().add(i)), wv);
                    m |= u64::from(vaddvq_u32(v)) << i;
                    i += 4;
                }
            }
        }
        _ => {}
    }
    while i < N {
        m |= raw[i].bit() << i;
        i += 1;
    }
    m
}

/// Counts the elements of `data` that satisfy `cmp` against `rhs` (in 4-lane chunks), summing the
/// per-lane counters in blocks so that they cannot overflow.
#[inline(always)]
fn count4<T: Copy>(
    data: &[T],
    rhs: T,
    cmp: impl Fn(&[T; 4]) -> uint32x4_t,
    scalar: impl Fn(T) -> bool,
) -> usize {
    let mut total = 0usize;
    for block in data.chunks(1 << 28) {
        let (body, tail) = block.as_chunks::<4>();
        // SAFETY: NEON is part of the aarch64 baseline; `vsubq_u32` adds 1 per set (all-ones) lane.
        let acc = unsafe {
            let mut acc = vdupq_n_u32(0);
            for c in body {
                acc = vsubq_u32(acc, cmp(c));
            }
            acc
        };
        let mut lanes = [0u32; 4];
        // SAFETY: `lanes` is 16 writable bytes.
        unsafe { vst1q_u32(lanes.as_mut_ptr(), acc) };
        total += lanes.iter().map(|&x| x as usize).sum::<usize>();
        total += tail.iter().filter(|&&x| scalar(x)).count();
    }
    let _ = rhs;
    total
}

macro_rules! count_i32 {
    ($name:ident, $cmp:ident, |$x:ident, $t:ident| $body:expr, $op:tt) => {
        pub fn $name(data: &[i32], rhs: i32) -> usize {
            // SAFETY: NEON is part of the aarch64 baseline.
            let t = unsafe { vdupq_n_s32(rhs) };
            count4(
                data,
                rhs,
                |c| {
                    // SAFETY: `c` holds four i32 values, 16 readable bytes.
                    unsafe {
                        let $x = vld1q_s32(c.as_ptr());
                        let $t = t;
                        $body
                    }
                },
                |x| x $op rhs,
            )
        }
    };
}

count_i32!(count_gt_i32, vcgtq_s32, |x, t| vcgtq_s32(x, t), >);
count_i32!(count_lt_i32, vcltq_s32, |x, t| vcltq_s32(x, t), <);
count_i32!(count_eq_i32, vceqq_s32, |x, t| vceqq_s32(x, t), ==);

macro_rules! count_f32 {
    ($name:ident, |$x:ident, $t:ident| $body:expr, $op:tt) => {
        pub fn $name(data: &[f32], rhs: f32) -> usize {
            // SAFETY: NEON is part of the aarch64 baseline.
            let t = unsafe { vdupq_n_f32(rhs) };
            count4(
                data,
                rhs,
                |c| {
                    // SAFETY: `c` holds four f32 values, 16 readable bytes.
                    unsafe {
                        let $x = vld1q_f32(c.as_ptr());
                        let $t = t;
                        $body
                    }
                },
                |x| x $op rhs,
            )
        }
    };
}

count_f32!(count_gt_f32, |x, t| vcgtq_f32(x, t), >);
count_f32!(count_lt_f32, |x, t| vcltq_f32(x, t), <);
count_f32!(count_eq_f32, |x, t| vceqq_f32(x, t), ==);
