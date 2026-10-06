//! SIMD branch-free conditional selection.
//!
//! This is the **home crate of [`select_f32`]** (re-exported by
//! `tpt-simd-select`).
//!
//! * [`blend_i32`], [`blend_f32`], [`blend_i8`]: per lane, `mask ? a : b`
//!   with a [`SimdMask`] (as produced by `tpt-simd-compare`).
//! * [`select_f32`]: like `_mm256_blendv_ps`, the condition is a float
//!   vector and a lane takes `a` when the condition lane's **sign bit** is
//!   set, else `b`. See its docs for NaN and zero behaviour.
//! * [`blend_imm_i32`], [`blend_imm_f32`], [`blend_imm_i8`]: compile-time
//!   constant masks (`_mm256_blend_epi32` style); a set bit `i` selects lane
//!   `i` of `a` (note: *opposite* operand order to the Intel immediate
//!   intrinsics, matching `blend_*`/`SimdMask::select` where true means `a`).
//!
//! All functions are branch-free; LLVM lowers them to `vblendv*` /
//! `vpblendd` on AVX2.
//!
//! ```
//! use tpt_simd_blend::blend_i32;
//! use tpt_simd_core::{I32x8, SimdMask};
//! let m = SimdMask::<i32, 8>::from_bitmask(0b0000_1111);
//! let r = blend_i32(m, I32x8::splat(1), I32x8::splat(2));
//! assert_eq!(r.to_array(), [1, 1, 1, 1, 2, 2, 2, 2]);
//! ```
#![no_std]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

use tpt_simd_core::Simd;
pub use tpt_simd_core::SimdMask;

/// Per lane: `if mask { a } else { b }` for `i32x8`.
///
/// Performance: one `vblendvps`/`vpblendvb` on AVX2.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_blend::blend_i32;
/// use tpt_simd_core::{I32x8, SimdMask};
/// let r = blend_i32(SimdMask::splat(true), I32x8::splat(7), I32x8::splat(9));
/// assert_eq!(r, I32x8::splat(7));
/// ```
#[inline]
pub fn blend_i32(mask: SimdMask<i32, 8>, a: Simd<i32, 8>, b: Simd<i32, 8>) -> Simd<i32, 8> {
    mask.select(a, b)
}

/// Per lane: `if mask { a } else { b }` for `f32x8`. Lane values (including
/// NaN payloads and signed zeros) are copied bit-exactly.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_blend::blend_f32;
/// use tpt_simd_core::{F32x8, SimdMask};
/// let m = SimdMask::<f32, 8>::from_bitmask(0b01);
/// let r = blend_f32(m, F32x8::splat(1.0), F32x8::splat(-1.0));
/// assert_eq!(r.to_array()[..2], [1.0, -1.0]);
/// ```
#[inline]
pub fn blend_f32(mask: SimdMask<f32, 8>, a: Simd<f32, 8>, b: Simd<f32, 8>) -> Simd<f32, 8> {
    mask.select(a, b)
}

/// Per lane: `if mask { a } else { b }` for `i8x32`.
///
/// Performance: one `vpblendvb` on AVX2.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_blend::blend_i8;
/// use tpt_simd_core::{I8x32, SimdMask};
/// let m = SimdMask::<i8, 32>::from_bitmask(0xFFFF_0000);
/// let r = blend_i8(m, I8x32::splat(1), I8x32::splat(0));
/// assert_eq!(r.to_array()[15], 0);
/// assert_eq!(r.to_array()[16], 1);
/// ```
#[inline]
pub fn blend_i8(mask: SimdMask<i8, 32>, a: Simd<i8, 32>, b: Simd<i8, 32>) -> Simd<i8, 32> {
    mask.select(a, b)
}

/// Per lane: `a` if the **sign bit** of `condition` is set, else `b`
/// (`_mm256_blendv_ps` semantics).
///
/// Only bit 31 of each condition lane matters: `-0.0`, negative values and
/// NaNs with the sign bit set select `a`; `+0.0`, positive values and
/// positive NaNs select `b`. Comparison masks from other crates can be
/// used via [`blend_f32`] instead; to build a condition from a boolean,
/// use `-1.0`/`+1.0` (or any negative/positive value).
///
/// Performance: one `vblendvps` on AVX2.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_blend::select_f32;
/// use tpt_simd_core::F32x8;
/// let c = F32x8::from_array([-1.0, 1.0, -0.0, 0.0, -5.0, 5.0, f32::NEG_INFINITY, f32::INFINITY]);
/// let r = select_f32(c, F32x8::splat(10.0), F32x8::splat(20.0));
/// assert_eq!(r.to_array(), [10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0]);
/// ```
#[inline]
pub fn select_f32(condition: Simd<f32, 8>, a: Simd<f32, 8>, b: Simd<f32, 8>) -> Simd<f32, 8> {
    let c = condition.to_array();
    let (x, y) = (a.to_array(), b.to_array());
    Simd::from_array(core::array::from_fn(|i| {
        if c[i].is_sign_negative() { x[i] } else { y[i] }
    }))
}

/// Compile-time-mask blend for `i32x8`: lane `i` is `a[i]` if bit `i` of
/// `MASK` is set, else `b[i]`.
///
/// Performance: a single `vpblendd` with an immediate on AVX2.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_blend::blend_imm_i32;
/// use tpt_simd_core::I32x8;
/// let r = blend_imm_i32::<0b1010_1010>(I32x8::splat(1), I32x8::splat(0));
/// assert_eq!(r.to_array(), [0, 1, 0, 1, 0, 1, 0, 1]);
/// ```
#[inline]
pub fn blend_imm_i32<const MASK: u8>(a: Simd<i32, 8>, b: Simd<i32, 8>) -> Simd<i32, 8> {
    Simd::from_fn(|i| if (MASK >> i) & 1 != 0 { a[i] } else { b[i] })
}

/// Compile-time-mask blend for `f32x8`: lane `i` is `a[i]` if bit `i` of
/// `MASK` is set, else `b[i]`.
///
/// Performance: a single `vblendps` with an immediate on AVX2.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_blend::blend_imm_f32;
/// use tpt_simd_core::F32x8;
/// let r = blend_imm_f32::<0x0F>(F32x8::splat(1.0), F32x8::splat(2.0));
/// assert_eq!(r.to_array(), [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
/// ```
#[inline]
pub fn blend_imm_f32<const MASK: u8>(a: Simd<f32, 8>, b: Simd<f32, 8>) -> Simd<f32, 8> {
    Simd::from_fn(|i| if (MASK >> i) & 1 != 0 { a[i] } else { b[i] })
}

/// Compile-time-mask blend for `i8x32`: lane `i` is `a[i]` if bit `i` of
/// `MASK` is set, else `b[i]`.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_blend::blend_imm_i8;
/// use tpt_simd_core::I8x32;
/// let r = blend_imm_i8::<0x0000_0001>(I8x32::splat(1), I8x32::splat(0));
/// assert_eq!(r.to_array()[0], 1);
/// assert_eq!(r.to_array()[1], 0);
/// ```
#[inline]
pub fn blend_imm_i8<const MASK: u32>(a: Simd<i8, 32>, b: Simd<i8, 32>) -> Simd<i8, 32> {
    Simd::from_fn(|i| if (MASK >> i) & 1 != 0 { a[i] } else { b[i] })
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use tpt_simd_core::SimdElement;

    fn lanes<T: SimdElement + core::fmt::Debug, S: Strategy<Value = T>, const N: usize>(
        s: S,
    ) -> impl Strategy<Value = Simd<T, N>> {
        proptest::collection::vec(s, N).prop_map(|v| Simd::from_slice(&v))
    }
    use tpt_simd_core::{F32x8, I8x32, I32x8};
    use tpt_simd_testutil::{f32_with_specials, i32_edgy};

    fn ref_blend<T: Copy, const N: usize>(bits: u64, a: [T; N], b: [T; N]) -> [T; N] {
        core::array::from_fn(|i| if (bits >> i) & 1 != 0 { a[i] } else { b[i] })
    }

    #[test]
    fn select_sign_bit_nan() {
        let neg_nan = f32::from_bits(0xFFC0_0000);
        let c = F32x8::from_array([neg_nan, f32::NAN, 1.0, -1.0, 0.0, -0.0, 3.0, -3.0]);
        let r = select_f32(c, F32x8::splat(1.0), F32x8::splat(2.0));
        assert_eq!(r.to_array(), [1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0]);
    }

    #[test]
    fn blend_preserves_nan_bits() {
        let n = f32::from_bits(0x7FC0_1234);
        let r = blend_f32(SimdMask::splat(true), F32x8::splat(n), F32x8::splat(0.0));
        assert_eq!(r.to_array()[3].to_bits(), 0x7FC0_1234);
    }

    #[test]
    fn imm_extremes() {
        let (a, b) = (I32x8::splat(1), I32x8::splat(2));
        assert_eq!(blend_imm_i32::<0>(a, b), b);
        assert_eq!(blend_imm_i32::<0xFF>(a, b), a);
        let (a, b) = (I8x32::splat(1), I8x32::splat(2));
        assert_eq!(blend_imm_i8::<0>(a, b), b);
        assert_eq!(blend_imm_i8::<{ u32::MAX }>(a, b), a);
    }

    proptest! {
        #[test]
        fn i32_vs_scalar(bits in any::<u8>(), a in lanes::<i32, _, 8>(i32_edgy()), b in lanes::<i32, _, 8>(i32_edgy())) {
            let m = SimdMask::<i32, 8>::from_bitmask(u64::from(bits));
            let want = ref_blend(u64::from(bits), a.to_array(), b.to_array());
            prop_assert_eq!(blend_i32(m, a, b).to_array(), want);
            prop_assert_eq!(blend_imm_i32::<0b1100_0101>(a, b).to_array(),
                ref_blend(0b1100_0101, a.to_array(), b.to_array()));
        }

        #[test]
        fn f32_vs_scalar(bits in any::<u8>(), a in lanes::<f32, _, 8>(f32_with_specials()), b in lanes::<f32, _, 8>(f32_with_specials())) {
            let m = SimdMask::<f32, 8>::from_bitmask(u64::from(bits));
            let want = ref_blend(u64::from(bits), a.to_array().map(f32::to_bits), b.to_array().map(f32::to_bits));
            prop_assert_eq!(blend_f32(m, a, b).to_array().map(f32::to_bits), want);
            prop_assert_eq!(
                blend_imm_f32::<0x3C>(a, b).to_array().map(f32::to_bits),
                ref_blend(0x3C, a.to_array().map(f32::to_bits), b.to_array().map(f32::to_bits))
            );
        }

        #[test]
        fn select_vs_scalar(c in lanes::<f32, _, 8>(f32_with_specials()), a in lanes::<f32, _, 8>(f32_with_specials()), b in lanes::<f32, _, 8>(f32_with_specials())) {
            let bits = c.to_array().iter().enumerate()
                .fold(0u64, |m, (i, v)| m | (u64::from(v.to_bits() >> 31) << i));
            let want = ref_blend(bits, a.to_array().map(f32::to_bits), b.to_array().map(f32::to_bits));
            prop_assert_eq!(select_f32(c, a, b).to_array().map(f32::to_bits), want);
        }

        #[test]
        fn i8_vs_scalar(bits in any::<u32>(), a in lanes::<i8, _, 32>(any::<i8>()), b in lanes::<i8, _, 32>(any::<i8>())) {
            let m = SimdMask::<i8, 32>::from_bitmask(u64::from(bits));
            let want = ref_blend(u64::from(bits), a.to_array(), b.to_array());
            prop_assert_eq!(blend_i8(m, a, b).to_array(), want);
            prop_assert_eq!(blend_imm_i8::<0xF00F_00FF>(a, b).to_array(),
                ref_blend(0xF00F_00FF, a.to_array(), b.to_array()));
        }
    }
}
