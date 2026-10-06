//! SIMD comparisons producing masks.
//!
//! Every comparison takes two full vectors and returns a [`SimdMask`] with
//! one boolean per lane. Shapes match the 256-bit vectors used across the
//! workspace: `i32x8`, `f32x8`, `i16x16` and `i8x32`.
//!
//! Implemented, for each of `i32`, `f32`, `i16`, `i8`: `cmp_gt_*`,
//! `cmp_lt_*`, `cmp_eq_*`, `cmp_ne_*`, `cmp_ge_*`, `cmp_le_*` (24 functions;
//! integer comparisons are signed), plus [`cmp_unord_f32`].
//!
//! Mask helpers: [`mask_any`], [`mask_all`], [`mask_count`],
//! [`mask_to_bitmask`] and [`mask_from_bitmask`].
//!
//! ## NaN semantics (`f32`)
//!
//! Comparisons follow IEEE 754 *ordered* semantics, identical to Rust's
//! scalar `<`, `<=`, `>`, `>=`, `==`, `!=` operators:
//!
//! * `gt`, `lt`, `ge`, `le`, `eq` are **false** if either lane is NaN.
//! * `ne` is **true** if either lane is NaN (it is exactly `!eq`).
//! * `-0.0 == +0.0` is true.
//!
//! Consequently `cmp_ge(a, b)` is *not* the negation of `cmp_lt(a, b)` when
//! NaNs are present. Use [`cmp_unord_f32`] to find NaN lanes.
//!
//! ```
//! use tpt_simd_compare::*;
//! use tpt_simd_core::F32x8;
//! let a = F32x8::from_array([1.0, f32::NAN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
//! let b = F32x8::splat(3.0);
//! let m = cmp_gt_f32(a, b);
//! assert_eq!(mask_to_bitmask(m), 0b1111_1000);
//! assert!(mask_any(m) && !mask_all(m));
//! assert_eq!(mask_count(cmp_ne_f32(a, a)), 1); // only the NaN lane
//! ```
#![no_std]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

pub use tpt_simd_core::SimdMask;
use tpt_simd_core::{Simd, SimdElement};

macro_rules! cmp_fn {
    ($name:ident, $method:ident, $t:ty, $n:literal, $sym:literal, $alias:literal, $same:literal) => {
        #[doc = concat!("Lane-wise `a ", $sym, " b` for [`", $alias, "`](tpt_simd_core::", $alias, ") vectors.")]
        #[doc = ""]
        #[doc = "Performance: one compare instruction (`vpcmp*`/`vcmpps`) on AVX2 via auto-vectorisation."]
        #[doc = ""]
        #[doc = "# Panics"]
        #[doc = "Never."]
        #[doc = ""]
        #[doc = "```"]
        #[doc = concat!("use tpt_simd_core::", $alias, ";")]
        #[doc = concat!("use tpt_simd_compare::{", stringify!($name), ", mask_all};")]
        #[doc = concat!("let a = ", $alias, "::splat(1 as _);")]
        #[doc = concat!("assert_eq!(mask_all(", stringify!($name), "(a, a)), ", $same, ");")]
        #[doc = "```"]
        #[inline]
        pub fn $name(a: Simd<$t, $n>, b: Simd<$t, $n>) -> SimdMask<$t, $n> {
            a.$method(b)
        }
    };
}

macro_rules! cmp_family {
    ($t:ty, $n:literal, $alias:literal, $gt:ident, $lt:ident, $eq:ident, $ne:ident, $ge:ident, $le:ident) => {
        cmp_fn!($gt, simd_gt, $t, $n, ">", $alias, "false");
        cmp_fn!($lt, simd_lt, $t, $n, "<", $alias, "false");
        cmp_fn!($eq, simd_eq, $t, $n, "==", $alias, "true");
        cmp_fn!($ne, simd_ne, $t, $n, "!=", $alias, "false");
        cmp_fn!($ge, simd_ge, $t, $n, ">=", $alias, "true");
        cmp_fn!($le, simd_le, $t, $n, "<=", $alias, "true");
    };
}

cmp_family!(
    i32, 8, "I32x8", cmp_gt_i32, cmp_lt_i32, cmp_eq_i32, cmp_ne_i32, cmp_ge_i32, cmp_le_i32
);
cmp_family!(
    f32, 8, "F32x8", cmp_gt_f32, cmp_lt_f32, cmp_eq_f32, cmp_ne_f32, cmp_ge_f32, cmp_le_f32
);
cmp_family!(
    i16, 16, "I16x16", cmp_gt_i16, cmp_lt_i16, cmp_eq_i16, cmp_ne_i16, cmp_ge_i16, cmp_le_i16
);
cmp_family!(
    i8, 32, "I8x32", cmp_gt_i8, cmp_lt_i8, cmp_eq_i8, cmp_ne_i8, cmp_ge_i8, cmp_le_i8
);

/// Lanes where either operand is NaN (unordered compare, `_CMP_UNORD_Q`).
///
/// ```
/// use tpt_simd_compare::{cmp_unord_f32, mask_to_bitmask};
/// use tpt_simd_core::F32x8;
/// let a = F32x8::from_array([f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// let b = F32x8::from_array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, f32::NAN]);
/// assert_eq!(mask_to_bitmask(cmp_unord_f32(a, b)), 0b1000_0001);
/// ```
#[inline]
pub fn cmp_unord_f32(a: Simd<f32, 8>, b: Simd<f32, 8>) -> SimdMask<f32, 8> {
    a.is_nan() | b.is_nan()
}

/// `true` if any lane of the mask is set.
///
/// ```
/// use tpt_simd_compare::{mask_any, SimdMask};
/// assert!(!mask_any(SimdMask::<i32, 8>::splat(false)));
/// ```
#[inline]
pub fn mask_any<T: SimdElement, const N: usize>(m: SimdMask<T, N>) -> bool {
    m.any()
}

/// `true` if every lane of the mask is set.
///
/// ```
/// use tpt_simd_compare::{mask_all, SimdMask};
/// assert!(mask_all(SimdMask::<i32, 8>::splat(true)));
/// ```
#[inline]
pub fn mask_all<T: SimdElement, const N: usize>(m: SimdMask<T, N>) -> bool {
    m.all()
}

/// Number of set lanes.
///
/// ```
/// use tpt_simd_compare::{mask_count, SimdMask};
/// assert_eq!(mask_count(SimdMask::<i32, 8>::from_bitmask(0b1011)), 3);
/// ```
#[inline]
pub fn mask_count<T: SimdElement, const N: usize>(m: SimdMask<T, N>) -> usize {
    m.count()
}

/// Pack the mask into an integer, lane 0 in bit 0 (like `movmskps` /
/// `pmovmskb`).
///
/// # Panics
/// If `N > 64`.
///
/// ```
/// use tpt_simd_compare::{mask_to_bitmask, SimdMask};
/// let m = SimdMask::<i32, 4>::from_array([true, false, true, true]);
/// assert_eq!(mask_to_bitmask(m), 0b1101);
/// ```
#[inline]
pub fn mask_to_bitmask<T: SimdElement, const N: usize>(m: SimdMask<T, N>) -> u64 {
    m.to_bitmask()
}

/// Unpack a mask from an integer, bit 0 to lane 0; bits above `N` ignored.
///
/// # Panics
/// If `N > 64`.
///
/// ```
/// use tpt_simd_compare::{mask_from_bitmask, mask_to_bitmask, SimdMask};
/// let m: SimdMask<i8, 32> = mask_from_bitmask(0xF0F0_F0F0);
/// assert_eq!(mask_to_bitmask(m), 0xF0F0_F0F0);
/// ```
#[inline]
pub fn mask_from_bitmask<T: SimdElement, const N: usize>(bits: u64) -> SimdMask<T, N> {
    SimdMask::from_bitmask(bits)
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
    use tpt_simd_core::{F32x8, I8x32, I16x16, I32x8};
    use tpt_simd_testutil::{f32_with_specials, i16_edgy, i32_edgy};

    fn bits<T: Copy, const N: usize>(a: [T; N], b: [T; N], f: impl Fn(T, T) -> bool) -> u64 {
        (0..N).fold(0u64, |m, i| m | (u64::from(f(a[i], b[i])) << i))
    }

    #[test]
    fn nan_semantics() {
        let n = f32::NAN;
        let a = F32x8::from_array([n, 1.0, n, 0.0, -0.0, 2.0, 2.0, 2.0]);
        let b = F32x8::from_array([1.0, n, n, -0.0, 0.0, 1.0, 2.0, 3.0]);
        assert_eq!(mask_to_bitmask(cmp_gt_f32(a, b)), 0b0010_0000);
        assert_eq!(mask_to_bitmask(cmp_lt_f32(a, b)), 0b1000_0000);
        assert_eq!(mask_to_bitmask(cmp_ge_f32(a, b)), 0b0111_1000);
        assert_eq!(mask_to_bitmask(cmp_le_f32(a, b)), 0b1101_1000);
        assert_eq!(mask_to_bitmask(cmp_eq_f32(a, b)), 0b0101_1000);
        assert_eq!(mask_to_bitmask(cmp_ne_f32(a, b)), 0b1010_0111);
        assert_eq!(mask_to_bitmask(cmp_unord_f32(a, b)), 0b0000_0111);
    }

    #[test]
    fn integer_basics() {
        let a = I32x8::from_array([1, 2, 3, 4, 5, 6, 7, i32::MIN]);
        let b = I32x8::splat(4);
        assert_eq!(mask_to_bitmask(cmp_gt_i32(a, b)), 0b0111_0000);
        assert_eq!(mask_to_bitmask(cmp_le_i32(a, b)), 0b1000_1111);
        assert_eq!(mask_count(cmp_eq_i32(a, b)), 1);
        assert!(mask_all(cmp_ge_i16(I16x16::splat(0), I16x16::splat(0))));
        assert!(!mask_any(cmp_lt_i8(I8x32::splat(-1), I8x32::splat(-1))));
        // Comparisons are signed.
        assert!(mask_all(cmp_lt_i8(I8x32::splat(-1), I8x32::splat(0))));
    }

    #[test]
    fn bitmask_roundtrip() {
        let m: SimdMask<i32, 8> = mask_from_bitmask(0xA5);
        assert_eq!(mask_to_bitmask(m), 0xA5);
    }

    proptest! {
        #[test]
        fn i32_vs_scalar(a in lanes::<i32, _, 8>(i32_edgy()), b in lanes::<i32, _, 8>(i32_edgy())) {
            let (x, y) = (a.to_array(), b.to_array());
            prop_assert_eq!(mask_to_bitmask(cmp_gt_i32(a, b)), bits(x, y, |p, q| p > q));
            prop_assert_eq!(mask_to_bitmask(cmp_lt_i32(a, b)), bits(x, y, |p, q| p < q));
            prop_assert_eq!(mask_to_bitmask(cmp_eq_i32(a, b)), bits(x, y, |p, q| p == q));
            prop_assert_eq!(mask_to_bitmask(cmp_ne_i32(a, b)), bits(x, y, |p, q| p != q));
            prop_assert_eq!(mask_to_bitmask(cmp_ge_i32(a, b)), bits(x, y, |p, q| p >= q));
            prop_assert_eq!(mask_to_bitmask(cmp_le_i32(a, b)), bits(x, y, |p, q| p <= q));
        }

        #[test]
        fn f32_vs_scalar(a in lanes::<f32, _, 8>(f32_with_specials()), b in lanes::<f32, _, 8>(f32_with_specials())) {
            let (x, y) = (a.to_array(), b.to_array());
            prop_assert_eq!(mask_to_bitmask(cmp_gt_f32(a, b)), bits(x, y, |p, q| p > q));
            prop_assert_eq!(mask_to_bitmask(cmp_lt_f32(a, b)), bits(x, y, |p, q| p < q));
            prop_assert_eq!(mask_to_bitmask(cmp_eq_f32(a, b)), bits(x, y, |p, q| p == q));
            prop_assert_eq!(mask_to_bitmask(cmp_ne_f32(a, b)), bits(x, y, |p, q| p != q));
            prop_assert_eq!(mask_to_bitmask(cmp_ge_f32(a, b)), bits(x, y, |p, q| p >= q));
            prop_assert_eq!(mask_to_bitmask(cmp_le_f32(a, b)), bits(x, y, |p, q| p <= q));
            prop_assert_eq!(
                mask_to_bitmask(cmp_unord_f32(a, b)),
                bits(x, y, |p: f32, q: f32| p.is_nan() || q.is_nan())
            );
        }

        #[test]
        fn i16_vs_scalar(a in lanes::<i16, _, 16>(i16_edgy()), b in lanes::<i16, _, 16>(i16_edgy())) {
            let (x, y) = (a.to_array(), b.to_array());
            prop_assert_eq!(mask_to_bitmask(cmp_gt_i16(a, b)), bits(x, y, |p, q| p > q));
            prop_assert_eq!(mask_to_bitmask(cmp_lt_i16(a, b)), bits(x, y, |p, q| p < q));
            prop_assert_eq!(mask_to_bitmask(cmp_eq_i16(a, b)), bits(x, y, |p, q| p == q));
            prop_assert_eq!(mask_to_bitmask(cmp_ne_i16(a, b)), bits(x, y, |p, q| p != q));
            prop_assert_eq!(mask_to_bitmask(cmp_ge_i16(a, b)), bits(x, y, |p, q| p >= q));
            prop_assert_eq!(mask_to_bitmask(cmp_le_i16(a, b)), bits(x, y, |p, q| p <= q));
        }

        #[test]
        fn i8_vs_scalar(a in lanes::<i8, _, 32>(any::<i8>()), b in lanes::<i8, _, 32>(any::<i8>())) {
            let (x, y) = (a.to_array(), b.to_array());
            prop_assert_eq!(mask_to_bitmask(cmp_gt_i8(a, b)), bits(x, y, |p, q| p > q));
            prop_assert_eq!(mask_to_bitmask(cmp_lt_i8(a, b)), bits(x, y, |p, q| p < q));
            prop_assert_eq!(mask_to_bitmask(cmp_eq_i8(a, b)), bits(x, y, |p, q| p == q));
            prop_assert_eq!(mask_to_bitmask(cmp_ne_i8(a, b)), bits(x, y, |p, q| p != q));
            prop_assert_eq!(mask_to_bitmask(cmp_ge_i8(a, b)), bits(x, y, |p, q| p >= q));
            prop_assert_eq!(mask_to_bitmask(cmp_le_i8(a, b)), bits(x, y, |p, q| p <= q));
        }
    }
}
