//! SIMD horizontal reductions (sum, min, max, product).
//!
//! A *horizontal* reduction collapses the lanes of one vector into a scalar.
//!
//! ## Reduction order (all functions, all targets)
//!
//! Every reduction uses the same fixed **halving tree**: for a vector of `N`
//! lanes, lane `i` is combined with lane `i + ceil(N/2)` (for `i < floor(N/2)`),
//! leaving `ceil(N/2)` partial results, and the step repeats until one value
//! remains. For `N = 8`:
//!
//! ```text
//! t[i] = l[i]  + l[i+4]   (i = 0..4)
//! u[i] = t[i]  + t[i+2]   (i = 0..2)
//! r    = u[0]  + u[1]
//! ```
//!
//! This is the order an `extract-high-half, op, shuffle, op, ...` hardware
//! reduction uses. Integer sums wrap (so order is irrelevant), integer
//! min/max are order independent, and float sums/products are *reassociated*
//! relative to a left-to-right loop but are bit-for-bit deterministic across
//! targets because the tree is part of the contract. The portable code is the
//! only implementation: LLVM maps the fixed tree to shuffles + vector ops on
//! x86/AArch64, and no `unsafe` is needed.
//!
//! ## NaN policy (float min/max)
//!
//! NaN lanes are **ignored**: the result is the min/max of the non-NaN lanes.
//! If *every* lane is NaN the result is NaN. The sign of a zero result is
//! unspecified when both `+0.0` and `-0.0` are present (but deterministic).
//! `horizontal_sum_f32` and `horizontal_product_f32` propagate NaN normally.
//!
//! ## Scope
//!
//! Required by the spec: [`horizontal_sum_i32`], [`horizontal_sum_f32`],
//! [`horizontal_sum_i16`], [`horizontal_max_i16`], [`horizontal_min_f32`],
//! [`horizontal_product_f32`]. For symmetry we also provide min/max for
//! `i16`/`i32`/`f32`, widening sums for `u8`/`i8`, and the generic
//! [`reduce_sum`], [`reduce_product`], [`reduce_min`], [`reduce_max`] for any
//! lane count.
#![no_std]
#![forbid(unsafe_code)]

#[cfg(feature = "std")]
extern crate std;

use tpt_simd_core::{Simd, SimdElement};

/// Apply `f` over the halving tree described in the crate docs.
#[inline(always)]
fn tree<T: SimdElement, const N: usize>(v: Simd<T, N>, f: impl Fn(T, T) -> T) -> T {
    assert!(N > 0, "cannot reduce a zero-lane vector");
    let mut t = v.0;
    let mut w = N;
    while w > 1 {
        let h = w.div_ceil(2);
        for i in 0..(w - h) {
            t[i] = f(t[i], t[i + h]);
        }
        w = h;
    }
    t[0]
}

/// NaN-ignoring `f32` minimum (NaN only if both are NaN).
#[inline(always)]
fn fmin(a: f32, b: f32) -> f32 {
    if a.is_nan() || b < a { b } else { a }
}

/// NaN-ignoring `f32` maximum (NaN only if both are NaN).
#[inline(always)]
fn fmax(a: f32, b: f32) -> f32 {
    if a.is_nan() || b > a { b } else { a }
}

/// Sum of all lanes of any element type, in halving-tree order.
///
/// Integer lanes wrap; float lanes use the documented fixed tree.
///
/// # Panics
/// If `N == 0`.
///
/// # Examples
/// ```
/// use tpt_simd_horizontal::reduce_sum;
/// use tpt_simd_core::Simd;
/// assert_eq!(reduce_sum(Simd::<i32, 4>::from_array([1, 2, 3, 4])), 10);
/// ```
#[inline]
pub fn reduce_sum<T: SimdElement, const N: usize>(v: Simd<T, N>) -> T {
    tree(v, T::lane_add)
}

/// Product of all lanes of any element type, in halving-tree order (integers wrap).
///
/// # Panics
/// If `N == 0`.
///
/// # Examples
/// ```
/// use tpt_simd_horizontal::reduce_product;
/// use tpt_simd_core::Simd;
/// assert_eq!(reduce_product(Simd::<i32, 4>::from_array([1, 2, 3, 4])), 24);
/// ```
#[inline]
pub fn reduce_product<T: SimdElement, const N: usize>(v: Simd<T, N>) -> T {
    tree(v, T::lane_mul)
}

/// Minimum of all lanes of any element type (float NaN lanes ignored).
///
/// # Panics
/// If `N == 0`.
///
/// # Examples
/// ```
/// use tpt_simd_horizontal::reduce_min;
/// use tpt_simd_core::Simd;
/// assert_eq!(reduce_min(Simd::<i16, 4>::from_array([3, -2, 9, 0])), -2);
/// ```
#[inline]
pub fn reduce_min<T: SimdElement, const N: usize>(v: Simd<T, N>) -> T {
    tree(v, |a, b| {
        // NaN-ignoring for floats (unordered with itself), plain comparison for ints.
        if a.partial_cmp(&a).is_none() || b < a {
            b
        } else {
            a
        }
    })
}

/// Maximum of all lanes of any element type (float NaN lanes ignored).
///
/// # Panics
/// If `N == 0`.
///
/// # Examples
/// ```
/// use tpt_simd_horizontal::reduce_max;
/// use tpt_simd_core::Simd;
/// assert_eq!(reduce_max(Simd::<i16, 4>::from_array([3, -2, 9, 0])), 9);
/// ```
#[inline]
pub fn reduce_max<T: SimdElement, const N: usize>(v: Simd<T, N>) -> T {
    tree(v, |a, b| {
        if a.partial_cmp(&a).is_none() || b > a {
            b
        } else {
            a
        }
    })
}

/// Wrapping sum of the 8 lanes of an `i32` vector.
///
/// Performance: a few shuffles and adds; no per-lane loop on AVX2/NEON.
///
/// # Examples
/// ```
/// use tpt_simd_horizontal::horizontal_sum_i32;
/// use tpt_simd_core::I32x8;
/// assert_eq!(horizontal_sum_i32(I32x8::from_array([1, 2, 3, 4, 5, 6, 7, 8])), 36);
/// assert_eq!(horizontal_sum_i32(I32x8::splat(i32::MAX)), -8); // wraps
/// ```
#[inline]
pub fn horizontal_sum_i32(v: Simd<i32, 8>) -> i32 {
    tree(v, i32::wrapping_add)
}

/// Sum of the 8 lanes of an `f32` vector in the fixed pairwise order
/// `((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))` (see the crate docs).
///
/// The order is identical on every target, so the result is deterministic.
/// NaN and infinities propagate as in ordinary IEEE addition.
///
/// # Examples
/// ```
/// use tpt_simd_horizontal::horizontal_sum_f32;
/// use tpt_simd_core::F32x8;
/// assert_eq!(horizontal_sum_f32(F32x8::from_array([1.0; 8])), 8.0);
/// ```
#[inline]
pub fn horizontal_sum_f32(v: Simd<f32, 8>) -> f32 {
    tree(v, |a, b| a + b)
}

/// Widening sum of the 16 lanes of an `i16` vector into an `i32`.
///
/// Never overflows: `16 * 32768 < 2^31`.
///
/// # Examples
/// ```
/// use tpt_simd_horizontal::horizontal_sum_i16;
/// use tpt_simd_core::I16x16;
/// assert_eq!(horizontal_sum_i16(I16x16::splat(i16::MAX)), 16 * 32767);
/// assert_eq!(horizontal_sum_i16(I16x16::splat(i16::MIN)), -16 * 32768);
/// ```
#[inline]
pub fn horizontal_sum_i16(v: Simd<i16, 16>) -> i32 {
    tree(v.cast::<i32>(), i32::wrapping_add)
}

/// Widening sum of the 32 lanes of a `u8` vector into a `u32` (no overflow).
///
/// # Examples
/// ```
/// use tpt_simd_horizontal::horizontal_sum_u8;
/// use tpt_simd_core::U8x32;
/// assert_eq!(horizontal_sum_u8(U8x32::splat(255)), 32 * 255);
/// ```
#[inline]
pub fn horizontal_sum_u8(v: Simd<u8, 32>) -> u32 {
    tree(v.cast::<u32>(), u32::wrapping_add)
}

/// Widening sum of the 32 lanes of an `i8` vector into an `i32` (no overflow).
///
/// # Examples
/// ```
/// use tpt_simd_horizontal::horizontal_sum_i8;
/// use tpt_simd_core::I8x32;
/// assert_eq!(horizontal_sum_i8(I8x32::splat(-128)), -32 * 128);
/// ```
#[inline]
pub fn horizontal_sum_i8(v: Simd<i8, 32>) -> i32 {
    tree(v.cast::<i32>(), i32::wrapping_add)
}

/// Maximum of the 16 lanes of an `i16` vector.
///
/// # Examples
/// ```
/// use tpt_simd_horizontal::horizontal_max_i16;
/// use tpt_simd_core::I16x16;
/// assert_eq!(horizontal_max_i16(I16x16::from_fn(|i| i as i16 - 5)), 10);
/// ```
#[inline]
pub fn horizontal_max_i16(v: Simd<i16, 16>) -> i16 {
    tree(v, |a, b| if b > a { b } else { a })
}

/// Minimum of the 16 lanes of an `i16` vector.
///
/// # Examples
/// ```
/// use tpt_simd_horizontal::horizontal_min_i16;
/// use tpt_simd_core::I16x16;
/// assert_eq!(horizontal_min_i16(I16x16::from_fn(|i| i as i16 - 5)), -5);
/// ```
#[inline]
pub fn horizontal_min_i16(v: Simd<i16, 16>) -> i16 {
    tree(v, |a, b| if b < a { b } else { a })
}

/// Maximum of the 8 lanes of an `i32` vector.
///
/// # Examples
/// ```
/// use tpt_simd_horizontal::horizontal_max_i32;
/// use tpt_simd_core::I32x8;
/// assert_eq!(horizontal_max_i32(I32x8::from_array([1, 9, 3, 4, 5, 6, 7, 8])), 9);
/// ```
#[inline]
pub fn horizontal_max_i32(v: Simd<i32, 8>) -> i32 {
    tree(v, |a, b| if b > a { b } else { a })
}

/// Minimum of the 8 lanes of an `i32` vector.
///
/// # Examples
/// ```
/// use tpt_simd_horizontal::horizontal_min_i32;
/// use tpt_simd_core::I32x8;
/// assert_eq!(horizontal_min_i32(I32x8::from_array([1, 9, -3, 4, 5, 6, 7, 8])), -3);
/// ```
#[inline]
pub fn horizontal_min_i32(v: Simd<i32, 8>) -> i32 {
    tree(v, |a, b| if b < a { b } else { a })
}

/// Minimum of the 8 lanes of an `f32` vector; NaN lanes are ignored and an
/// all-NaN vector gives NaN (see the crate docs).
///
/// # Examples
/// ```
/// use tpt_simd_horizontal::horizontal_min_f32;
/// use tpt_simd_core::F32x8;
/// let v = F32x8::from_array([f32::NAN, 3.0, 2.0, f32::NAN, 5.0, 6.0, 7.0, 8.0]);
/// assert_eq!(horizontal_min_f32(v), 2.0);
/// assert!(horizontal_min_f32(F32x8::splat(f32::NAN)).is_nan());
/// ```
#[inline]
pub fn horizontal_min_f32(v: Simd<f32, 8>) -> f32 {
    tree(v, fmin)
}

/// Maximum of the 8 lanes of an `f32` vector; NaN lanes are ignored and an
/// all-NaN vector gives NaN (see the crate docs).
///
/// # Examples
/// ```
/// use tpt_simd_horizontal::horizontal_max_f32;
/// use tpt_simd_core::F32x8;
/// let v = F32x8::from_array([f32::NAN, 3.0, 2.0, f32::NAN, 5.0, 6.0, 7.0, 8.0]);
/// assert_eq!(horizontal_max_f32(v), 8.0);
/// assert!(horizontal_max_f32(F32x8::splat(f32::NAN)).is_nan());
/// ```
#[inline]
pub fn horizontal_max_f32(v: Simd<f32, 8>) -> f32 {
    tree(v, fmax)
}

/// Product of the 8 lanes of an `f32` vector, in the same halving-tree order
/// as [`horizontal_sum_f32`]. NaN propagates; overflow gives `inf`.
///
/// # Examples
/// ```
/// use tpt_simd_horizontal::horizontal_product_f32;
/// use tpt_simd_core::F32x8;
/// assert_eq!(horizontal_product_f32(F32x8::from_array([1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0])), 24.0);
/// ```
#[inline]
pub fn horizontal_product_f32(v: Simd<f32, 8>) -> f32 {
    tree(v, |a, b| a * b)
}

#[cfg(test)]
mod tests;
