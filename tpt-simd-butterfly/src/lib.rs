//! SIMD FFT/DCT butterfly operations.
//!
//! A radix-2 butterfly maps `(a, b)` to `(a + b, a - b)`. The functions here
//! work in place on vectors of eight `f32`, sixteen `i16` or split complex
//! vectors. They are written as straight lane loops that LLVM compiles to one
//! vector add and one vector sub per register, so no hand-written intrinsics
//! (and no `unsafe`) are needed.
//!
//! ## Deviation from the spec
//!
//! The spec's `butterfly_with_twiddle_f32` takes real vectors with a complex
//! twiddle, which is ambiguous. This crate provides two unambiguous forms:
//!
//! * [`butterfly_with_twiddle_complex_f32`]: complex `a`, `b` and twiddle,
//!   `(a + b*w, a - b*w)` (the decimation-in-time FFT butterfly).
//! * [`butterfly_with_twiddle_f32`]: real vectors with a real twiddle,
//!   `b' = b*w; (a + b', a - b')`.
//!
//! ## Example (spec Appendix B.5)
//!
//! ```
//! use tpt_simd_core::F32x8;
//! use tpt_simd_butterfly::butterfly_f32;
//!
//! let mut a = F32x8::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
//! let mut b = F32x8::from_array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
//! butterfly_f32(&mut a, &mut b);
//! assert_eq!(a.to_array(), [9.0; 8]);
//! assert_eq!(b.to_array(), [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]);
//! ```
#![no_std]
#![forbid(unsafe_code)]

#[cfg(any(feature = "std", test))]
extern crate std;

use tpt_simd_core::{ComplexSimd, Simd, SimdElement, SimdInt};

/// In-place radix-2 butterfly on 8 `f32` lanes: `(a, b) <- (a + b, a - b)`.
///
/// # Performance
/// One `vaddps` and one `vsubps` on AVX2.
///
/// # Panics
/// Never.
///
/// # Example
/// ```
/// use tpt_simd_core::F32x8;
/// use tpt_simd_butterfly::butterfly_f32;
/// let (mut a, mut b) = (F32x8::splat(3.0), F32x8::splat(1.0));
/// butterfly_f32(&mut a, &mut b);
/// assert_eq!((a, b), (F32x8::splat(4.0), F32x8::splat(2.0)));
/// ```
#[inline]
pub fn butterfly_f32(a: &mut Simd<f32, 8>, b: &mut Simd<f32, 8>) {
    butterfly_generic(a, b);
}

/// In-place butterfly on 16 `i16` lanes with **wrapping** arithmetic:
/// `(a, b) <- (a + b, a - b)` modulo 2^16.
///
/// See [`butterfly_i16_saturating`] for the clamping variant.
///
/// # Performance
/// One `vpaddw` and one `vpsubw` on AVX2.
///
/// # Panics
/// Never.
///
/// # Example
/// ```
/// use tpt_simd_core::I16x16;
/// use tpt_simd_butterfly::butterfly_i16;
/// let (mut a, mut b) = (I16x16::splat(i16::MAX), I16x16::splat(1));
/// butterfly_i16(&mut a, &mut b);
/// assert_eq!(a, I16x16::splat(i16::MIN)); // wrapped
/// assert_eq!(b, I16x16::splat(i16::MAX - 1));
/// ```
#[inline]
pub fn butterfly_i16(a: &mut Simd<i16, 16>, b: &mut Simd<i16, 16>) {
    butterfly_generic(a, b);
}

/// In-place butterfly on 16 `i16` lanes with **saturating** arithmetic.
///
/// # Performance
/// One `vpaddsw` and one `vpsubsw` on AVX2.
///
/// # Panics
/// Never.
///
/// # Example
/// ```
/// use tpt_simd_core::I16x16;
/// use tpt_simd_butterfly::butterfly_i16_saturating;
/// let (mut a, mut b) = (I16x16::splat(i16::MAX), I16x16::splat(-1));
/// butterfly_i16_saturating(&mut a, &mut b);
/// assert_eq!(a, I16x16::splat(i16::MAX - 1));
/// assert_eq!(b, I16x16::splat(i16::MAX)); // clamped
/// ```
#[inline]
pub fn butterfly_i16_saturating(a: &mut Simd<i16, 16>, b: &mut Simd<i16, 16>) {
    let (s, d) = (a.sat_add(*b), a.sat_sub(*b));
    *a = s;
    *b = d;
}

/// Generic radix-2 butterfly `(a, b) <- (a + b, a - b)` for any lane type and
/// width (integers wrap, floats follow IEEE).
///
/// # Performance
/// Same as the concrete versions; this is what they call.
///
/// # Panics
/// Never.
///
/// # Example
/// ```
/// use tpt_simd_core::Simd;
/// use tpt_simd_butterfly::butterfly_generic;
/// let mut a = Simd::<i32, 4>::from_array([1, 2, 3, 4]);
/// let mut b = Simd::<i32, 4>::splat(1);
/// butterfly_generic(&mut a, &mut b);
/// assert_eq!(a.to_array(), [2, 3, 4, 5]);
/// assert_eq!(b.to_array(), [0, 1, 2, 3]);
/// ```
#[inline]
pub fn butterfly_generic<T: SimdElement, const N: usize>(a: &mut Simd<T, N>, b: &mut Simd<T, N>) {
    let (s, d) = (*a + *b, *a - *b);
    *a = s;
    *b = d;
}

/// Generic saturating butterfly for integer lanes of any width.
///
/// # Performance
/// Same as [`butterfly_i16_saturating`].
///
/// # Panics
/// Never.
///
/// # Example
/// ```
/// use tpt_simd_core::Simd;
/// use tpt_simd_butterfly::butterfly_generic_saturating;
/// let mut a = Simd::<i8, 4>::splat(100);
/// let mut b = Simd::<i8, 4>::splat(100);
/// butterfly_generic_saturating(&mut a, &mut b);
/// assert_eq!(a.to_array(), [127; 4]);
/// assert_eq!(b.to_array(), [0; 4]);
/// ```
#[inline]
pub fn butterfly_generic_saturating<T: SimdInt, const N: usize>(
    a: &mut Simd<T, N>,
    b: &mut Simd<T, N>,
) {
    let (s, d) = (a.sat_add(*b), a.sat_sub(*b));
    *a = s;
    *b = d;
}

/// In-place radix-2 butterfly on 8 complex `f32` lanes (split layout):
/// `(a, b) <- (a + b, a - b)`.
///
/// # Performance
/// Two `vaddps` and two `vsubps` on AVX2.
///
/// # Panics
/// Never.
///
/// # Example
/// ```
/// use tpt_simd_core::{ComplexSimd, F32x8};
/// use tpt_simd_butterfly::butterfly_complex_f32;
/// let mut a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0));
/// let mut b = ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(5.0));
/// butterfly_complex_f32(&mut a, &mut b);
/// assert_eq!(a.real, F32x8::splat(4.0));
/// assert_eq!(a.imag, F32x8::splat(7.0));
/// assert_eq!(b.real, F32x8::splat(-2.0));
/// assert_eq!(b.imag, F32x8::splat(-3.0));
/// ```
#[inline]
pub fn butterfly_complex_f32(a: &mut ComplexSimd<f32, 8>, b: &mut ComplexSimd<f32, 8>) {
    butterfly_generic(&mut a.real, &mut b.real);
    butterfly_generic(&mut a.imag, &mut b.imag);
}

/// Real-twiddle butterfly on 8 `f32` lanes: `b' = b * w`, then
/// `(a, b) <- (a + b', a - b')`.
///
/// The spec's version of this function was ambiguous (real vectors with a
/// complex twiddle); for complex data use
/// [`butterfly_with_twiddle_complex_f32`].
///
/// # Performance
/// One `vmulps`, one `vaddps`, one `vsubps` (unfused, so results do not
/// depend on whether FMA is available).
///
/// # Panics
/// Never.
///
/// # Example
/// ```
/// use tpt_simd_core::F32x8;
/// use tpt_simd_butterfly::butterfly_with_twiddle_f32;
/// let (mut a, mut b) = (F32x8::splat(10.0), F32x8::splat(2.0));
/// butterfly_with_twiddle_f32(&mut a, &mut b, F32x8::splat(3.0));
/// assert_eq!((a, b), (F32x8::splat(16.0), F32x8::splat(4.0)));
/// ```
#[inline]
pub fn butterfly_with_twiddle_f32(
    a: &mut Simd<f32, 8>,
    b: &mut Simd<f32, 8>,
    twiddle: Simd<f32, 8>,
) {
    *b *= twiddle;
    butterfly_generic(a, b);
}

/// Decimation-in-time FFT butterfly on 8 complex `f32` lanes with a complex
/// twiddle: `t = b * w; (a, b) <- (a + t, a - t)`.
///
/// The complex product uses the unfused formula of
/// [`ComplexSimd::mul`](tpt_simd_core::ComplexSimd), so results are identical
/// on every target.
///
/// # Performance
/// 4 multiplies, 2 add/sub for the twiddle, then 2 adds and 2 subs.
///
/// # Panics
/// Never.
///
/// # Example
/// ```
/// use tpt_simd_core::{ComplexSimd, F32x8};
/// use tpt_simd_butterfly::butterfly_with_twiddle_complex_f32;
/// let mut a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(0.0));
/// let mut b = ComplexSimd::new(F32x8::splat(0.0), F32x8::splat(1.0)); // i
/// let w = ComplexSimd::new(F32x8::splat(0.0), F32x8::splat(1.0)); // i
/// butterfly_with_twiddle_complex_f32(&mut a, &mut b, w); // b*w = -1
/// assert_eq!(a.real, F32x8::splat(0.0));
/// assert_eq!(b.real, F32x8::splat(2.0));
/// ```
#[inline]
pub fn butterfly_with_twiddle_complex_f32(
    a: &mut ComplexSimd<f32, 8>,
    b: &mut ComplexSimd<f32, 8>,
    twiddle: ComplexSimd<f32, 8>,
) {
    *b = b.mul(twiddle);
    butterfly_complex_f32(a, b);
}

#[cfg(test)]
mod tests;
