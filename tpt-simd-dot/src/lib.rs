//! SIMD dot product variants.
//!
//! | function | element | accumulator | order |
//! |---|---|---|---|
//! | [`dot_product_i16`] | `i16` | `i32`, wrapping | irrelevant (wrapping add is associative) |
//! | [`dot_product_saturating_i16`] | `i16` | `i32`, saturating | fixed, see function docs |
//! | [`dot_product_f32`] | `f32` | 32 independent `f32` partial sums | fixed, see function docs |
//! | [`dot_product_complex_f32`] | `ComplexSimd<f32, 8>` | one running complex sum per lane | fixed, see function docs |
//!
//! ## Length policy
//!
//! The two input slices must have the same length; otherwise the function
//! **panics** with a message naming the function and both lengths. Empty
//! input is valid and returns zero. Any length is accepted (there is no
//! multiple-of-vector-width requirement); tails are handled internally.
//!
//! ## Determinism
//!
//! The portable implementation in [`portable`] is the behavioural reference.
//! Where an AVX2 fast path exists (currently only [`dot_product_i16`], on
//! x86_64 built with `avx2` and without the `scalar-only` feature) it is
//! **bit-identical** to the portable one. The `f32` dot product uses separate
//! multiply and add (never FMA) and a documented fixed accumulation order, so
//! its result is the same on every target and build configuration.
//!
//! ## Example (FLAC-style LPC)
//!
//! ```
//! use tpt_simd_dot::dot_product_i16;
//!
//! let a: Vec<i16> = (0..256).collect();
//! let b: Vec<i16> = (0..256).collect();
//! // sum of k^2 for k in 0..256
//! assert_eq!(dot_product_i16(&a, &b), 5_559_680);
//! ```
#![no_std]
#![cfg_attr(
    not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    )),
    forbid(unsafe_code)
)]

#[cfg(feature = "std")]
extern crate std;

use tpt_simd_core::ComplexSimd;

pub mod portable;

#[cfg(all(
    not(feature = "scalar-only"),
    target_arch = "x86_64",
    target_feature = "avx2"
))]
mod x86;

#[inline(never)]
#[cold]
#[track_caller]
fn len_mismatch(name: &str, a: usize, b: usize) -> ! {
    panic!("{name}: length mismatch ({a} vs {b})");
}

/// Dot product of two `i16` slices, accumulated in a wrapping `i32`.
///
/// Each product `a[i] * b[i]` is exact in `i32` (at most `2^30`); the products
/// are summed with **wrapping** `i32` addition, so on overflow the result is
/// the true sum modulo `2^32` (two's complement). Because wrapping addition is
/// associative the result does not depend on the SIMD summation order.
///
/// Performance: on AVX2 builds this uses `vpmaddwd` (pairwise widening
/// multiply-add) with two accumulators, 32 elements per iteration.
/// See the crate benchmarks for measured numbers; LLVM also auto-vectorises
/// a plain scalar loop of this shape, so the gain over it is modest.
///
/// # Panics
/// If `a.len() != b.len()`.
///
/// # Examples
/// ```
/// use tpt_simd_dot::dot_product_i16;
/// assert_eq!(dot_product_i16(&[1, 2, 3], &[4, 5, 6]), 32);
/// assert_eq!(dot_product_i16(&[], &[]), 0);
/// // (-32768)^2 * 2 = 2^31 wraps to i32::MIN
/// assert_eq!(dot_product_i16(&[i16::MIN; 2], &[i16::MIN; 2]), i32::MIN);
/// ```
#[inline]
pub fn dot_product_i16(a: &[i16], b: &[i16]) -> i32 {
    if a.len() != b.len() {
        len_mismatch("dot_product_i16", a.len(), b.len());
    }
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    ))]
    {
        // SAFETY: this block is compiled only when AVX2 is enabled for the
        // whole crate, and the slice lengths were checked equal above.
        unsafe { x86::dot_i16(a, b) }
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    )))]
    {
        portable::dot_product_i16(a, b)
    }
}

/// Dot product of two `i16` slices with a **saturating** `i32` accumulator.
///
/// Exactly what saturates: every product `a[i] * b[i]` is computed exactly in
/// `i32` (never saturates, `|p| <= 2^30`). The products are added with
/// `i32::saturating_add` into 8 independent lane accumulators, where element
/// `i` goes to lane `i % 8` (processing elements in increasing `i`). The 8
/// lanes are then combined by the halving tree
/// `(l0+l4, l1+l5, l2+l6, l3+l7) -> (t0+t2, t1+t3) -> u0+u1`, again with
/// `saturating_add`. There is no intermediate widening, so an accumulator
/// that saturates stays clamped even if later products would pull it back;
/// results can therefore differ from the (non-saturating) true sum clamped
/// once at the end. The order is part of the contract and identical on every
/// target.
///
/// Performance: portable path only (a saturating 32-bit lane add has no
/// direct AVX2 instruction); LLVM vectorises the lane loop.
///
/// # Panics
/// If `a.len() != b.len()`.
///
/// # Examples
/// ```
/// use tpt_simd_dot::dot_product_saturating_i16;
/// assert_eq!(dot_product_saturating_i16(&[1, 2, 3], &[4, 5, 6]), 32);
/// let big = [i16::MIN; 64];
/// assert_eq!(dot_product_saturating_i16(&big, &big), i32::MAX);
/// ```
#[inline]
pub fn dot_product_saturating_i16(a: &[i16], b: &[i16]) -> i32 {
    if a.len() != b.len() {
        len_mismatch("dot_product_saturating_i16", a.len(), b.len());
    }
    portable::dot_product_saturating_i16(a, b)
}

/// Dot product of two `f32` slices.
///
/// **Accumulation order (fixed, identical on every target):** there are 32
/// independent `f32` partial sums. Element `i` is multiplied (`a[i] * b[i]`,
/// rounded) and then added (rounded) to partial sum `i % 32`, in increasing
/// `i`; i.e. 4 accumulator vectors of 8 lanes, element `i` going to
/// accumulator `(i / 8) % 4`, lane `i % 8`. **No FMA is used** (multiply and
/// add round separately), so SIMD and portable paths are bit-identical. The
/// accumulators are then combined as `(acc0 + acc1) + (acc2 + acc3)`
/// lane-wise, followed by the pairwise lane tree of
/// [`horizontal_sum_f32`](tpt_simd_horizontal::horizontal_sum_f32).
///
/// This reassociates the sum relative to a left-to-right loop, so results can
/// differ from a naive scalar loop by normal floating-point rounding (usually
/// *more* accurate, since each partial sum is shorter). NaN and infinities
/// propagate as in IEEE arithmetic. Empty input returns `0.0`.
///
/// Performance: portable path only. An explicit AVX2 version (same four
/// `__m256` accumulators) was benchmarked and was no faster than the
/// auto-vectorised portable code, so it was not kept. The multi-accumulator
/// order is what removes the serial add dependency (about 10x vs a naive
/// loop, which LLVM may not reassociate); see `benches/dot.rs`.
///
/// # Panics
/// If `a.len() != b.len()`.
///
/// # Examples
/// ```
/// use tpt_simd_dot::dot_product_f32;
/// assert_eq!(dot_product_f32(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
/// assert!(dot_product_f32(&[f32::NAN], &[1.0]).is_nan());
/// ```
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        len_mismatch("dot_product_f32", a.len(), b.len());
    }
    portable::dot_product_f32(a, b)
}

/// Lane-wise complex dot product: `sum_i a[i] * b[i]` (**not** conjugated).
///
/// Each element is a [`ComplexSimd<f32, 8>`] holding 8 independent complex
/// numbers. The result's lane `j` is `sum_i a[i][j] * b[i][j]` using the
/// ordinary complex product `(ar + ai·i)(br + bi·i) = (ar·br − ai·bi) +
/// (ar·bi + ai·br)·i` (unfused arithmetic, as in
/// [`ComplexSimd::mul`]). There is no reduction across the 8 lanes; reduce
/// them with the `tpt-simd-horizontal` functions if needed. For the Hermitian
/// inner product conjugate one input first ([`ComplexSimd::conj`]).
///
/// **Accumulation order (fixed):** per lane, products are added to a single
/// running sum in increasing index order (`acc = acc + a[i]*b[i]`, each
/// product and sum rounded separately, no FMA). Empty input returns zero.
///
/// Performance: portable path only; LLVM vectorises the split real/imag
/// arithmetic across the 8 lanes, and the loop is throughput- rather than
/// latency-bound, so it is on par with a hand-written per-lane scalar loop.
///
/// # Panics
/// If `a.len() != b.len()`.
///
/// # Examples
/// ```
/// use tpt_simd_core::{ComplexSimd, F32x8};
/// use tpt_simd_dot::dot_product_complex_f32;
/// let a = [ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0))];
/// let b = [ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0))];
/// let r = dot_product_complex_f32(&a, &b); // (1+2i)(3+4i) = -5+10i
/// assert_eq!(r.real.to_array(), [-5.0; 8]);
/// assert_eq!(r.imag.to_array(), [10.0; 8]);
/// ```
#[inline]
pub fn dot_product_complex_f32(
    a: &[ComplexSimd<f32, 8>],
    b: &[ComplexSimd<f32, 8>],
) -> ComplexSimd<f32, 8> {
    if a.len() != b.len() {
        len_mismatch("dot_product_complex_f32", a.len(), b.len());
    }
    portable::dot_product_complex_f32(a, b)
}

#[cfg(test)]
mod tests;
