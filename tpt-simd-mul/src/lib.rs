//! SIMD multiply variants (high/low/widening/complex).
//!
//! Implemented:
//! * [`complex_mul_f32`] (plus its portable reference
//!   [`complex_mul_f32_portable`]) — fused complex multiply of 8 `f32` lanes.
//! * [`mul_hi_i16`] — high 16 bits of the 32-bit `i16` product.
//! * [`mul_lo_i32`] — low 32 bits of the 64-bit `i32` product.
//! * [`mul_hi_i32`] — high 32 bits of the 64-bit `i32` product.
//! * [`mul_add_sub_f32`] — `a*b + c` on even lanes, `a*b - c` on odd lanes.
//! * [`mul_widen_i16`], [`mul_widen_i32`] — full-width widening multiplies.
//!
//! ## Rounding rule for complex multiply
//!
//! `complex_mul_f32` is *always* computed as
//!
//! ```text
//! re = fma(a.re, b.re, -(a.im * b.im))
//! im = fma(a.re, b.im,   a.im * b.re )
//! ```
//!
//! This is exactly what the AVX2+FMA path (`vmulps` + `vfmsub`/`vfmadd`)
//! computes, and the portable path uses `Simd::mul_add` (a correctly rounded
//! `libm::fmaf`), so results are bit-identical on every target. The price is
//! that targets without hardware FMA pay for a software `fmaf`; if you only
//! need speed there use the unfused [`ComplexSimd::mul`] from `tpt-simd-core`
//! (its results differ from this function in the last bit).
#![no_std]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

pub use tpt_simd_core::ComplexSimd;

use tpt_simd_core::Simd;

/// High 16 bits of the signed 32-bit product of each `i16` lane pair
/// (`(a * b) >> 16`, like `pmulhw`).
///
/// Performance: lowers to `vpmulhw` on AVX2 / `sqdmulh`-style sequences on
/// NEON via auto-vectorisation of the lane loop.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::I16x16;
/// use tpt_simd_mul::mul_hi_i16;
/// let r = mul_hi_i16(I16x16::splat(i16::MIN), I16x16::splat(i16::MIN));
/// assert_eq!(r, I16x16::splat(0x4000));
/// ```
#[inline]
pub fn mul_hi_i16(a: Simd<i16, 16>, b: Simd<i16, 16>) -> Simd<i16, 16> {
    a.zip_with(b, |x, y| ((i32::from(x) * i32::from(y)) >> 16) as i16)
}

/// Low 32 bits of the 64-bit product of each `i32` lane pair (wrapping
/// multiply, like `pmulld`).
///
/// Performance: one `vpmulld` on AVX2 (10-cycle latency on Intel cores).
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_mul::mul_lo_i32;
/// assert_eq!(mul_lo_i32(I32x8::splat(65536), I32x8::splat(65536)), I32x8::splat(0));
/// ```
#[inline]
pub fn mul_lo_i32(a: Simd<i32, 8>, b: Simd<i32, 8>) -> Simd<i32, 8> {
    a.zip_with(b, i32::wrapping_mul)
}

/// High 32 bits of the signed 64-bit product of each `i32` lane pair
/// (`(a * b) >> 32`). AVX2 has no direct instruction (`_mm256_mulhi_epi32`
/// does not exist); LLVM synthesises it from `vpmuldq` and shuffles.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_mul::mul_hi_i32;
/// assert_eq!(mul_hi_i32(I32x8::splat(i32::MIN), I32x8::splat(i32::MIN)), I32x8::splat(1 << 30));
/// ```
#[inline]
pub fn mul_hi_i32(a: Simd<i32, 8>, b: Simd<i32, 8>) -> Simd<i32, 8> {
    a.zip_with(b, |x, y| ((i64::from(x) * i64::from(y)) >> 32) as i32)
}

/// Widening multiply: full 32-bit product of each `i16` lane pair.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::{I16x8, I32x8};
/// use tpt_simd_mul::mul_widen_i16;
/// assert_eq!(mul_widen_i16(I16x8::splat(i16::MIN), I16x8::splat(i16::MIN)), I32x8::splat(1 << 30));
/// ```
#[inline]
pub fn mul_widen_i16(a: Simd<i16, 8>, b: Simd<i16, 8>) -> Simd<i32, 8> {
    a.zip_with(b, |x, y| i32::from(x) * i32::from(y))
}

/// Widening multiply: full 64-bit product of each `i32` lane pair.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::{I32x4, I64x4};
/// use tpt_simd_mul::mul_widen_i32;
/// assert_eq!(mul_widen_i32(I32x4::splat(i32::MIN), I32x4::splat(i32::MIN)), I64x4::splat(1 << 62));
/// ```
#[inline]
pub fn mul_widen_i32(a: Simd<i32, 4>, b: Simd<i32, 4>) -> Simd<i64, 4> {
    a.zip_with(b, |x, y| i64::from(x) * i64::from(y))
}

/// `a*b + c` on even lanes and `a*b - c` on odd lanes, each fused (single
/// rounding, bit-identical on every target).
///
/// Note this is the *opposite* lane convention to Intel's
/// `_mm256_fmaddsub_ps` (which subtracts on even lanes); it matches the
/// function name ("add" first, then "sub").
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::F32x8;
/// use tpt_simd_mul::mul_add_sub_f32;
/// let r = mul_add_sub_f32(F32x8::splat(2.0), F32x8::splat(3.0), F32x8::splat(1.0));
/// assert_eq!(r.to_array(), [7.0, 5.0, 7.0, 5.0, 7.0, 5.0, 7.0, 5.0]);
/// ```
#[inline]
pub fn mul_add_sub_f32(a: Simd<f32, 8>, b: Simd<f32, 8>, c: Simd<f32, 8>) -> Simd<f32, 8> {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        mul_add_sub_f32_avx2(a, b, c)
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        mul_add_sub_f32_portable(a, b, c)
    }
}

/// Portable reference for [`mul_add_sub_f32`] (one software `fmaf` per lane
/// without hardware FMA).
#[inline]
#[allow(dead_code)] // unused in non-test AVX2+FMA builds
fn mul_add_sub_f32_portable(a: Simd<f32, 8>, b: Simd<f32, 8>, c: Simd<f32, 8>) -> Simd<f32, 8> {
    let sign = Simd::from_fn(|i| if i % 2 == 0 { 1.0 } else { -1.0 });
    a.mul_add(b, c * sign)
}

#[cfg(all(
    not(feature = "scalar-only"),
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline]
fn mul_add_sub_f32_avx2(a: Simd<f32, 8>, b: Simd<f32, 8>, c: Simd<f32, 8>) -> Simd<f32, 8> {
    use core::arch::x86_64::*;
    let (a, b, c) = (a.to_array(), b.to_array(), c.to_array());
    let mut out = [0.0f32; 8];
    // SAFETY: this block is only compiled when `avx2` and `fma` are enabled
    // at compile time, so the intrinsics are available. All pointers come
    // from `[f32; 8]` arrays (32 bytes) and the loads/stores are unaligned.
    unsafe {
        // Negating `c` on odd lanes (xor of the sign bit) is exact, and
        // `fma(a, b, -c)` equals `a*b - c` with a single rounding, matching
        // the portable `mul_add(a, b, c * -1.0)`.
        let flip = _mm256_castsi256_ps(_mm256_setr_epi32(
            0,
            i32::MIN,
            0,
            i32::MIN,
            0,
            i32::MIN,
            0,
            i32::MIN,
        ));
        let cv = _mm256_xor_ps(_mm256_loadu_ps(c.as_ptr()), flip);
        let r = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr()), _mm256_loadu_ps(b.as_ptr()), cv);
        _mm256_storeu_ps(out.as_mut_ptr(), r);
    }
    Simd::from_array(out)
}

/// True when the intrinsic fast path is compiled in.
pub const COMPLEX_MUL_F32_USES_INTRINSICS: bool = cfg!(all(
    not(feature = "scalar-only"),
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
));

/// Portable (reference) complex multiply of 8 `f32` complex lanes using the
/// fused rule described in the [crate docs](crate).
///
/// Performance: slow without hardware FMA (software `fmaf`); it exists as the
/// behavioural reference for [`complex_mul_f32`].
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_mul::{complex_mul_f32_portable, ComplexSimd};
/// use tpt_simd_core::F32x8;
/// let a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0));
/// let b = ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0));
/// let r = complex_mul_f32_portable(a, b);
/// assert_eq!(r.real.to_array(), [-5.0; 8]);
/// assert_eq!(r.imag.to_array(), [10.0; 8]);
/// ```
#[inline]
pub fn complex_mul_f32_portable(
    a: ComplexSimd<f32, 8>,
    b: ComplexSimd<f32, 8>,
) -> ComplexSimd<f32, 8> {
    let re = a.real.mul_add(b.real, -(a.imag * b.imag));
    let im = a.real.mul_add(b.imag, a.imag * b.real);
    ComplexSimd::new(re, im)
}

#[cfg(all(
    not(feature = "scalar-only"),
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline]
fn complex_mul_f32_avx2(a: ComplexSimd<f32, 8>, b: ComplexSimd<f32, 8>) -> ComplexSimd<f32, 8> {
    use core::arch::x86_64::*;
    use tpt_simd_core::Simd;
    let (ar, ai) = (a.real.to_array(), a.imag.to_array());
    let (br, bi) = (b.real.to_array(), b.imag.to_array());
    let mut re = [0.0f32; 8];
    let mut im = [0.0f32; 8];
    // SAFETY: this block is only compiled when `avx2` and `fma` are enabled
    // at compile time, so the intrinsics are available. All pointers come
    // from `[f32; 8]` arrays (32 bytes) and the loads/stores are unaligned.
    unsafe {
        let ar = _mm256_loadu_ps(ar.as_ptr());
        let ai = _mm256_loadu_ps(ai.as_ptr());
        let br = _mm256_loadu_ps(br.as_ptr());
        let bi = _mm256_loadu_ps(bi.as_ptr());
        // re = ar*br - (ai*bi) ; im = ar*bi + (ai*br)
        let r = _mm256_fmsub_ps(ar, br, _mm256_mul_ps(ai, bi));
        let i = _mm256_fmadd_ps(ar, bi, _mm256_mul_ps(ai, br));
        _mm256_storeu_ps(re.as_mut_ptr(), r);
        _mm256_storeu_ps(im.as_mut_ptr(), i);
    }
    ComplexSimd::new(Simd::from_array(re), Simd::from_array(im))
}

/// Complex multiply of 8 `f32` lanes in split layout.
///
/// Computes `re = fma(a.re, b.re, -(a.im*b.im))`,
/// `im = fma(a.re, b.im, a.im*b.re)` on every target (see the
/// [crate docs](crate)), so the result is bit-identical between the AVX2+FMA
/// path and [`complex_mul_f32_portable`]. NaN/inf propagate as in IEEE
/// arithmetic.
///
/// Performance: 2 `vmulps` + 2 fused ops on AVX2+FMA (no shuffles thanks to
/// the split layout); software `fmaf` elsewhere. NEON is not implemented
/// (portable path is used on aarch64).
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_mul::{complex_mul_f32, ComplexSimd};
/// use tpt_simd_core::F32x8;
/// let a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0));
/// let b = ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0));
/// let r = complex_mul_f32(a, b); // (1+2i)(3+4i) = -5 + 10i
/// assert_eq!(r.real.to_array(), [-5.0; 8]);
/// assert_eq!(r.imag.to_array(), [10.0; 8]);
/// ```
#[inline]
pub fn complex_mul_f32(a: ComplexSimd<f32, 8>, b: ComplexSimd<f32, 8>) -> ComplexSimd<f32, 8> {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        complex_mul_f32_avx2(a, b)
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        complex_mul_f32_portable(a, b)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use proptest::prelude::*;
    use tpt_simd_core::Simd;
    use tpt_simd_testutil::f32_with_specials;

    fn scalar(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
        [
            a[0].mul_add(b[0], -(a[1] * b[1])),
            a[0].mul_add(b[1], a[1] * b[0]),
        ]
    }

    fn cv(r: [f32; 8], i: [f32; 8]) -> ComplexSimd<f32, 8> {
        ComplexSimd::new(Simd::from_array(r), Simd::from_array(i))
    }

    fn same(x: f32, y: f32) -> bool {
        x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
    }

    #[test]
    fn specials() {
        let n = f32::NAN;
        let inf = f32::INFINITY;
        let a = cv(
            [n, inf, 0.0, f32::MAX, 1.0, -0.0, 2.0, 1e30],
            [0.0, 1.0, inf, f32::MAX, 1.0, 0.0, -3.0, 1e30],
        );
        let b = cv(
            [1.0, 0.0, 1.0, 2.0, inf, -0.0, 0.5, 1e30],
            [1.0, 1.0, 0.0, 2.0, 0.0, 1.0, 4.0, 1e30],
        );
        let r = complex_mul_f32(a, b);
        let p = complex_mul_f32_portable(a, b);
        for l in 0..8 {
            let e = scalar([a.real[l], a.imag[l]], [b.real[l], b.imag[l]]);
            assert!(same(r.real[l], e[0]) && same(r.imag[l], e[1]), "lane {l}");
            assert!(same(p.real[l], e[0]) && same(p.imag[l], e[1]), "lane {l}");
        }
    }

    proptest! {
        #[test]
        fn matches_scalar_and_portable(
            v in prop::collection::vec(f32_with_specials(), 32),
        ) {
            let g = |k: usize| -> [f32; 8] { v[k * 8..k * 8 + 8].try_into().unwrap() };
            let (ar, ai, br, bi) = (g(0), g(1), g(2), g(3));
            let (a, b) = (cv(ar, ai), cv(br, bi));
            let r = complex_mul_f32(a, b);
            let p = complex_mul_f32_portable(a, b);
            for l in 0..8 {
                let e = scalar([ar[l], ai[l]], [br[l], bi[l]]);
                prop_assert!(same(r.real[l], e[0]) && same(r.imag[l], e[1]));
                prop_assert!(same(p.real[l], e[0]) && same(p.imag[l], e[1]));
            }
        }
    }

    #[test]
    fn int_variants_match_scalar() {
        let xs = [i32::MIN, -1, 0, 1, 12345, i32::MAX, -65536, 65536];
        let a = Simd::<i32, 8>::from_array(xs);
        let b =
            Simd::<i32, 8>::from_array([7, i32::MIN, i32::MAX, -3, 99999, i32::MAX, 65536, 65536]);
        let (lo, hi) = (mul_lo_i32(a, b), mul_hi_i32(a, b));
        for l in 0..8 {
            let p = i64::from(a[l]) * i64::from(b[l]);
            assert_eq!(lo[l], p as i32);
            assert_eq!(hi[l], (p >> 32) as i32);
        }
        let x = Simd::<i16, 16>::from_fn(|i| [i16::MIN, i16::MAX, -1, 3][i % 4]);
        let y = Simd::<i16, 16>::from_fn(|i| [i16::MIN, i16::MIN, 5, -7][(i / 2) % 4]);
        let h = mul_hi_i16(x, y);
        for l in 0..16 {
            assert_eq!(h[l], ((i32::from(x[l]) * i32::from(y[l])) >> 16) as i16);
        }
        let w = mul_widen_i16(
            Simd::from_slice(&x.to_array()[..8]),
            Simd::from_slice(&y.to_array()[..8]),
        );
        assert_eq!(w[0], i32::from(i16::MIN) * i32::from(i16::MIN));
    }

    proptest! {
        #[test]
        fn mul_add_sub_matches_scalar(v in prop::collection::vec(f32_with_specials(), 24)) {
            let g = |k: usize| -> [f32; 8] { v[k * 8..k * 8 + 8].try_into().unwrap() };
            let r = mul_add_sub_f32(Simd::from_array(g(0)), Simd::from_array(g(1)), Simd::from_array(g(2)));
            let p = mul_add_sub_f32_portable(Simd::from_array(g(0)), Simd::from_array(g(1)), Simd::from_array(g(2)));
            for l in 0..8 {
                let c = if l % 2 == 0 { g(2)[l] } else { -g(2)[l] };
                prop_assert!(same(r[l], g(0)[l].mul_add(g(1)[l], c)));
                prop_assert!(same(p[l], g(0)[l].mul_add(g(1)[l], c)));
            }
        }
    }
}
