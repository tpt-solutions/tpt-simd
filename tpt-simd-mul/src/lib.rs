//! SIMD multiply variants (high/low/widening/complex).
//!
//! Currently implemented: [`complex_mul_f32`] (plus its portable reference
//! [`complex_mul_f32_portable`]). The integer multiply variants are a later
//! phase.
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
}
