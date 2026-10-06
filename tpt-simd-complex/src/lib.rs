//! SIMD complex number operations for FFT/MDCT.
//!
//! Builds on [`ComplexSimd`] (split real/imaginary layout, defined in
//! `tpt-simd-core`) and adds an extension trait, [`ComplexSimdF32Ext`], with
//! fused multiply-add forms, twiddle multiplication, conjugate multiply,
//! polar construction, multiply-by-`±i`, an FFT butterfly and AoS
//! (interleaved) <-> SoA (split) conversion, plus slice helpers
//! [`interleaved_to_split`] and [`split_to_interleaved`].
//!
//! ## Rounding
//!
//! The operators on [`ComplexSimd`] (`+ - * /`, `mul`, `mag_sq`) are unfused
//! and identical everywhere. Every *fused* method of this crate
//! (`twiddle_mul_fma`, `fmadd`, `fmsub`, `mul_conj`, `norm_sq`, `fft_butterfly`)
//! uses `fma` rounding on all targets (hardware FMA when compiled in,
//! correctly rounded software `fmaf` otherwise), so results are bit-identical
//! across targets; see `tpt-simd-mul` for the rule on `complex_mul_f32`.
//!
//! ## Example (spec Appendix B.1)
//!
//! ```
//! use tpt_simd_complex::ComplexSimd;
//! use tpt_simd_core::F32x8;
//! let a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0));
//! let b = ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0));
//! let result = a * b; // (1+2i)(3+4i) = (3-8) + (4+6)i = -5 + 10i
//! assert_eq!(result.real.to_array(), [-5.0; 8]);
//! assert_eq!(result.imag.to_array(), [10.0; 8]);
//! ```
#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

pub use tpt_simd_core::{ComplexSimd, F32x8, Simd};
pub use tpt_simd_mul::complex_mul_f32;

/// Fused complex multiply for any lane count; `N == 8` uses
/// [`complex_mul_f32`] (hardware FMA path), other widths the identical
/// formula via `Simd::mul_add`.
#[inline]
fn cmul<const N: usize>(a: ComplexSimd<f32, N>, b: ComplexSimd<f32, N>) -> ComplexSimd<f32, N> {
    if N == 8 {
        let conv = |c: ComplexSimd<f32, N>| {
            let mut r = [0.0f32; 8];
            let mut i = [0.0f32; 8];
            r.copy_from_slice(&c.real.to_array()[..8]);
            i.copy_from_slice(&c.imag.to_array()[..8]);
            ComplexSimd::new(Simd::from_array(r), Simd::from_array(i))
        };
        let r = complex_mul_f32(conv(a), conv(b));
        let mut re = [0.0f32; N];
        let mut im = [0.0f32; N];
        re.copy_from_slice(&r.real.to_array()[..N]);
        im.copy_from_slice(&r.imag.to_array()[..N]);
        ComplexSimd::new(Simd::from_array(re), Simd::from_array(im))
    } else {
        ComplexSimd::new(
            a.real.mul_add(b.real, -(a.imag * b.imag)),
            a.real.mul_add(b.imag, a.imag * b.real),
        )
    }
}

/// Extension methods for `f32` complex vectors in split layout.
///
/// Implemented for every `ComplexSimd<f32, N>`; `N = 8` ([`F32x8`]) takes the
/// AVX2+FMA path for multiplies.
pub trait ComplexSimdF32Ext<const N: usize>: Sized {
    /// `self * b + c` with fused rounding.
    ///
    /// Performance: 4 fused ops (hardware FMA when available).
    ///
    /// # Panics
    /// Never.
    ///
    /// ```
    /// use tpt_simd_complex::{ComplexSimd, ComplexSimdF32Ext};
    /// let a = ComplexSimd::<f32, 4>::splat(1.0, 2.0);
    /// let r = a.fmadd(ComplexSimd::splat(3.0, 4.0), ComplexSimd::splat(1.0, 1.0));
    /// assert_eq!(r.real.to_array(), [-4.0; 4]);
    /// assert_eq!(r.imag.to_array(), [11.0; 4]);
    /// ```
    fn fmadd(self, b: Self, c: Self) -> Self;

    /// `self * b - c` with fused rounding.
    ///
    /// # Panics
    /// Never.
    ///
    /// ```
    /// use tpt_simd_complex::{ComplexSimd, ComplexSimdF32Ext};
    /// let a = ComplexSimd::<f32, 4>::splat(1.0, 2.0);
    /// let r = a.fmsub(ComplexSimd::splat(3.0, 4.0), ComplexSimd::splat(1.0, 1.0));
    /// assert_eq!(r.real.to_array(), [-6.0; 4]);
    /// assert_eq!(r.imag.to_array(), [9.0; 4]);
    /// ```
    fn fmsub(self, b: Self, c: Self) -> Self;

    /// Multiply by an FFT twiddle factor (fused; same bits as
    /// [`complex_mul_f32`]). Named `_fma` because the inherent, unfused
    /// `ComplexSimd::twiddle_mul` in core would shadow a trait method of the
    /// same name.
    ///
    /// # Panics
    /// Never.
    ///
    /// ```
    /// use tpt_simd_complex::{ComplexSimd, ComplexSimdF32Ext};
    /// let a = ComplexSimd::<f32, 8>::splat(1.0, 2.0);
    /// let r = a.twiddle_mul_fma(ComplexSimd::splat(3.0, 4.0));
    /// assert_eq!(r.real.to_array(), [-5.0; 8]);
    /// ```
    fn twiddle_mul_fma(self, w: Self) -> Self;

    /// `self * conj(b)` (fused).
    ///
    /// # Panics
    /// Never.
    ///
    /// ```
    /// use tpt_simd_complex::{ComplexSimd, ComplexSimdF32Ext};
    /// let a = ComplexSimd::<f32, 4>::splat(1.0, 2.0);
    /// let r = a.mul_conj(ComplexSimd::splat(3.0, 4.0)); // (1+2i)(3-4i) = 11 + 2i
    /// assert_eq!(r.real.to_array(), [11.0; 4]);
    /// assert_eq!(r.imag.to_array(), [2.0; 4]);
    /// ```
    fn mul_conj(self, b: Self) -> Self;

    /// `re² + im²` with one fused operation (`fma(re, re, im*im)`).
    ///
    /// # Panics
    /// Never.
    ///
    /// ```
    /// use tpt_simd_complex::{ComplexSimd, ComplexSimdF32Ext};
    /// let a = ComplexSimd::<f32, 4>::splat(3.0, 4.0);
    /// assert_eq!(a.norm_sq().to_array(), [25.0; 4]);
    /// ```
    fn norm_sq(self) -> Simd<f32, N>;

    /// Build `mag * (cos(phase) + i sin(phase))` per lane (via `libm`).
    ///
    /// Performance: scalar `libm` calls per lane; not a hot-path operation.
    ///
    /// # Panics
    /// Never.
    ///
    /// ```
    /// use tpt_simd_complex::{ComplexSimd, ComplexSimdF32Ext};
    /// use tpt_simd_core::F32x4;
    /// let z = ComplexSimd::<f32, 4>::from_polar(F32x4::splat(2.0), F32x4::splat(0.0));
    /// assert_eq!(z.real.to_array(), [2.0; 4]);
    /// assert_eq!(z.imag.to_array(), [0.0; 4]);
    /// ```
    fn from_polar(mag: Simd<f32, N>, phase: Simd<f32, N>) -> Self;

    /// Multiply by `i`: `(a + bi)·i = -b + ai` (no multiplies, exact).
    ///
    /// # Panics
    /// Never.
    ///
    /// ```
    /// use tpt_simd_complex::{ComplexSimd, ComplexSimdF32Ext};
    /// let r = ComplexSimd::<f32, 4>::splat(1.0, 2.0).i_mul();
    /// assert_eq!((r.real[0], r.imag[0]), (-2.0, 1.0));
    /// ```
    fn i_mul(self) -> Self;

    /// Multiply by `-i`: `(a + bi)·(-i) = b - ai` (no multiplies, exact).
    ///
    /// # Panics
    /// Never.
    ///
    /// ```
    /// use tpt_simd_complex::{ComplexSimd, ComplexSimdF32Ext};
    /// let r = ComplexSimd::<f32, 4>::splat(1.0, 2.0).neg_i_mul();
    /// assert_eq!((r.real[0], r.imag[0]), (2.0, -1.0));
    /// ```
    fn neg_i_mul(self) -> Self;

    /// Multiply every lane by the scalar `s`.
    ///
    /// # Panics
    /// Never.
    ///
    /// ```
    /// use tpt_simd_complex::{ComplexSimd, ComplexSimdF32Ext};
    /// let r = ComplexSimd::<f32, 4>::splat(1.0, 2.0).scale_scalar(0.5);
    /// assert_eq!((r.real[0], r.imag[0]), (0.5, 1.0));
    /// ```
    fn scale_scalar(self, s: f32) -> Self;

    /// Radix-2 decimation-in-time butterfly with twiddle:
    /// `(a, b) <- (a + b·w, a - b·w)` (`b·w` is the fused twiddle multiply).
    ///
    /// # Panics
    /// Never.
    ///
    /// ```
    /// use tpt_simd_complex::{ComplexSimd, ComplexSimdF32Ext};
    /// let mut a = ComplexSimd::<f32, 8>::splat(1.0, 0.0);
    /// let mut b = ComplexSimd::<f32, 8>::splat(0.0, 1.0);
    /// ComplexSimd::fft_butterfly(&mut a, &mut b, ComplexSimd::splat(0.0, 1.0)); // w = i
    /// // b*w = -1 -> a = 0, b = 2
    /// assert_eq!((a.real[0], a.imag[0]), (0.0, 0.0));
    /// assert_eq!((b.real[0], b.imag[0]), (2.0, 0.0));
    /// ```
    fn fft_butterfly(a: &mut Self, b: &mut Self, w: Self);

    /// Load `N` complex numbers from AoS `[re0, im0, re1, im1, ...]`.
    ///
    /// # Panics
    /// If `src.len() != 2 * N`.
    ///
    /// ```
    /// use tpt_simd_complex::{ComplexSimd, ComplexSimdF32Ext};
    /// let z = ComplexSimd::<f32, 2>::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(z.real.to_array(), [1.0, 3.0]);
    /// assert_eq!(z.imag.to_array(), [2.0, 4.0]);
    /// ```
    fn from_interleaved(src: &[f32]) -> Self;

    /// Store as AoS `[re0, im0, re1, im1, ...]`.
    ///
    /// # Panics
    /// If `dst.len() != 2 * N`.
    ///
    /// ```
    /// use tpt_simd_complex::{ComplexSimd, ComplexSimdF32Ext};
    /// let z = ComplexSimd::<f32, 2>::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let mut out = [0.0; 4];
    /// z.to_interleaved(&mut out);
    /// assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    /// ```
    fn to_interleaved(self, dst: &mut [f32]);
}

impl<const N: usize> ComplexSimdF32Ext<N> for ComplexSimd<f32, N> {
    #[inline]
    fn fmadd(self, b: Self, c: Self) -> Self {
        let re = self
            .real
            .mul_add(b.real, (-self.imag).mul_add(b.imag, c.real));
        let im = self.real.mul_add(b.imag, self.imag.mul_add(b.real, c.imag));
        ComplexSimd::new(re, im)
    }

    #[inline]
    fn fmsub(self, b: Self, c: Self) -> Self {
        let re = self
            .real
            .mul_add(b.real, (-self.imag).mul_add(b.imag, -c.real));
        let im = self
            .real
            .mul_add(b.imag, self.imag.mul_add(b.real, -c.imag));
        ComplexSimd::new(re, im)
    }

    #[inline]
    fn twiddle_mul_fma(self, w: Self) -> Self {
        cmul(self, w)
    }

    #[inline]
    fn mul_conj(self, b: Self) -> Self {
        cmul(self, b.conj())
    }

    #[inline]
    fn norm_sq(self) -> Simd<f32, N> {
        self.real.mul_add(self.real, self.imag * self.imag)
    }

    fn from_polar(mag: Simd<f32, N>, phase: Simd<f32, N>) -> Self {
        let (mut re, mut im) = ([0.0f32; N], [0.0f32; N]);
        for l in 0..N {
            let (s, c) = libm::sincosf(phase[l]);
            re[l] = mag[l] * c;
            im[l] = mag[l] * s;
        }
        ComplexSimd::new(Simd::from_array(re), Simd::from_array(im))
    }

    #[inline]
    fn i_mul(self) -> Self {
        ComplexSimd::new(-self.imag, self.real)
    }

    #[inline]
    fn neg_i_mul(self) -> Self {
        ComplexSimd::new(self.imag, -self.real)
    }

    #[inline]
    fn scale_scalar(self, s: f32) -> Self {
        self.scale(Simd::splat(s))
    }

    #[inline]
    fn fft_butterfly(a: &mut Self, b: &mut Self, w: Self) {
        let t = cmul(*b, w);
        let (s, d) = (a.add(t), a.sub(t));
        *a = s;
        *b = d;
    }

    #[inline]
    fn from_interleaved(src: &[f32]) -> Self {
        assert!(
            src.len() == 2 * N,
            "from_interleaved: expected {} floats, got {}",
            2 * N,
            src.len()
        );
        ComplexSimd::new(
            Simd::from_fn(|l| src[2 * l]),
            Simd::from_fn(|l| src[2 * l + 1]),
        )
    }

    #[inline]
    fn to_interleaved(self, dst: &mut [f32]) {
        assert!(
            dst.len() == 2 * N,
            "to_interleaved: expected {} floats, got {}",
            2 * N,
            dst.len()
        );
        for l in 0..N {
            dst[2 * l] = self.real[l];
            dst[2 * l + 1] = self.imag[l];
        }
    }
}

/// Deinterleave AoS `[re0, im0, re1, im1, ...]` into separate `re`/`im`
/// slices. Any length (including empty) is valid.
///
/// Performance: simple strided loop that LLVM vectorises with shuffles.
///
/// # Panics
/// If `re.len() != im.len()` or `src.len() != 2 * re.len()`.
///
/// ```
/// let (mut re, mut im) = ([0.0; 3], [0.0; 3]);
/// tpt_simd_complex::interleaved_to_split(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &mut re, &mut im);
/// assert_eq!(re, [1.0, 3.0, 5.0]);
/// assert_eq!(im, [2.0, 4.0, 6.0]);
/// ```
pub fn interleaved_to_split(src: &[f32], re: &mut [f32], im: &mut [f32]) {
    assert!(
        re.len() == im.len() && src.len() == 2 * re.len(),
        "interleaved_to_split: length mismatch (src {}, re {}, im {})",
        src.len(),
        re.len(),
        im.len()
    );
    for ((p, r), i) in src.chunks_exact(2).zip(re.iter_mut()).zip(im.iter_mut()) {
        *r = p[0];
        *i = p[1];
    }
}

/// Interleave separate `re`/`im` slices into AoS
/// `[re0, im0, re1, im1, ...]`. Any length (including empty) is valid.
///
/// # Panics
/// If `re.len() != im.len()` or `dst.len() != 2 * re.len()`.
///
/// ```
/// let mut out = [0.0; 6];
/// tpt_simd_complex::split_to_interleaved(&[1.0, 3.0, 5.0], &[2.0, 4.0, 6.0], &mut out);
/// assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
pub fn split_to_interleaved(re: &[f32], im: &[f32], dst: &mut [f32]) {
    assert!(
        re.len() == im.len() && dst.len() == 2 * re.len(),
        "split_to_interleaved: length mismatch (re {}, im {}, dst {})",
        re.len(),
        im.len(),
        dst.len()
    );
    for ((p, r), i) in dst.chunks_exact_mut(2).zip(re).zip(im) {
        p[0] = *r;
        p[1] = *i;
    }
}

/// Fused complex multiply/add helpers for `f64` lanes (portable).
///
/// Same rounding rule as the `f32` extension: fused everywhere.
pub trait ComplexSimdF64Ext<const N: usize>: Sized {
    /// `self * b` with fused rounding.
    ///
    /// # Panics
    /// Never.
    ///
    /// ```
    /// use tpt_simd_complex::{ComplexSimd, ComplexSimdF64Ext};
    /// let r = ComplexSimd::<f64, 2>::splat(1.0, 2.0).twiddle_mul_fma(ComplexSimd::splat(3.0, 4.0));
    /// assert_eq!((r.real[0], r.imag[0]), (-5.0, 10.0));
    /// ```
    fn twiddle_mul_fma(self, w: Self) -> Self;
    /// `self * conj(b)`.
    ///
    /// # Panics
    /// Never.
    ///
    /// ```
    /// use tpt_simd_complex::{ComplexSimd, ComplexSimdF64Ext};
    /// let r = ComplexSimd::<f64, 2>::splat(1.0, 2.0).mul_conj(ComplexSimd::splat(3.0, 4.0));
    /// assert_eq!((r.real[0], r.imag[0]), (11.0, 2.0));
    /// ```
    fn mul_conj(self, b: Self) -> Self;
    /// `re² + im²`.
    ///
    /// # Panics
    /// Never.
    ///
    /// ```
    /// use tpt_simd_complex::{ComplexSimd, ComplexSimdF64Ext};
    /// assert_eq!(ComplexSimd::<f64, 2>::splat(3.0, 4.0).norm_sq()[0], 25.0);
    /// ```
    fn norm_sq(self) -> Simd<f64, N>;
    /// Multiply by `i`.
    ///
    /// # Panics
    /// Never.
    ///
    /// ```
    /// use tpt_simd_complex::{ComplexSimd, ComplexSimdF64Ext};
    /// let r = ComplexSimd::<f64, 2>::splat(1.0, 2.0).i_mul();
    /// assert_eq!((r.real[0], r.imag[0]), (-2.0, 1.0));
    /// ```
    fn i_mul(self) -> Self;
    /// Multiply by `-i`.
    ///
    /// # Panics
    /// Never.
    ///
    /// ```
    /// use tpt_simd_complex::{ComplexSimd, ComplexSimdF64Ext};
    /// let r = ComplexSimd::<f64, 2>::splat(1.0, 2.0).neg_i_mul();
    /// assert_eq!((r.real[0], r.imag[0]), (2.0, -1.0));
    /// ```
    fn neg_i_mul(self) -> Self;
}

impl<const N: usize> ComplexSimdF64Ext<N> for ComplexSimd<f64, N> {
    #[inline]
    fn twiddle_mul_fma(self, w: Self) -> Self {
        ComplexSimd::new(
            self.real.mul_add(w.real, -(self.imag * w.imag)),
            self.real.mul_add(w.imag, self.imag * w.real),
        )
    }
    #[inline]
    fn mul_conj(self, b: Self) -> Self {
        ComplexSimdF64Ext::twiddle_mul_fma(self, b.conj())
    }
    #[inline]
    fn norm_sq(self) -> Simd<f64, N> {
        self.real.mul_add(self.real, self.imag * self.imag)
    }
    #[inline]
    fn i_mul(self) -> Self {
        ComplexSimd::new(-self.imag, self.real)
    }
    #[inline]
    fn neg_i_mul(self) -> Self {
        ComplexSimd::new(self.imag, -self.real)
    }
}

#[cfg(test)]
mod tests;
