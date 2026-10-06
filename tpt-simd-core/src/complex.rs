//! Split (structure-of-arrays) complex vectors.

use core::ops::{Add, Div, Mul, Neg, Sub};
use tpt_simd_vector::{Simd, SimdElement, SimdFloat};

/// `N` complex numbers stored as a vector of real parts and a vector of
/// imaginary parts.
///
/// The split layout means complex multiply is 4 vector multiplies and 2
/// add/subs with no shuffles. The generic methods here use plain (unfused)
/// arithmetic so results are identical on every target; see
/// `tpt-simd-mul`/`tpt-simd-complex` for FMA-accelerated `f32` versions.
///
/// ```
/// use tpt_simd_core::{ComplexSimd, F32x8};
/// let a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0));
/// let b = ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0));
/// let r = a * b; // (1+2i)(3+4i) = -5 + 10i
/// assert_eq!(r.real.to_array(), [-5.0; 8]);
/// assert_eq!(r.imag.to_array(), [10.0; 8]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplexSimd<T: SimdElement, const N: usize> {
    /// Real parts.
    pub real: Simd<T, N>,
    /// Imaginary parts.
    pub imag: Simd<T, N>,
}

impl<T: SimdElement, const N: usize> ComplexSimd<T, N> {
    /// Build from real and imaginary vectors.
    #[inline(always)]
    pub const fn new(real: Simd<T, N>, imag: Simd<T, N>) -> Self {
        Self { real, imag }
    }

    /// All lanes `re + im·i`.
    #[inline(always)]
    pub fn splat(re: T, im: T) -> Self {
        Self::new(Simd::splat(re), Simd::splat(im))
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero() -> Self {
        Self::new(Simd::zero(), Simd::zero())
    }

    /// Lane-wise sum.
    #[inline(always)]
    pub fn add(self, o: Self) -> Self {
        Self::new(self.real + o.real, self.imag + o.imag)
    }

    /// Lane-wise difference.
    #[inline(always)]
    pub fn sub(self, o: Self) -> Self {
        Self::new(self.real - o.real, self.imag - o.imag)
    }

    /// Lane-wise negation.
    #[inline(always)]
    pub fn neg(self) -> Self {
        Self::new(-self.real, -self.imag)
    }

    /// Complex conjugate.
    #[inline(always)]
    pub fn conj(self) -> Self {
        Self::new(self.real, -self.imag)
    }

    /// Lane-wise product: `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`.
    #[inline(always)]
    pub fn mul(self, o: Self) -> Self {
        Self::new(
            self.real * o.real - self.imag * o.imag,
            self.real * o.imag + self.imag * o.real,
        )
    }

    /// Multiply every lane by a real vector.
    #[inline(always)]
    pub fn scale(self, s: Simd<T, N>) -> Self {
        Self::new(self.real * s, self.imag * s)
    }

    /// `|z|² = re² + im²`.
    #[inline(always)]
    pub fn mag_sq(self) -> Simd<T, N> {
        self.real * self.real + self.imag * self.imag
    }

    /// Lane-wise quotient `self / o`. Division by zero follows the element
    /// type (float: inf/NaN). Intended for float lanes.
    #[inline(always)]
    pub fn div(self, o: Self) -> Self {
        let d = o.mag_sq();
        Self::new(
            (self.real * o.real + self.imag * o.imag) / d,
            (self.imag * o.real - self.real * o.imag) / d,
        )
    }

    /// Multiply by an FFT twiddle factor (alias of [`ComplexSimd::mul`]).
    #[inline(always)]
    pub fn twiddle_mul(self, twiddle: Self) -> Self {
        self.mul(twiddle)
    }

    /// In-place radix-2 butterfly: `(a, b) <- (a + b, a - b)`.
    #[inline(always)]
    pub fn butterfly(&mut self, other: &mut Self) {
        let (s, d) = (self.add(*other), self.sub(*other));
        *self = s;
        *other = d;
    }
}

impl<T: SimdFloat, const N: usize> ComplexSimd<T, N> {
    /// `|z|`.
    #[inline(always)]
    pub fn mag(self) -> Simd<T, N> {
        self.mag_sq().sqrt()
    }

    /// `atan2(im, re)` in radians.
    #[inline(always)]
    pub fn phase(self) -> Simd<T, N> {
        self.imag.atan2(self.real)
    }
}

macro_rules! cop {
    ($tr:ident, $m:ident) => {
        impl<T: SimdElement, const N: usize> $tr for ComplexSimd<T, N> {
            type Output = Self;
            #[inline(always)]
            fn $m(self, rhs: Self) -> Self {
                ComplexSimd::$m(self, rhs)
            }
        }
    };
}
cop!(Add, add);
cop!(Sub, sub);
cop!(Mul, mul);
cop!(Div, div);

impl<T: SimdElement, const N: usize> Neg for ComplexSimd<T, N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        ComplexSimd::neg(self)
    }
}
