//! The [`Fixed`] vector type.

use crate::repr::{
    FixedRepr, MUL_ROUND_SAT, MUL_ROUND_WRAP, MUL_TRUNC_WRAP, max_w, min_w, shr_round_away,
};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use tpt_simd_core::{Simd, SimdFloat};

/// `N` fixed-point numbers with `INT_BITS` integer bits (including the sign
/// bit) and `FRAC_BITS` fractional bits, stored as raw `T` lanes.
///
/// A lane with raw value `r` represents `r / 2^FRAC_BITS`. See the
/// [crate docs](crate) for the overflow and rounding policy.
///
/// The format must satisfy `INT_BITS >= 1` and
/// `INT_BITS + FRAC_BITS <= T::BITS`, otherwise instantiating any
/// constructor or operation fails to compile.
///
/// # Examples
///
/// ```
/// use tpt_simd_core::I32x4;
/// use tpt_simd_fixed::Fixed;
///
/// type Q = Fixed<i32, 16, 16, 4>;
/// let a = Q::from_raw(I32x4::splat(3 << 16)); // 3.0
/// assert_eq!((a + a).raw(), I32x4::splat(6 << 16));
/// ```
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Fixed<T: FixedRepr, const INT_BITS: u32, const FRAC_BITS: u32, const N: usize>(
    Simd<T, N>,
);

impl<T: FixedRepr, const I: u32, const F: u32, const N: usize> Fixed<T, I, F, N> {
    /// Compile-time format validation (evaluated when the const is used).
    #[inline(always)]
    const fn check() {
        const {
            assert!(I >= 1, "INT_BITS must be >= 1 (it counts the sign bit)");
            assert!(
                I + F <= T::BITS,
                "INT_BITS + FRAC_BITS must be <= the storage type's bit width"
            );
        }
    }

    /// Wrap a raw vector without any checks (other than the format check).
    #[inline(always)]
    fn new(v: Simd<T, N>) -> Self {
        Self::check();
        Self(v)
    }

    #[inline(always)]
    fn lo() -> T {
        T::narrow(min_w(I, F))
    }

    #[inline(always)]
    fn hi() -> T {
        T::narrow(max_w(I, F))
    }

    /// Clamp raw lanes into the format range (no-op work for full-width formats).
    #[inline(always)]
    fn clamp_raw(v: Simd<T, N>) -> Simd<T, N> {
        if I + F < T::BITS {
            v.clamp(Simd::splat(Self::lo()), Simd::splat(Self::hi()))
        } else {
            v
        }
    }

    /// Build from raw integer lanes (`value = raw / 2^FRAC_BITS`).
    ///
    /// The raw values are not range-checked against the format; for narrow
    /// formats use [`clamp_to_format`](Self::clamp_to_format) if needed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_core::I32x8;
    /// use tpt_simd_fixed::Fixed;
    /// let x: Fixed<i32, 16, 16, 8> = Fixed::from_raw(I32x8::splat(1 << 15));
    /// assert_eq!(x.to_f32().to_array(), [0.5; 8]);
    /// ```
    #[inline(always)]
    pub fn from_raw(raw: Simd<T, N>) -> Self {
        Self::new(raw)
    }

    /// The raw integer lanes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_core::F32x4;
    /// use tpt_simd_fixed::Fixed;
    /// let x: Fixed<i32, 8, 24, 4> = Fixed::from_f32(F32x4::splat(1.0));
    /// assert_eq!(x.raw().to_array(), [1 << 24; 4]);
    /// ```
    #[inline(always)]
    pub fn raw(self) -> Simd<T, N> {
        self.0
    }

    /// Smallest representable value, in every lane.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// assert_eq!(Fixed::<i32, 16, 16, 2>::min_value().raw().to_array(), [i32::MIN; 2]);
    /// assert_eq!(Fixed::<i32, 4, 12, 2>::min_value().raw().to_array(), [-(1 << 15); 2]);
    /// ```
    #[inline(always)]
    pub fn min_value() -> Self {
        Self::new(Simd::splat(Self::lo()))
    }

    /// Largest representable value, in every lane.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// assert_eq!(Fixed::<i32, 16, 16, 2>::max_value().raw().to_array(), [i32::MAX; 2]);
    /// ```
    #[inline(always)]
    pub fn max_value() -> Self {
        Self::new(Simd::splat(Self::hi()))
    }

    /// The value `1.0` in every lane.
    ///
    /// Compile error unless `INT_BITS >= 2` (1.0 is not representable in a
    /// format with only a sign bit).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// assert_eq!(Fixed::<i32, 16, 16, 2>::one().raw().to_array(), [1 << 16; 2]);
    /// ```
    #[inline(always)]
    pub fn one() -> Self {
        const {
            assert!(I >= 2, "1.0 needs INT_BITS >= 2");
        }
        Self::new(Simd::splat(T::narrow(1i64 << F)))
    }

    /// The smallest positive increment (one least-significant bit, `2^-FRAC_BITS`).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// assert_eq!(Fixed::<i32, 16, 16, 2>::epsilon().raw().to_array(), [1; 2]);
    /// ```
    #[inline(always)]
    pub fn epsilon() -> Self {
        Self::new(Simd::splat(T::narrow(1)))
    }

    /// Broadcast one raw value to all lanes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// let x = Fixed::<i32, 16, 16, 4>::splat_raw(1 << 16);
    /// assert_eq!(x, Fixed::one());
    /// ```
    #[inline(always)]
    pub fn splat_raw(raw: T) -> Self {
        Self::new(Simd::splat(raw))
    }

    /// Broadcast one `f32` to all lanes (see [`from_f32`](Self::from_f32)).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// let x = Fixed::<i32, 16, 16, 4>::splat_f32(0.25);
    /// assert_eq!(x.raw().to_array(), [1 << 14; 4]);
    /// ```
    #[inline(always)]
    pub fn splat_f32(x: f32) -> Self {
        Self::from_f32(Simd::splat(x))
    }

    /// Convert from `f32` lanes: scale by `2^FRAC_BITS`, round to nearest
    /// (**ties to even**), **saturate** to the format range. NaN becomes 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_core::F32x4;
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 4>;
    /// let x = Q::from_f32(F32x4::from_array([1.5, 1e9, -1e9, f32::NAN]));
    /// assert_eq!(x.raw().to_array(), [3 << 15, i32::MAX, i32::MIN, 0]);
    /// // ties to even: 0.5 LSB rounds to 0, 1.5 LSB rounds to 2
    /// let eps = 2f32.powi(-16);
    /// let y = Q::from_f32(F32x4::from_array([0.5 * eps, 1.5 * eps, 2.5 * eps, -0.5 * eps]));
    /// assert_eq!(y.raw().to_array(), [0, 2, 2, 0]);
    /// ```
    #[inline]
    pub fn from_f32(x: Simd<f32, N>) -> Self {
        Self::check();
        let scale = (1u64 << F) as f32; // exact power of two
        Self::new(Self::clamp_raw(x.map(|v| {
            let s = v * scale;
            // Fast ties-to-even for |s| < 2^22 via the 1.5 * 2^23 trick
            // (exact in f32 round-to-nearest-even); rare large/NaN values
            // take the libm path.
            let r = if s.abs() < 4_194_304.0 {
                (s + 12_582_912.0) - 12_582_912.0
            } else {
                s.lane_round_ties_even()
            };
            T::from_f32_sat(r)
        })))
    }

    /// Convert to `f32` lanes (`raw * 2^-FRAC_BITS`). Exact when the raw
    /// value fits in 24 bits, otherwise rounded to nearest by the `f32`
    /// conversion.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_core::I32x4;
    /// use tpt_simd_fixed::Fixed;
    /// let x = Fixed::<i32, 16, 16, 4>::from_raw(I32x4::splat(-(5 << 14)));
    /// assert_eq!(x.to_f32().to_array(), [-1.25; 4]);
    /// ```
    #[inline]
    pub fn to_f32(self) -> Simd<f32, N> {
        Self::check();
        let inv = f32::from_bits((127 - F) << 23); // 2^-F, exact for F <= 126
        self.0.map(|v| v.to_f32() * inv)
    }

    /// Clamp every lane into the format range `[min_value, max_value]`.
    /// A no-op when `INT_BITS + FRAC_BITS == T::BITS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_core::Simd;
    /// use tpt_simd_fixed::Fixed;
    /// let x = Fixed::<i32, 4, 12, 2>::from_raw(Simd::from_array([1 << 20, -(1 << 20)]));
    /// assert_eq!(x.clamp_to_format().raw().to_array(), [(1 << 15) - 1, -(1 << 15)]);
    /// ```
    #[inline(always)]
    pub fn clamp_to_format(self) -> Self {
        Self::new(Self::clamp_raw(self.0))
    }

    /// Lane addition, wrapping at the storage width on overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 2>;
    /// assert_eq!(Q::max_value().wrapping_add(Q::epsilon()), Q::min_value());
    /// ```
    #[inline(always)]
    pub fn wrapping_add(self, rhs: Self) -> Self {
        Self::new(self.0 + rhs.0)
    }

    /// Lane subtraction, wrapping at the storage width on overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 2>;
    /// assert_eq!(Q::min_value().wrapping_sub(Q::epsilon()), Q::max_value());
    /// ```
    #[inline(always)]
    pub fn wrapping_sub(self, rhs: Self) -> Self {
        Self::new(self.0 - rhs.0)
    }

    /// Lane addition, saturating to the format range.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 2>;
    /// assert_eq!(Q::max_value().saturating_add(Q::one()), Q::max_value());
    /// ```
    #[inline(always)]
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self::new(Self::clamp_raw(self.0.sat_add(rhs.0)))
    }

    /// Lane subtraction, saturating to the format range.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 2>;
    /// assert_eq!(Q::min_value().saturating_sub(Q::one()), Q::min_value());
    /// ```
    #[inline(always)]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self::new(Self::clamp_raw(self.0.sat_sub(rhs.0)))
    }

    /// Lane multiply with a 64-bit intermediate, rounded to nearest
    /// (**ties away from zero**), **wrapping** at the storage width.
    ///
    /// This is what the `*` operator does.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 8>;
    /// let h = Q::splat_raw(1); // 2^-16
    /// // 0.5 LSB: ties away from zero
    /// assert_eq!(h.wrapping_mul(Q::splat_raw(1 << 15)), Q::splat_raw(1));
    /// assert_eq!((-h).wrapping_mul(Q::splat_raw(1 << 15)), Q::splat_raw(-1));
    /// ```
    #[inline(always)]
    pub fn wrapping_mul(self, rhs: Self) -> Self {
        Self::check();
        Self::new(Simd::from_array(T::mul_lanes::<I, F, MUL_ROUND_WRAP, N>(
            self.0.as_array(),
            rhs.0.as_array(),
        )))
    }

    /// Lane multiply with a 64-bit intermediate, rounded to nearest
    /// (**ties away from zero**), **saturating** to the format range.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 8>;
    /// let big = Q::from_raw(tpt_simd_core::I32x8::splat(1 << 30)); // 16384.0
    /// assert_eq!(big.saturating_mul(big), Q::max_value());
    /// assert_eq!(big.saturating_mul(-big), Q::min_value());
    /// ```
    #[inline(always)]
    pub fn saturating_mul(self, rhs: Self) -> Self {
        Self::check();
        Self::new(Simd::from_array(T::mul_lanes::<I, F, MUL_ROUND_SAT, N>(
            self.0.as_array(),
            rhs.0.as_array(),
        )))
    }

    /// Lane multiply that **truncates** the 64-bit product (arithmetic shift
    /// right by `FRAC_BITS`, i.e. rounds toward negative infinity) and wraps
    /// at the storage width. Slightly cheaper than the rounding variants.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 8>;
    /// let h = Q::splat_raw(1 << 15); // 0.5
    /// assert_eq!(Q::splat_raw(3).mul_trunc(h), Q::splat_raw(1)); // 1.5 LSB -> 1
    /// assert_eq!(Q::splat_raw(-3).mul_trunc(h), Q::splat_raw(-2)); // -1.5 LSB -> -2
    /// ```
    #[inline(always)]
    pub fn mul_trunc(self, rhs: Self) -> Self {
        Self::check();
        Self::new(Simd::from_array(T::mul_lanes::<I, F, MUL_TRUNC_WRAP, N>(
            self.0.as_array(),
            rhs.0.as_array(),
        )))
    }

    /// Shared division kernel: `(a << F) / b` rounded to nearest, ties away
    /// from zero, in 64-bit. `b` must be non-zero.
    #[inline(always)]
    fn div_lane(a: i64, b: i64) -> i64 {
        let n = a << F;
        let q = n / b;
        let r = n % b;
        if 2 * r.abs() >= b.abs() {
            if (n < 0) != (b < 0) { q - 1 } else { q + 1 }
        } else {
            q
        }
    }

    /// Lane division, `(a << FRAC_BITS) / b` with a 64-bit intermediate,
    /// rounded to nearest (**ties away from zero**), **wrapping** on overflow.
    /// This is what the `/` operator does.
    ///
    /// # Panics
    ///
    /// Panics if any lane of `rhs` is zero. See [`checked_div`](Self::checked_div)
    /// and [`saturating_div`](Self::saturating_div).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 4>;
    /// let r = Q::splat_f32(1.0).div(Q::splat_f32(3.0));
    /// assert_eq!(r.raw().to_array(), [21845; 4]); // 0.33333 rounded
    /// ```
    #[inline]
    pub fn div(self, rhs: Self) -> Self {
        Self::check();
        assert!(
            rhs.0.as_array().iter().all(|&b| b != T::ZERO),
            "Fixed division by zero"
        );
        Self::new(self.0.zip_with(rhs.0, |a, b| {
            T::narrow(Self::div_lane(a.widen(), b.widen()))
        }))
    }

    /// Like [`div`](Self::div), but returns `None` if any lane of `rhs` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 2>;
    /// assert!(Q::one().checked_div(Q::splat_raw(0)).is_none());
    /// assert_eq!(Q::one().checked_div(Q::one()), Some(Q::one()));
    /// ```
    #[inline]
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        if rhs.0.as_array().contains(&T::ZERO) {
            None
        } else {
            Some(self.div(rhs))
        }
    }

    /// Lane division, saturating to the format range. `x / 0` maps to
    /// `max_value()` for `x > 0`, `min_value()` for `x < 0` and `0` for `0 / 0`.
    /// Never panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_core::Simd;
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 2>;
    /// let a = Q::from_raw(Simd::from_array([5 << 16, -(5 << 16)]));
    /// assert_eq!(a.saturating_div(Q::splat_raw(0)).raw().to_array(), [i32::MAX, i32::MIN]);
    /// let tiny = Q::epsilon();
    /// assert_eq!(Q::max_value().saturating_div(tiny), Q::max_value());
    /// ```
    #[inline]
    pub fn saturating_div(self, rhs: Self) -> Self {
        Self::check();
        Self::new(self.0.zip_with(rhs.0, |a, b| {
            let (a, b) = (a.widen(), b.widen());
            let q = if b == 0 {
                match a.cmp(&0) {
                    core::cmp::Ordering::Greater => max_w(I, F),
                    core::cmp::Ordering::Less => min_w(I, F),
                    core::cmp::Ordering::Equal => 0,
                }
            } else {
                Self::div_lane(a, b).clamp(min_w(I, F), max_w(I, F))
            };
            T::narrow(q)
        }))
    }

    /// Lane negation, wrapping (`-MIN == MIN`). The `-` operator does this.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 2>;
    /// assert_eq!(Q::min_value().wrapping_neg(), Q::min_value());
    /// ```
    #[inline(always)]
    pub fn wrapping_neg(self) -> Self {
        Self::new(-self.0)
    }

    /// Lane negation, saturating (`-MIN == MAX`).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 2>;
    /// assert_eq!(Q::min_value().saturating_neg(), Q::max_value());
    /// ```
    #[inline(always)]
    pub fn saturating_neg(self) -> Self {
        Self::new(Self::clamp_raw(Simd::<T, N>::zero().sat_sub(self.0)))
    }

    /// Lane absolute value, wrapping (`|MIN| == MIN`).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 2>;
    /// assert_eq!(Q::splat_f32(-2.5).abs(), Q::splat_f32(2.5));
    /// assert_eq!(Q::min_value().abs(), Q::min_value());
    /// ```
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::new(self.0.abs())
    }

    /// Lane absolute value, saturating (`|MIN| == MAX`).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 2>;
    /// assert_eq!(Q::min_value().saturating_abs(), Q::max_value());
    /// ```
    #[inline(always)]
    pub fn saturating_abs(self) -> Self {
        let neg = Simd::<T, N>::zero().sat_sub(self.0);
        Self::new(Self::clamp_raw(self.0.max(neg)))
    }

    /// Lane-wise minimum.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 2>;
    /// assert_eq!(Q::one().min(Q::epsilon()), Q::epsilon());
    /// ```
    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.0.min(rhs.0))
    }

    /// Lane-wise maximum.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 2>;
    /// assert_eq!(Q::one().max(Q::epsilon()), Q::one());
    /// ```
    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.0.max(rhs.0))
    }

    /// Lane-wise clamp to `[lo, hi]`.
    ///
    /// # Panics
    ///
    /// May panic if `lo > hi` in any lane (see `Simd::clamp`).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// type Q = Fixed<i32, 16, 16, 2>;
    /// let x = Q::splat_f32(5.0).clamp(Q::splat_f32(-1.0), Q::splat_f32(1.0));
    /// assert_eq!(x, Q::one());
    /// ```
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self::new(self.0.clamp(lo.0, hi.0))
    }

    /// Convert to another Q format `INT2.FRAC2` of the same storage type.
    ///
    /// Going to more fractional bits shifts left; to fewer, the value is
    /// rounded to nearest (**ties away from zero**). The result is
    /// **saturated** to the target format range. The target format is
    /// checked at compile time like the source.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_simd_fixed::Fixed;
    /// let a = Fixed::<i32, 16, 16, 4>::splat_f32(1.5);
    /// let b: Fixed<i32, 8, 24, 4> = a.convert();
    /// assert_eq!(b.to_f32().to_array(), [1.5; 4]);
    /// let big = Fixed::<i32, 16, 16, 4>::splat_f32(1000.0);
    /// let c: Fixed<i32, 8, 24, 4> = big.convert(); // 1000 > 127.99: saturates
    /// assert_eq!(c, Fixed::max_value());
    /// ```
    #[inline]
    pub fn convert<const I2: u32, const F2: u32>(self) -> Fixed<T, I2, F2, N> {
        Self::check();
        Fixed::<T, I2, F2, N>::check();
        Fixed::new(self.0.map(|v| {
            let w = v.widen();
            let r = if F2 >= F {
                w << (F2 - F)
            } else {
                shr_round_away(w, F - F2)
            };
            T::narrow(r.clamp(min_w(I2, F2), max_w(I2, F2)))
        }))
    }
}

macro_rules! op {
    ($tr:ident, $m:ident, $atr:ident, $am:ident, $impl:ident) => {
        impl<T: FixedRepr, const I: u32, const F: u32, const N: usize> $tr for Fixed<T, I, F, N> {
            type Output = Self;
            #[inline(always)]
            fn $m(self, rhs: Self) -> Self {
                self.$impl(rhs)
            }
        }
        impl<T: FixedRepr, const I: u32, const F: u32, const N: usize> $atr for Fixed<T, I, F, N> {
            #[inline(always)]
            fn $am(&mut self, rhs: Self) {
                *self = self.$impl(rhs);
            }
        }
    };
}

op!(Add, add, AddAssign, add_assign, wrapping_add);
op!(Sub, sub, SubAssign, sub_assign, wrapping_sub);
op!(Mul, mul, MulAssign, mul_assign, wrapping_mul);
op!(Div, div, DivAssign, div_assign, div);

impl<T: FixedRepr, const I: u32, const F: u32, const N: usize> Neg for Fixed<T, I, F, N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
}
