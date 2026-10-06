//! Lane element traits: the scalar types a [`Simd`](crate::Simd) vector can hold.
//!
//! * [`SimdElement`]: arithmetic every lane type supports (integers wrap).
//! * [`SimdInt`]: integer-only operations (bitwise, saturating, shifts).
//! * [`SimdFloat`]: float-only operations (`sqrt`, rounding, FMA, ...).
//! * [`LaneCast`]: `as`-style numeric casts between lane types.
//!
//! Float math goes through `libm` so results are identical on every target
//! and in `no_std`.

use core::fmt::Debug;
use core::ops::{BitAnd, BitOr, BitXor, Not};

/// A scalar type that can be a lane of a [`Simd`](crate::Simd) vector.
///
/// Integer arithmetic is **wrapping** (two's complement), matching what SIMD
/// hardware does. Float min/max ignore NaN operands (like `f32::min`).
pub trait SimdElement:
    Copy + Default + PartialEq + PartialOrd + Debug + Send + Sync + 'static
{
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
    /// Smallest finite value (`-MAX` for floats is *not* used; this is `MIN`).
    const MIN: Self;
    /// Largest finite value.
    const MAX: Self;
    /// Width of the element in bits.
    const BITS: u32;
    /// `true` for `f32` / `f64`.
    const IS_FLOAT: bool;
    /// `true` for signed integers and floats.
    const IS_SIGNED: bool;

    /// Lane addition (wrapping for integers).
    fn lane_add(self, rhs: Self) -> Self;
    /// Lane subtraction (wrapping for integers).
    fn lane_sub(self, rhs: Self) -> Self;
    /// Lane multiplication (wrapping for integers).
    fn lane_mul(self, rhs: Self) -> Self;
    /// Lane division. Integer division panics on a zero divisor;
    /// `MIN / -1` wraps.
    fn lane_div(self, rhs: Self) -> Self;
    /// Lane negation (wrapping for integers; `0 - x` for unsigned).
    fn lane_neg(self) -> Self;
    /// Lane minimum. For floats a NaN operand is ignored.
    fn lane_min(self, rhs: Self) -> Self;
    /// Lane maximum. For floats a NaN operand is ignored.
    fn lane_max(self, rhs: Self) -> Self;
    /// Absolute value (wrapping for `MIN`, identity for unsigned).
    fn lane_abs(self) -> Self;
}

/// Integer lane types.
pub trait SimdInt:
    SimdElement
    + Eq
    + Ord
    + Not<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
{
    /// Saturating addition.
    fn sat_add(self, rhs: Self) -> Self;
    /// Saturating subtraction.
    fn sat_sub(self, rhs: Self) -> Self;
    /// Saturating multiplication.
    fn sat_mul(self, rhs: Self) -> Self;
    /// Shift left. `amount >= BITS` yields `0`.
    fn lane_shl(self, amount: u32) -> Self;
    /// Shift right: arithmetic for signed types, logical for unsigned.
    /// `amount >= BITS` yields `0` (unsigned) or the sign fill (signed).
    fn lane_shr(self, amount: u32) -> Self;
    /// Logical shift right for every type. `amount >= BITS` yields `0`.
    fn lane_shr_logical(self, amount: u32) -> Self;
    /// Rotate left by `amount % BITS`.
    fn lane_rotl(self, amount: u32) -> Self;
}

/// Floating-point lane types.
pub trait SimdFloat: SimdElement {
    /// Square root.
    fn lane_sqrt(self) -> Self;
    /// Round toward negative infinity.
    fn lane_floor(self) -> Self;
    /// Round toward positive infinity.
    fn lane_ceil(self) -> Self;
    /// Round to nearest, ties away from zero.
    fn lane_round(self) -> Self;
    /// Round to nearest, ties to even.
    fn lane_round_ties_even(self) -> Self;
    /// Round toward zero.
    fn lane_trunc(self) -> Self;
    /// Fused multiply-add: `self * a + b` with a single rounding.
    fn lane_mul_add(self, a: Self, b: Self) -> Self;
    /// `true` if the value is NaN.
    fn lane_is_nan(self) -> bool;
    /// Four-quadrant arctangent of `self / x` (`self` is `y`).
    fn lane_atan2(self, x: Self) -> Self;
}

/// `as`-style numeric cast of a lane to another lane type.
///
/// Float to integer casts saturate and map NaN to zero; integer
/// narrowing truncates (the semantics of Rust's `as`).
pub trait LaneCast<U: SimdElement>: SimdElement {
    /// Perform the cast.
    fn lane_cast(self) -> U;
}

macro_rules! impl_int {
    ($($t:ty, $u:ty, $signed:expr);* $(;)?) => {$(
        #[allow(unused_comparisons)]
        impl SimdElement for $t {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const MIN: Self = <$t>::MIN;
            const MAX: Self = <$t>::MAX;
            const BITS: u32 = <$t>::BITS;
            const IS_FLOAT: bool = false;
            const IS_SIGNED: bool = $signed;
            #[inline(always)] fn lane_add(self, r: Self) -> Self { self.wrapping_add(r) }
            #[inline(always)] fn lane_sub(self, r: Self) -> Self { self.wrapping_sub(r) }
            #[inline(always)] fn lane_mul(self, r: Self) -> Self { self.wrapping_mul(r) }
            #[inline(always)] fn lane_div(self, r: Self) -> Self { self.wrapping_div(r) }
            #[inline(always)] fn lane_neg(self) -> Self { (0 as $t).wrapping_sub(self) }
            #[inline(always)] fn lane_min(self, r: Self) -> Self { if r < self { r } else { self } }
            #[inline(always)] fn lane_max(self, r: Self) -> Self { if r > self { r } else { self } }
            #[inline(always)] fn lane_abs(self) -> Self {
                if $signed && self < 0 as $t { (0 as $t).wrapping_sub(self) } else { self }
            }
        }
        #[allow(unused_comparisons)]
        impl SimdInt for $t {
            #[inline(always)] fn sat_add(self, r: Self) -> Self { self.saturating_add(r) }
            #[inline(always)] fn sat_sub(self, r: Self) -> Self { self.saturating_sub(r) }
            #[inline(always)] fn sat_mul(self, r: Self) -> Self { self.saturating_mul(r) }
            #[inline(always)]
            fn lane_shl(self, n: u32) -> Self { if n >= <$t>::BITS { 0 } else { self << n } }
            #[inline(always)]
            fn lane_shr(self, n: u32) -> Self {
                if n >= <$t>::BITS {
                    if $signed && self < 0 as $t { !(0 as $t) } else { 0 }
                } else {
                    self >> n
                }
            }
            #[inline(always)]
            fn lane_shr_logical(self, n: u32) -> Self {
                if n >= <$t>::BITS { 0 } else { ((self as $u) >> n) as $t }
            }
            #[inline(always)] fn lane_rotl(self, n: u32) -> Self { self.rotate_left(n % <$t>::BITS) }
        }
    )*};
}

impl_int! {
    i8, u8, true; i16, u16, true; i32, u32, true; i64, u64, true;
    u8, u8, false; u16, u16, false; u32, u32, false; u64, u64, false;
}

macro_rules! impl_float {
    ($t:ty, $sqrt:ident, $floor:ident, $ceil:ident, $round:ident, $rint:ident,
     $trunc:ident, $fma:ident, $fabs:ident, $fmin:ident, $fmax:ident, $atan2:ident, $bits:expr) => {
        impl SimdElement for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const MIN: Self = <$t>::MIN;
            const MAX: Self = <$t>::MAX;
            const BITS: u32 = $bits;
            const IS_FLOAT: bool = true;
            const IS_SIGNED: bool = true;
            #[inline(always)] fn lane_add(self, r: Self) -> Self { self + r }
            #[inline(always)] fn lane_sub(self, r: Self) -> Self { self - r }
            #[inline(always)] fn lane_mul(self, r: Self) -> Self { self * r }
            #[inline(always)] fn lane_div(self, r: Self) -> Self { self / r }
            #[inline(always)] fn lane_neg(self) -> Self { -self }
            #[inline(always)] fn lane_min(self, r: Self) -> Self { libm::$fmin(self, r) }
            #[inline(always)] fn lane_max(self, r: Self) -> Self { libm::$fmax(self, r) }
            #[inline(always)] fn lane_abs(self) -> Self { libm::$fabs(self) }
        }
        impl SimdFloat for $t {
            #[inline(always)] fn lane_sqrt(self) -> Self { libm::$sqrt(self) }
            #[inline(always)] fn lane_floor(self) -> Self { libm::$floor(self) }
            #[inline(always)] fn lane_ceil(self) -> Self { libm::$ceil(self) }
            #[inline(always)] fn lane_round(self) -> Self { libm::$round(self) }
            #[inline(always)] fn lane_round_ties_even(self) -> Self { libm::$rint(self) }
            #[inline(always)] fn lane_trunc(self) -> Self { libm::$trunc(self) }
            #[inline(always)] fn lane_mul_add(self, a: Self, b: Self) -> Self { libm::$fma(self, a, b) }
            #[inline(always)] fn lane_is_nan(self) -> bool { self != self }
            #[inline(always)] fn lane_atan2(self, x: Self) -> Self { libm::$atan2(self, x) }
        }
    };
}

impl_float!(f32, sqrtf, floorf, ceilf, roundf, rintf, truncf, fmaf, fabsf, fminf, fmaxf, atan2f, 32);
impl_float!(f64, sqrt, floor, ceil, round, rint, trunc, fma, fabs, fmin, fmax, atan2, 64);

macro_rules! cast_from {
    ($t:ty; $($u:ty),*) => {$(
        impl LaneCast<$u> for $t {
            #[inline(always)] fn lane_cast(self) -> $u { self as $u }
        }
    )*};
}
macro_rules! cast_all {
    ($($t:ty),*) => {$(
        cast_from!($t; i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
    )*};
}
cast_all!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
