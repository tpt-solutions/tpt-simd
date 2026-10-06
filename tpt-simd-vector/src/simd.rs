//! The array-backed [`Simd`] vector type.

use crate::element::{LaneCast, SimdElement, SimdFloat, SimdInt};
use crate::mask::SimdMask;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Index, IndexMut, Mul, MulAssign, Neg, Not, Shl, Shr, Sub, SubAssign,
};

/// A vector of `N` lanes of type `T`.
///
/// The layout is `#[repr(transparent)]` over `[T; N]`: lane `i` lives at
/// byte offset `i * size_of::<T>()`. Operations are written as fixed-length
/// lane loops that LLVM turns into SIMD instructions for the target.
///
/// Integer arithmetic wraps; see [`SimdElement`] for the full policy.
///
/// ```
/// use tpt_simd_vector::Simd;
/// let a = Simd::<f32, 4>::from_array([1.0, 2.0, 3.0, 4.0]);
/// let b = Simd::splat(2.0);
/// assert_eq!((a * b).to_array(), [2.0, 4.0, 6.0, 8.0]);
/// ```
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Simd<T: SimdElement, const N: usize>(pub [T; N]);

impl<T: SimdElement, const N: usize> Simd<T, N> {
    /// Number of lanes.
    pub const LANES: usize = N;

    /// All lanes set to `value`.
    #[inline(always)]
    pub fn splat(value: T) -> Self {
        Self([value; N])
    }

    /// All lanes zero.
    #[inline(always)]
    pub fn zero() -> Self {
        Self([T::ZERO; N])
    }

    /// Build from an array (lane `i` = `array[i]`).
    #[inline(always)]
    pub const fn from_array(array: [T; N]) -> Self {
        Self(array)
    }

    /// Convert to an array.
    #[inline(always)]
    pub const fn to_array(self) -> [T; N] {
        self.0
    }

    /// Borrow as an array.
    #[inline(always)]
    pub const fn as_array(&self) -> &[T; N] {
        &self.0
    }

    /// Mutably borrow as an array.
    #[inline(always)]
    pub fn as_mut_array(&mut self) -> &mut [T; N] {
        &mut self.0
    }

    /// Build a vector from a function of the lane index.
    #[inline(always)]
    pub fn from_fn(f: impl FnMut(usize) -> T) -> Self {
        Self(core::array::from_fn(f))
    }

    /// Load `N` lanes from the start of `slice`.
    ///
    /// # Panics
    /// If `slice.len() < N`.
    #[inline(always)]
    pub fn from_slice(slice: &[T]) -> Self {
        assert!(slice.len() >= N, "slice too short for Simd::from_slice");
        let mut out = [T::ZERO; N];
        out.copy_from_slice(&slice[..N]);
        Self(out)
    }

    /// Load up to `N` lanes from `slice`; missing lanes are `fill`.
    #[inline(always)]
    pub fn from_slice_or(slice: &[T], fill: T) -> Self {
        let mut out = [fill; N];
        let n = slice.len().min(N);
        out[..n].copy_from_slice(&slice[..n]);
        Self(out)
    }

    /// Store all `N` lanes to the start of `slice`.
    ///
    /// # Panics
    /// If `slice.len() < N`.
    #[inline(always)]
    pub fn copy_to_slice(self, slice: &mut [T]) {
        assert!(slice.len() >= N, "slice too short for Simd::copy_to_slice");
        slice[..N].copy_from_slice(&self.0);
    }

    /// Store as many lanes as fit in `slice` (at most `N`).
    #[inline(always)]
    pub fn store_partial(self, slice: &mut [T]) {
        let n = slice.len().min(N);
        slice[..n].copy_from_slice(&self.0[..n]);
    }

    /// Apply `f` to every lane.
    #[inline(always)]
    pub fn map<U: SimdElement>(self, mut f: impl FnMut(T) -> U) -> Simd<U, N> {
        Simd(core::array::from_fn(|i| f(self.0[i])))
    }

    /// Combine two vectors lane by lane with `f`.
    #[inline(always)]
    pub fn zip_with<U: SimdElement>(self, other: Self, mut f: impl FnMut(T, T) -> U) -> Simd<U, N> {
        Simd(core::array::from_fn(|i| f(self.0[i], other.0[i])))
    }

    /// Numeric `as`-style cast of every lane. See [`LaneCast`].
    #[inline(always)]
    pub fn cast<U: SimdElement>(self) -> Simd<U, N>
    where
        T: LaneCast<U>,
    {
        Simd(core::array::from_fn(|i| self.0[i].lane_cast()))
    }

    /// Lane-wise minimum (NaN operands ignored for floats).
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].lane_min(other.0[i])))
    }

    /// Lane-wise maximum (NaN operands ignored for floats).
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].lane_max(other.0[i])))
    }

    /// Clamp every lane to `[lo, hi]`.
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Lane-wise absolute value (wrapping for the minimum signed integer).
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].lane_abs()))
    }

    /// Reverse the lane order.
    #[inline(always)]
    pub fn reverse(self) -> Self {
        Self(core::array::from_fn(|i| self.0[N - 1 - i]))
    }

    /// Rearrange lanes: `out[i] = self[indices[i]]`.
    ///
    /// # Panics
    /// If any index is `>= N`.
    #[inline(always)]
    pub fn swizzle(self, indices: [usize; N]) -> Self {
        Self(core::array::from_fn(|i| self.0[indices[i]]))
    }

    /// `self == other` per lane.
    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> SimdMask<T, N> {
        SimdMask::from_array(core::array::from_fn(|i| self.0[i] == other.0[i]))
    }

    /// `self != other` per lane.
    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> SimdMask<T, N> {
        SimdMask::from_array(core::array::from_fn(|i| self.0[i] != other.0[i]))
    }

    /// `self < other` per lane.
    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> SimdMask<T, N> {
        SimdMask::from_array(core::array::from_fn(|i| self.0[i] < other.0[i]))
    }

    /// `self <= other` per lane.
    #[inline(always)]
    pub fn simd_le(self, other: Self) -> SimdMask<T, N> {
        SimdMask::from_array(core::array::from_fn(|i| self.0[i] <= other.0[i]))
    }

    /// `self > other` per lane.
    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> SimdMask<T, N> {
        SimdMask::from_array(core::array::from_fn(|i| self.0[i] > other.0[i]))
    }

    /// `self >= other` per lane.
    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> SimdMask<T, N> {
        SimdMask::from_array(core::array::from_fn(|i| self.0[i] >= other.0[i]))
    }
}

impl<T: SimdInt, const N: usize> Simd<T, N> {
    /// Saturating lane addition.
    #[inline(always)]
    pub fn sat_add(self, other: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].sat_add(other.0[i])))
    }

    /// Saturating lane subtraction.
    #[inline(always)]
    pub fn sat_sub(self, other: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].sat_sub(other.0[i])))
    }

    /// Saturating lane multiplication.
    #[inline(always)]
    pub fn sat_mul(self, other: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].sat_mul(other.0[i])))
    }

    /// Shift every lane left by `amount` bits (`amount >= BITS` gives 0).
    #[inline(always)]
    pub fn shl_scalar(self, amount: u32) -> Self {
        Self(core::array::from_fn(|i| self.0[i].lane_shl(amount)))
    }

    /// Shift every lane right by `amount` bits: arithmetic for signed
    /// lanes, logical for unsigned (`amount >= BITS` gives the sign fill).
    #[inline(always)]
    pub fn shr_scalar(self, amount: u32) -> Self {
        Self(core::array::from_fn(|i| self.0[i].lane_shr(amount)))
    }

    /// Logical shift right of every lane, regardless of signedness.
    #[inline(always)]
    pub fn shr_logical(self, amount: u32) -> Self {
        Self(core::array::from_fn(|i| self.0[i].lane_shr_logical(amount)))
    }

    /// Rotate every lane left by `amount % BITS`.
    #[inline(always)]
    pub fn rotl(self, amount: u32) -> Self {
        Self(core::array::from_fn(|i| self.0[i].lane_rotl(amount)))
    }
}

impl<T: SimdFloat, const N: usize> Simd<T, N> {
    /// Square root of every lane.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].lane_sqrt()))
    }

    /// Floor of every lane.
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].lane_floor()))
    }

    /// Ceiling of every lane.
    #[inline(always)]
    pub fn ceil(self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].lane_ceil()))
    }

    /// Round to nearest, ties away from zero.
    #[inline(always)]
    pub fn round(self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].lane_round()))
    }

    /// Round to nearest, ties to even.
    #[inline(always)]
    pub fn round_ties_even(self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].lane_round_ties_even()))
    }

    /// Truncate toward zero.
    #[inline(always)]
    pub fn trunc(self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].lane_trunc()))
    }

    /// Fused multiply-add `self * a + b` (single rounding per lane).
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(core::array::from_fn(|i| {
            self.0[i].lane_mul_add(a.0[i], b.0[i])
        }))
    }

    /// Per-lane `atan2(self, x)`.
    #[inline(always)]
    pub fn atan2(self, x: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].lane_atan2(x.0[i])))
    }

    /// Mask of lanes that are NaN.
    #[inline(always)]
    pub fn is_nan(self) -> SimdMask<T, N> {
        SimdMask::from_array(core::array::from_fn(|i| self.0[i].lane_is_nan()))
    }
}

// ---- trait impls ---------------------------------------------------------

impl<T: SimdElement, const N: usize> Default for Simd<T, N> {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl<T: SimdElement, const N: usize> core::fmt::Debug for Simd<T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Simd").field(&self.0).finish()
    }
}

/// Lane-wise equality of *all* lanes (float NaN lanes compare unequal).
impl<T: SimdElement, const N: usize> PartialEq for Simd<T, N> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: SimdElement, const N: usize> From<[T; N]> for Simd<T, N> {
    #[inline(always)]
    fn from(a: [T; N]) -> Self {
        Self(a)
    }
}

impl<T: SimdElement, const N: usize> From<Simd<T, N>> for [T; N] {
    #[inline(always)]
    fn from(v: Simd<T, N>) -> Self {
        v.0
    }
}

impl<T: SimdElement, const N: usize> Index<usize> for Simd<T, N> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        &self.0[i]
    }
}

impl<T: SimdElement, const N: usize> IndexMut<usize> for Simd<T, N> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.0[i]
    }
}

macro_rules! binop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident, $lane:ident) => {
        impl<T: SimdElement, const N: usize> $tr for Simd<T, N> {
            type Output = Self;
            #[inline(always)]
            fn $m(self, rhs: Self) -> Self {
                Self(core::array::from_fn(|i| self.0[i].$lane(rhs.0[i])))
            }
        }
        impl<T: SimdElement, const N: usize> $atr for Simd<T, N> {
            #[inline(always)]
            fn $am(&mut self, rhs: Self) {
                *self = $tr::$m(*self, rhs);
            }
        }
    };
}
binop!(Add, add, AddAssign, add_assign, lane_add);
binop!(Sub, sub, SubAssign, sub_assign, lane_sub);
binop!(Mul, mul, MulAssign, mul_assign, lane_mul);
binop!(Div, div, DivAssign, div_assign, lane_div);

impl<T: SimdElement, const N: usize> Neg for Simd<T, N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].lane_neg()))
    }
}

macro_rules! bitop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident, $op:tt) => {
        impl<T: SimdInt, const N: usize> $tr for Simd<T, N> {
            type Output = Self;
            #[inline(always)]
            fn $m(self, rhs: Self) -> Self {
                Self(core::array::from_fn(|i| self.0[i] $op rhs.0[i]))
            }
        }
        impl<T: SimdInt, const N: usize> $atr for Simd<T, N> {
            #[inline(always)]
            fn $am(&mut self, rhs: Self) {
                *self = $tr::$m(*self, rhs);
            }
        }
    };
}
bitop!(BitAnd, bitand, BitAndAssign, bitand_assign, &);
bitop!(BitOr, bitor, BitOrAssign, bitor_assign, |);
bitop!(BitXor, bitxor, BitXorAssign, bitxor_assign, ^);

impl<T: SimdInt, const N: usize> Not for Simd<T, N> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(core::array::from_fn(|i| !self.0[i]))
    }
}

impl<T: SimdInt, const N: usize> Shl<u32> for Simd<T, N> {
    type Output = Self;
    #[inline(always)]
    fn shl(self, amount: u32) -> Self {
        self.shl_scalar(amount)
    }
}

impl<T: SimdInt, const N: usize> Shr<u32> for Simd<T, N> {
    type Output = Self;
    #[inline(always)]
    fn shr(self, amount: u32) -> Self {
        self.shr_scalar(amount)
    }
}
