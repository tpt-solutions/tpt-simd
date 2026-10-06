//! Lane masks produced by comparisons.

use crate::element::SimdElement;
use crate::simd::Simd;
use core::marker::PhantomData;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

/// A per-lane boolean mask for vectors of `N` lanes of `T`.
///
/// Produced by [`Simd::simd_eq`] and friends; consumed by
/// [`SimdMask::select`]. `T` only ties the mask to a lane type so masks of
/// different element types cannot be mixed up.
pub struct SimdMask<T: SimdElement, const N: usize> {
    lanes: [bool; N],
    _marker: PhantomData<fn() -> T>,
}

impl<T: SimdElement, const N: usize> SimdMask<T, N> {
    /// Build from per-lane booleans.
    #[inline(always)]
    pub const fn from_array(lanes: [bool; N]) -> Self {
        Self { lanes, _marker: PhantomData }
    }

    /// Per-lane booleans.
    #[inline(always)]
    pub const fn to_array(self) -> [bool; N] {
        self.lanes
    }

    /// Every lane set to `value`.
    #[inline(always)]
    pub const fn splat(value: bool) -> Self {
        Self::from_array([value; N])
    }

    /// Whether lane `i` is set.
    ///
    /// # Panics
    /// If `i >= N`.
    #[inline(always)]
    pub fn test(&self, i: usize) -> bool {
        self.lanes[i]
    }

    /// Set lane `i`.
    ///
    /// # Panics
    /// If `i >= N`.
    #[inline(always)]
    pub fn set(&mut self, i: usize, value: bool) {
        self.lanes[i] = value;
    }

    /// `true` if any lane is set.
    #[inline(always)]
    pub fn any(self) -> bool {
        self.lanes.iter().fold(false, |a, &b| a | b)
    }

    /// `true` if every lane is set.
    #[inline(always)]
    pub fn all(self) -> bool {
        self.lanes.iter().fold(true, |a, &b| a & b)
    }

    /// Number of set lanes.
    #[inline(always)]
    pub fn count(self) -> usize {
        self.lanes.iter().fold(0usize, |a, &b| a + b as usize)
    }

    /// Pack into an integer, lane 0 in bit 0.
    ///
    /// # Panics
    /// If `N > 64`.
    #[inline(always)]
    pub fn to_bitmask(self) -> u64 {
        assert!(N <= 64, "to_bitmask supports at most 64 lanes");
        let mut m = 0u64;
        let mut i = 0;
        while i < N {
            m |= (self.lanes[i] as u64) << i;
            i += 1;
        }
        m
    }

    /// Unpack from an integer, bit 0 → lane 0. Bits above `N` are ignored.
    ///
    /// # Panics
    /// If `N > 64`.
    #[inline(always)]
    pub fn from_bitmask(bits: u64) -> Self {
        assert!(N <= 64, "from_bitmask supports at most 64 lanes");
        Self::from_array(core::array::from_fn(|i| (bits >> i) & 1 != 0))
    }

    /// Per lane: `if self { a } else { b }`.
    #[inline(always)]
    pub fn select(self, a: Simd<T, N>, b: Simd<T, N>) -> Simd<T, N> {
        Simd::from_array(core::array::from_fn(|i| if self.lanes[i] { a[i] } else { b[i] }))
    }
}

impl<T: SimdElement, const N: usize> Clone for SimdMask<T, N> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: SimdElement, const N: usize> Copy for SimdMask<T, N> {}

impl<T: SimdElement, const N: usize> PartialEq for SimdMask<T, N> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.lanes == other.lanes
    }
}
impl<T: SimdElement, const N: usize> Eq for SimdMask<T, N> {}

impl<T: SimdElement, const N: usize> core::fmt::Debug for SimdMask<T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("SimdMask").field(&self.lanes).finish()
    }
}

impl<T: SimdElement, const N: usize> Default for SimdMask<T, N> {
    fn default() -> Self {
        Self::splat(false)
    }
}

impl<T: SimdElement, const N: usize> Not for SimdMask<T, N> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self::from_array(core::array::from_fn(|i| !self.lanes[i]))
    }
}

macro_rules! maskop {
    ($tr:ident, $m:ident, $atr:ident, $am:ident, $op:tt) => {
        impl<T: SimdElement, const N: usize> $tr for SimdMask<T, N> {
            type Output = Self;
            #[inline(always)]
            fn $m(self, rhs: Self) -> Self {
                Self::from_array(core::array::from_fn(|i| self.lanes[i] $op rhs.lanes[i]))
            }
        }
        impl<T: SimdElement, const N: usize> $atr for SimdMask<T, N> {
            #[inline(always)]
            fn $am(&mut self, rhs: Self) {
                *self = $tr::$m(*self, rhs);
            }
        }
    };
}
maskop!(BitAnd, bitand, BitAndAssign, bitand_assign, &);
maskop!(BitOr, bitor, BitOrAssign, bitor_assign, |);
maskop!(BitXor, bitxor, BitXorAssign, bitxor_assign, ^);
