//! Backend-abstraction traits.

use tpt_simd_vector::{Simd, SimdElement};

/// Operations every backend vector type provides.
pub trait SimdVector<T: SimdElement, const N: usize>: Copy {
    /// Number of lanes.
    const LANES: usize = N;
    /// Broadcast a scalar.
    fn splat(value: T) -> Self;
    /// Build from an array.
    fn from_array(array: [T; N]) -> Self;
    /// Convert to an array.
    fn to_array(self) -> [T; N];
}

impl<T: SimdElement, const N: usize> SimdVector<T, N> for Simd<T, N> {
    #[inline(always)]
    fn splat(value: T) -> Self {
        Simd::splat(value)
    }
    #[inline(always)]
    fn from_array(array: [T; N]) -> Self {
        Simd::from_array(array)
    }
    #[inline(always)]
    fn to_array(self) -> [T; N] {
        Simd::to_array(self)
    }
}

/// Maps an element type and lane count to a backend vector type.
pub trait SimdOps<T: SimdElement, const N: usize> {
    /// The vector type.
    type Vector: SimdVector<T, N>;
}

/// The backend selected at compile time (`tpt-simd-vector`'s [`Simd`]).
pub struct DefaultBackend;

impl<T: SimdElement, const N: usize> SimdOps<T, N> for DefaultBackend {
    type Vector = Simd<T, N>;
}
