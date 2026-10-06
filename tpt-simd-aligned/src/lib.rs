//! SIMD-aligned memory types and helpers.
//!
//! * [`Aligned16`], [`Aligned32`], [`Aligned64`] — `#[repr(align(N))]`
//!   wrappers that force a value onto a 16/32/64-byte boundary (SSE, AVX2,
//!   AVX-512/cache line).
//! * [`is_aligned`] — pointer alignment check.
//! * [`load_aligned`] / [`store_aligned`] (and `try_` variants) — safe,
//!   alignment-checked [`Simd`] loads and stores on slices.
//! * `AlignedBuf` and the `aligned_vec_*` helpers (feature `alloc`, on by
//!   default) — a heap buffer whose first element is aligned to a
//!   const-generic boundary. Unlike `Vec<Aligned32<f32>>` (which pads every
//!   `f32` to 32 bytes) the elements are densely packed.
//!
//! ## Panic policy
//!
//! Per ADR 0002 the plain functions panic (documented under `# Panics`) when
//! the slice is too short or misaligned; the `try_*` variants return `None`
//! instead.
#![no_std]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "alloc")]
extern crate alloc;

use core::ops::{Deref, DerefMut};
use tpt_simd_core::{Simd, SimdElement};

macro_rules! aligned_type {
    ($(#[$m:meta])* $name:ident, $n:literal) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(C, align($n))]
        pub struct $name<T>(pub T);

        impl<T> $name<T> {
            /// Alignment of this wrapper in bytes.
            pub const ALIGN: usize = $n;

            /// Wraps `value`.
            #[inline(always)]
            pub const fn new(value: T) -> Self {
                Self(value)
            }

            /// Unwraps the inner value.
            #[inline(always)]
            pub fn into_inner(self) -> T {
                self.0
            }
        }

        impl<T> Deref for $name<T> {
            type Target = T;
            #[inline(always)]
            fn deref(&self) -> &T {
                &self.0
            }
        }

        impl<T> DerefMut for $name<T> {
            #[inline(always)]
            fn deref_mut(&mut self) -> &mut T {
                &mut self.0
            }
        }

        impl<T> From<T> for $name<T> {
            #[inline(always)]
            fn from(value: T) -> Self {
                Self(value)
            }
        }
    };
}

aligned_type!(
    /// Forces `T` onto a 16-byte boundary (SSE `movaps`).
    ///
    /// ```
    /// use tpt_simd_aligned::Aligned16;
    /// let a = Aligned16::new([1.0f32; 4]);
    /// assert_eq!(core::mem::align_of_val(&a), 16);
    /// assert_eq!(a[2], 1.0);
    /// ```
    Aligned16,
    16
);
aligned_type!(
    /// Forces `T` onto a 32-byte boundary (AVX2 `vmovaps ymm`).
    ///
    /// ```
    /// use tpt_simd_aligned::Aligned32;
    /// let a = Aligned32([0i16; 16]);
    /// assert_eq!(core::mem::align_of_val(&a), 32);
    /// assert_eq!((&a as *const _ as usize) % 32, 0);
    /// ```
    Aligned32,
    32
);
aligned_type!(
    /// Forces `T` onto a 64-byte boundary (AVX-512 / cache line).
    ///
    /// ```
    /// use tpt_simd_aligned::Aligned64;
    /// let mut a = Aligned64::new(5u8);
    /// *a += 1;
    /// assert_eq!(a.into_inner(), 6);
    /// assert_eq!(core::mem::size_of_val(&a), 64);
    /// ```
    Aligned64,
    64
);

/// Returns `true` if `ptr` is a multiple of `alignment` bytes.
///
/// `alignment` must be a power of two; anything else (including 0) returns
/// `false` rather than panicking. Null is considered aligned to everything
/// (address 0).
///
/// ```
/// use tpt_simd_aligned::{is_aligned, Aligned32};
/// let a = Aligned32([0u8; 32]);
/// assert!(is_aligned(a.0.as_ptr(), 32));
/// assert!(!is_aligned(a.0.as_ptr().wrapping_add(1), 2));
/// assert!(!is_aligned(a.0.as_ptr(), 3));
/// ```
#[inline]
pub fn is_aligned<T>(ptr: *const T, alignment: usize) -> bool {
    alignment.is_power_of_two() && (ptr as usize) & (alignment - 1) == 0
}

/// Byte alignment required by [`load_aligned`] / [`store_aligned`] for a
/// vector of `N` lanes of `T` (the vector's size in bytes, e.g. 32 for
/// `Simd<f32, 8>`), rounded up to a power of two.
#[inline(always)]
pub const fn vector_align<T: SimdElement, const N: usize>() -> usize {
    (core::mem::size_of::<T>() * N).next_power_of_two()
}

/// Loads `N` lanes from the start of `slice`, or `None` if the slice has
/// fewer than `N` elements or its address is not a multiple of
/// [`vector_align::<T, N>()`](vector_align).
///
/// ```
/// use tpt_simd_aligned::{try_load_aligned, Aligned32};
/// use tpt_simd_core::Simd;
/// let buf = Aligned32([1.0f32; 16]);
/// assert!(try_load_aligned::<f32, 8>(&buf.0).is_some());
/// assert!(try_load_aligned::<f32, 8>(&buf.0[1..]).is_none());
/// assert!(try_load_aligned::<f32, 8>(&buf.0[..4]).is_none());
/// ```
#[inline]
pub fn try_load_aligned<T: SimdElement, const N: usize>(slice: &[T]) -> Option<Simd<T, N>> {
    if slice.len() < N || !is_aligned(slice.as_ptr(), vector_align::<T, N>()) {
        return None;
    }
    Some(Simd::from_slice(slice))
}

/// Stores `v` to the start of `slice`; returns `false` (and writes nothing)
/// if the slice is too short or misaligned for [`vector_align::<T, N>()`](vector_align).
///
/// ```
/// use tpt_simd_aligned::{try_store_aligned, Aligned16};
/// use tpt_simd_core::Simd;
/// let mut buf = Aligned16([0i32; 8]);
/// assert!(try_store_aligned(Simd::<i32, 4>::splat(7), &mut buf.0));
/// assert!(!try_store_aligned(Simd::<i32, 4>::splat(9), &mut buf.0[1..]));
/// assert_eq!(buf.0, [7, 7, 7, 7, 0, 0, 0, 0]);
/// ```
#[inline]
pub fn try_store_aligned<T: SimdElement, const N: usize>(v: Simd<T, N>, slice: &mut [T]) -> bool {
    if slice.len() < N || !is_aligned(slice.as_ptr(), vector_align::<T, N>()) {
        return false;
    }
    v.copy_to_slice(slice);
    true
}

/// Alignment-checked [`Simd`] load. Panicking form of [`try_load_aligned`].
///
/// Performance: the check is two compares; the load itself compiles to an
/// aligned vector move.
///
/// # Panics
/// If `slice.len() < N` or `slice` is not aligned to
/// [`vector_align::<T, N>()`](vector_align) bytes.
///
/// ```
/// use tpt_simd_aligned::{load_aligned, Aligned32};
/// let buf = Aligned32([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
/// let v = load_aligned::<f32, 8>(&buf.0);
/// assert_eq!(v[7], 8.0);
/// ```
#[inline]
pub fn load_aligned<T: SimdElement, const N: usize>(slice: &[T]) -> Simd<T, N> {
    assert!(slice.len() >= N, "slice too short for aligned load");
    assert!(
        is_aligned(slice.as_ptr(), vector_align::<T, N>()),
        "slice is not aligned for aligned load"
    );
    Simd::from_slice(slice)
}

/// Alignment-checked [`Simd`] store. Panicking form of [`try_store_aligned`].
///
/// # Panics
/// If `slice.len() < N` or `slice` is not aligned to
/// [`vector_align::<T, N>()`](vector_align) bytes.
///
/// ```
/// use tpt_simd_aligned::{store_aligned, Aligned32};
/// use tpt_simd_core::Simd;
/// let mut buf = Aligned32([0.0f32; 8]);
/// store_aligned(Simd::<f32, 8>::splat(2.5), &mut buf.0);
/// assert_eq!(buf.0, [2.5; 8]);
/// ```
#[inline]
pub fn store_aligned<T: SimdElement, const N: usize>(v: Simd<T, N>, slice: &mut [T]) {
    assert!(slice.len() >= N, "slice too short for aligned store");
    assert!(
        is_aligned(slice.as_ptr(), vector_align::<T, N>()),
        "slice is not aligned for aligned store"
    );
    v.copy_to_slice(slice);
}

#[cfg(feature = "alloc")]
mod buf;
#[cfg(feature = "alloc")]
pub use buf::{AlignedBuf, aligned_vec_f32, aligned_vec_i16, aligned_vec_i32};

#[cfg(test)]
mod tests;
