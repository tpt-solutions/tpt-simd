//! Type-safe SIMD fixed-point arithmetic.
//!
//! [`Fixed<T, INT_BITS, FRAC_BITS, N>`](Fixed) wraps a [`Simd<T, N>`](tpt_simd_core::Simd) of
//! raw two's-complement integers interpreted as `raw / 2^FRAC_BITS`.
//!
//! * `INT_BITS` **includes the sign bit** (Q-format `Q<INT_BITS-1>.<FRAC_BITS>`
//!   in ARM notation; `Fixed<i32, 16, 16, N>` is the classic 16.16 format with
//!   range `[-32768, 32768)`).
//! * `INT_BITS >= 1` and `INT_BITS + FRAC_BITS <= T::BITS` are checked at
//!   compile time; a bad format is a compile error as soon as any
//!   constructor or operation is instantiated. If the sum is smaller than
//!   `T::BITS` the format is a *narrow* format stored in a wider lane:
//!   saturating operations clamp to the format range, wrapping operations
//!   wrap at the storage width.
//!
//! ## Overflow policy (ADR 0002)
//!
//! Operators (`+ - * / -x`) **wrap** at the storage width, like integers
//! everywhere in the workspace. Every operation also has explicit
//! `wrapping_*` and `saturating_*` variants; the saturating ones clamp to
//! the format range `[min_value(), max_value()]`.
//!
//! This replaces the spec's `Fixed::saturate`.
//!
//! ## Rounding
//!
//! `mul`, `div`, [`Fixed::convert`] round to nearest, **ties away from
//! zero**. [`Fixed::mul_trunc`] truncates (arithmetic shift, i.e. floors).
//! `from_f32` rounds to nearest, **ties to even**.
//!
//! ## Backends
//!
//! The portable implementation is the reference. With
//! `target_feature = "avx2"` (and not `scalar-only`), `i32` multiplies use
//! `_mm256_mul_epi32` on even/odd lanes with 64-bit intermediates; results
//! are bit-identical to the portable path.
//!
//! ```
//! use tpt_simd_core::F32x8;
//! use tpt_simd_fixed::Fixed;
//!
//! let a: Fixed<i32, 16, 16, 8> = Fixed::from_f32(F32x8::splat(1.5));
//! let b: Fixed<i32, 16, 16, 8> = Fixed::from_f32(F32x8::splat(2.5));
//! let result = a * b; // 1.5 * 2.5 = 3.75 with correct rounding
//! assert_eq!(result.to_f32(), F32x8::splat(3.75));
//! ```
#![no_std]
#![cfg_attr(
    not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    )),
    forbid(unsafe_code)
)]

#[cfg(any(feature = "std", test))]
extern crate std;

mod fixed;
mod repr;

pub use fixed::Fixed;
pub use repr::FixedRepr;

#[cfg(test)]
mod tests;
