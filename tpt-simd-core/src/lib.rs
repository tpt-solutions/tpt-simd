//! Core traits, type aliases, width constants and feature detection for tpt-simd.
//!
//! Re-exports the backend ([`Simd`], [`SimdMask`] and the lane traits from
//! `tpt-simd-vector`) and adds:
//!
//! * [`aliases`]: `F32x8`, `I16x16`, ... for 128/256/512-bit families.
//! * [`SimdVector`] / [`SimdOps`]: traits abstracting over the backend.
//! * [`width`]: native vector width constants for the compile target.
//! * [`detect`]: compile-time and (with `std`) runtime CPU feature detection.
//! * [`ComplexSimd`]: split (SoA) complex vectors shared by the complex,
//!   mul, butterfly and dot crates.
//!
//! ## Backend selection
//!
//! The stable backend is `tpt-simd-vector`. The `nightly` feature is
//! reserved for a `core::simd` backend and is currently a no-op; see
//! `docs/adr/0001-backend-and-dispatch.md`.
#![no_std]
#![forbid(unsafe_code)]

#[cfg(any(feature = "std", test))]
extern crate std;

pub mod aliases;
mod complex;
pub mod detect;
mod traits;
pub mod width;

pub use aliases::*;
pub use complex::ComplexSimd;
pub use tpt_simd_vector::{LaneCast, Simd, SimdElement, SimdFloat, SimdInt, SimdMask};
pub use traits::{DefaultBackend, SimdOps, SimdVector};

#[cfg(test)]
mod tests;
