//! Umbrella crate re-exporting all tpt-simd crates.
//!
//! Each member crate is available as a module (`tpt_simd::dot`, ...); the
//! core types are re-exported at the root.
#![no_std]
#![forbid(unsafe_code)]

pub use tpt_simd_core::*;

#[doc(inline)]
pub use tpt_simd_butterfly as butterfly;
#[doc(inline)]
pub use tpt_simd_complex as complex;
#[doc(inline)]
pub use tpt_simd_dot as dot;
#[doc(inline)]
pub use tpt_simd_fixed as fixed;
#[doc(inline)]
pub use tpt_simd_horizontal as horizontal;
#[doc(inline)]
pub use tpt_simd_mul as mul;
#[doc(inline)]
pub use tpt_simd_saturate as saturate;
