//! Umbrella crate re-exporting all tpt-simd crates.
//!
//! Each member crate is available as a module (`tpt_simd::dot`, ...); the
//! core types are re-exported at the root.
#![no_std]
#![forbid(unsafe_code)]

pub use tpt_simd_core::*;

#[doc(inline)]
pub use tpt_simd_aligned as aligned;
#[doc(inline)]
pub use tpt_simd_blas as blas;
#[doc(inline)]
pub use tpt_simd_blend as blend;
#[doc(inline)]
pub use tpt_simd_butterfly as butterfly;
#[doc(inline)]
pub use tpt_simd_compare as compare;
#[doc(inline)]
pub use tpt_simd_complex as complex;
#[doc(inline)]
pub use tpt_simd_convolve as convolve;
#[doc(inline)]
pub use tpt_simd_dot as dot;
#[doc(inline)]
pub use tpt_simd_fixed as fixed;
#[doc(inline)]
pub use tpt_simd_gather as gather;
#[doc(inline)]
pub use tpt_simd_horizontal as horizontal;
#[doc(inline)]
pub use tpt_simd_interpolate as interpolate;
#[doc(inline)]
pub use tpt_simd_math as math;
#[doc(inline)]
pub use tpt_simd_matrix as matrix;
#[doc(inline)]
pub use tpt_simd_mul as mul;
#[doc(inline)]
pub use tpt_simd_permute as permute;
#[doc(inline)]
pub use tpt_simd_reduce as reduce;
#[doc(inline)]
pub use tpt_simd_rng as rng;
#[doc(inline)]
pub use tpt_simd_rounding as rounding;
#[doc(inline)]
pub use tpt_simd_saturate as saturate;
#[doc(inline)]
pub use tpt_simd_scatter as scatter;
#[doc(inline)]
pub use tpt_simd_select as select;
#[doc(inline)]
pub use tpt_simd_shift as shift;
#[doc(inline)]
pub use tpt_simd_sparse as sparse;
#[doc(inline)]
pub use tpt_simd_window as window;
