//! Stable, array-backed `Simd<T, N>` and `SimdMask<T, N>` types.
//!
//! This crate is the backend every other tpt-simd crate builds on. The
//! types are plain `[T; N]` wrappers whose operations are fixed-length lane
//! loops; LLVM lowers them to SSE/AVX/NEON/RVV on the target. It works on
//! stable Rust, in `no_std`, with no `unsafe`.
//!
//! * [`Simd`]: the vector type (arithmetic, bitwise, compare, saturate, shifts, casts).
//! * [`SimdMask`]: per-lane booleans from comparisons.
//! * [`SimdElement`], [`SimdInt`], [`SimdFloat`], [`LaneCast`]: lane traits.
//!
//! Semantics (shared by every crate in the workspace):
//!
//! * Integer `+ - *` and negation **wrap**; use `sat_*` for saturation.
//! * Shifts by `>= BITS` yield `0` (or sign fill for arithmetic right shift).
//! * Float `min`/`max` ignore NaN operands; comparisons with NaN are false
//!   (except `!=`).
//! * Float math is computed with `libm`, so results are target-independent.
#![no_std]
#![forbid(unsafe_code)]

mod element;
mod mask;
mod simd;

pub use element::{LaneCast, SimdElement, SimdFloat, SimdInt};
pub use mask::SimdMask;
pub use simd::Simd;

#[cfg(test)]
extern crate std;
#[cfg(test)]
mod tests;
