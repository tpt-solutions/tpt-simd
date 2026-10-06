//! The sealed scalar trait implemented for `f32` and `f64`.

use core::ops::{Add, Mul, Sub};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Floating-point element type of the kernels (`f32` or `f64`).
///
/// Sealed: it only exists so the kernels are written once. Arithmetic is
/// plain `+ - *` (no fused multiply-add), so results do not depend on the
/// target features.
pub trait Real:
    Copy + PartialEq + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + sealed::Sealed
{
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}
impl Real for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}
