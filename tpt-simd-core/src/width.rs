//! Native vector width of the compile target.
//!
//! Derived from `cfg(target_feature)`, so build with
//! `RUSTFLAGS="-C target-cpu=native"` (or explicit `-C target-feature`) to
//! raise it. These are hints for choosing a lane count; every width works
//! on every target.

/// Native vector register width in bits for the compile target.
pub const NATIVE_VECTOR_BITS: usize = if cfg!(target_feature = "avx512f") {
    512
} else if cfg!(target_feature = "avx") {
    256
} else {
    128
};

/// Native vector register width in bytes.
pub const NATIVE_VECTOR_BYTES: usize = NATIVE_VECTOR_BITS / 8;

/// Lanes of `T` that fill one native vector register.
pub const fn native_lanes<T>() -> usize {
    let size = core::mem::size_of::<T>();
    if size == 0 { 1 } else { NATIVE_VECTOR_BYTES / size }
}

/// Native lane count for `f32`.
pub const NATIVE_F32_LANES: usize = native_lanes::<f32>();
/// Native lane count for `i32`.
pub const NATIVE_I32_LANES: usize = native_lanes::<i32>();
/// Native lane count for `i16`.
pub const NATIVE_I16_LANES: usize = native_lanes::<i16>();
/// Native lane count for `i8`.
pub const NATIVE_I8_LANES: usize = native_lanes::<i8>();
