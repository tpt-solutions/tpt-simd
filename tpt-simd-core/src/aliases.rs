//! Type aliases for common vector shapes: `<Type><lane count>`, e.g. [`F32x8`].
//!
//! 128-bit, 256-bit and 512-bit families are provided for every lane type.

use tpt_simd_vector::{Simd, SimdMask};

macro_rules! aliases {
    ($($name:ident = $t:ty, $n:expr;)*) => {$(
        #[doc = concat!("`Simd<", stringify!($t), ", ", stringify!($n), ">`")]
        pub type $name = Simd<$t, $n>;
    )*};
}

aliases! {
    F32x4 = f32, 4; F32x8 = f32, 8; F32x16 = f32, 16;
    F64x2 = f64, 2; F64x4 = f64, 4; F64x8 = f64, 8;
    I8x16 = i8, 16; I8x32 = i8, 32; I8x64 = i8, 64;
    I16x8 = i16, 8; I16x16 = i16, 16; I16x32 = i16, 32;
    I32x4 = i32, 4; I32x8 = i32, 8; I32x16 = i32, 16;
    I64x2 = i64, 2; I64x4 = i64, 4; I64x8 = i64, 8;
    U8x16 = u8, 16; U8x32 = u8, 32; U8x64 = u8, 64;
    U16x8 = u16, 8; U16x16 = u16, 16; U16x32 = u16, 32;
    U32x4 = u32, 4; U32x8 = u32, 8; U32x16 = u32, 16;
    U64x2 = u64, 2; U64x4 = u64, 4; U64x8 = u64, 8;
}

/// Mask for 8-lane vectors of `T`.
pub type Mask8<T> = SimdMask<T, 8>;
/// Mask for 16-lane vectors of `T`.
pub type Mask16<T> = SimdMask<T, 16>;
/// Mask for 32-lane vectors of `T`.
pub type Mask32<T> = SimdMask<T, 32>;
