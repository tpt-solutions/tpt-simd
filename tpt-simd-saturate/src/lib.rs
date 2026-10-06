//! SIMD saturating arithmetic and saturating pack operations.
//!
//! Everything here clamps instead of wrapping. The implementations are
//! fixed-length lane loops over [`Simd`] which LLVM lowers to `paddsw`,
//! `psubusb`, `packsswb`, ... on x86 and to `sqadd`/`sqxtn` on aarch64, so the
//! portable code *is* the fast path (see the crate benchmark). Lane order is
//! always the natural order: lane `i` of the result comes from lane `i` of the
//! input (no 128-bit-lane interleaving as with raw `_mm256_packs_epi16`).
//!
//! ```
//! use tpt_simd_core::{I16x16, I8x16};
//! use tpt_simd_saturate::{pack_i16_to_i8, saturating_add_i16};
//!
//! let a = I16x16::splat(i16::MAX);
//! assert_eq!(saturating_add_i16(a, I16x16::splat(1)), a);
//!
//! let v = I16x16::from_fn(|i| (i as i16 - 8) * 100);
//! let p: I8x16 = pack_i16_to_i8(v);
//! assert_eq!(p[0], -128); // -800 clamps
//! assert_eq!(p[8], 0);
//! assert_eq!(p[15], 127); // 700 clamps
//! ```
#![no_std]
#![forbid(unsafe_code)]

#[cfg(any(feature = "std", test))]
extern crate std;

use tpt_simd_core::Simd;

// ---- saturating add / sub -------------------------------------------------

macro_rules! sat_binop {
    ($(#[$m:meta])* $name:ident, $method:ident, $t:ty, $n:literal, $ex_ty:literal, $ex:literal) => {
        $(#[$m])*
        ///
        /// # Performance
        /// One instruction on SSE2/AVX2/NEON (`padds*`/`psubs*`/`paddus*`/
        /// `psubus*`); auto-vectorised from the lane loop.
        ///
        /// # Panics
        /// Never.
        ///
        /// # Example
        /// ```
        #[doc = concat!("use tpt_simd_core::", $ex_ty, ";")]
        #[doc = concat!("use tpt_simd_saturate::", stringify!($name), ";")]
        #[doc = $ex]
        /// ```
        #[inline]
        pub fn $name(a: Simd<$t, $n>, b: Simd<$t, $n>) -> Simd<$t, $n> {
            a.$method(b)
        }
    };
}

sat_binop!(
    /// Lane-wise saturating `a + b` on 16 `i16` lanes (clamps to `i16::MIN..=i16::MAX`).
    saturating_add_i16, sat_add, i16, 16, "I16x16",
    "let r = saturating_add_i16(I16x16::splat(30000), I16x16::splat(10000));\nassert_eq!(r, I16x16::splat(i16::MAX));"
);
sat_binop!(
    /// Lane-wise saturating `a - b` on 16 `i16` lanes.
    saturating_sub_i16, sat_sub, i16, 16, "I16x16",
    "let r = saturating_sub_i16(I16x16::splat(-30000), I16x16::splat(10000));\nassert_eq!(r, I16x16::splat(i16::MIN));"
);
sat_binop!(
    /// Lane-wise saturating `a + b` on 32 `i8` lanes.
    saturating_add_i8, sat_add, i8, 32, "I8x32",
    "let r = saturating_add_i8(I8x32::splat(100), I8x32::splat(100));\nassert_eq!(r, I8x32::splat(127));"
);
sat_binop!(
    /// Lane-wise saturating `a - b` on 32 `i8` lanes.
    saturating_sub_i8, sat_sub, i8, 32, "I8x32",
    "let r = saturating_sub_i8(I8x32::splat(-100), I8x32::splat(100));\nassert_eq!(r, I8x32::splat(-128));"
);
sat_binop!(
    /// Lane-wise saturating `a + b` on 32 `u8` lanes (clamps at 255).
    saturating_add_u8, sat_add, u8, 32, "U8x32",
    "let r = saturating_add_u8(U8x32::splat(200), U8x32::splat(100));\nassert_eq!(r, U8x32::splat(255));"
);
sat_binop!(
    /// Lane-wise saturating `a - b` on 32 `u8` lanes (clamps at 0).
    saturating_sub_u8, sat_sub, u8, 32, "U8x32",
    "let r = saturating_sub_u8(U8x32::splat(5), U8x32::splat(10));\nassert_eq!(r, U8x32::splat(0));"
);
sat_binop!(
    /// Lane-wise saturating `a + b` on 16 `u16` lanes (clamps at 65535).
    saturating_add_u16, sat_add, u16, 16, "U16x16",
    "let r = saturating_add_u16(U16x16::splat(60000), U16x16::splat(10000));\nassert_eq!(r, U16x16::splat(u16::MAX));"
);
sat_binop!(
    /// Lane-wise saturating `a - b` on 16 `u16` lanes (clamps at 0).
    saturating_sub_u16, sat_sub, u16, 16, "U16x16",
    "let r = saturating_sub_u16(U16x16::splat(5), U16x16::splat(10));\nassert_eq!(r, U16x16::splat(0));"
);

/// Lane-wise saturating product of 16 `i16` lanes: the exact 32-bit product
/// `a * b` clamped to `i16::MIN..=i16::MAX`.
///
/// This is **not** the high/low half product (`pmulhw`/`pmullw`); it is a
/// single clamped result per lane.
///
/// # Performance
/// Auto-vectorises to `pmullw` + `pmulhw` + unpack + pack (or a widening
/// multiply and saturating narrow on NEON).
///
/// # Panics
/// Never.
///
/// # Example
/// ```
/// use tpt_simd_core::I16x16;
/// use tpt_simd_saturate::saturating_mul_i16;
/// let r = saturating_mul_i16(I16x16::splat(300), I16x16::splat(300));
/// assert_eq!(r, I16x16::splat(i16::MAX));
/// let r = saturating_mul_i16(I16x16::splat(-300), I16x16::splat(300));
/// assert_eq!(r, I16x16::splat(i16::MIN));
/// assert_eq!(saturating_mul_i16(I16x16::splat(-7), I16x16::splat(6)), I16x16::splat(-42));
/// ```
#[inline]
pub fn saturating_mul_i16(a: Simd<i16, 16>, b: Simd<i16, 16>) -> Simd<i16, 16> {
    Simd::from_fn(|i| (a[i] as i32 * b[i] as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16)
}

// ---- packs ----------------------------------------------------------------

/// Saturating narrow of 16 `i16` lanes to 16 `i8` lanes, **lane order
/// preserved** (`out[i] = clamp(v[i], -128, 127)`).
///
/// Unlike a raw `_mm256_packs_epi16`, which interleaves the two 128-bit
/// halves, the result is in natural order.
///
/// # Performance
/// `packsswb` (+ a lane fix-up on AVX2); auto-vectorised.
///
/// # Panics
/// Never.
///
/// # Example
/// ```
/// use tpt_simd_core::I16x16;
/// use tpt_simd_saturate::pack_i16_to_i8;
/// let v = I16x16::from_fn(|i| if i == 0 { -1000 } else if i == 1 { 1000 } else { i as i16 });
/// let p = pack_i16_to_i8(v);
/// assert_eq!(p[0], -128);
/// assert_eq!(p[1], 127);
/// assert_eq!(p[15], 15);
/// ```
#[inline]
pub fn pack_i16_to_i8(v: Simd<i16, 16>) -> Simd<i8, 16> {
    Simd::from_fn(|i| v[i].clamp(i8::MIN as i16, i8::MAX as i16) as i8)
}

/// Saturating narrow of 16 `i16` lanes to 16 `u8` lanes: clamps to `0..=255`
/// (negative values become 0). Lane order preserved.
///
/// # Performance
/// `packuswb` (+ lane fix-up on AVX2); auto-vectorised.
///
/// # Panics
/// Never.
///
/// # Example
/// ```
/// use tpt_simd_core::I16x16;
/// use tpt_simd_saturate::pack_i16_to_u8;
/// let v = I16x16::from_fn(|i| (i as i16 - 2) * 100);
/// let p = pack_i16_to_u8(v);
/// assert_eq!(p[0], 0);
/// assert_eq!(p[3], 100);
/// assert_eq!(p[15], 255);
/// ```
#[inline]
pub fn pack_i16_to_u8(v: Simd<i16, 16>) -> Simd<u8, 16> {
    Simd::from_fn(|i| v[i].clamp(0, u8::MAX as i16) as u8)
}

/// Saturating narrow of 8 `i32` lanes to 8 `i16` lanes. Lane order preserved.
///
/// # Performance
/// `packssdw` (+ lane fix-up on AVX2); auto-vectorised.
///
/// # Panics
/// Never.
///
/// # Example
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_saturate::pack_i32_to_i16;
/// let v = I32x8::from_array([0, 1, -1, 40000, -40000, i32::MAX, i32::MIN, 123]);
/// assert_eq!(
///     pack_i32_to_i16(v).to_array(),
///     [0, 1, -1, 32767, -32768, 32767, -32768, 123]
/// );
/// ```
#[inline]
pub fn pack_i32_to_i16(v: Simd<i32, 8>) -> Simd<i16, 8> {
    Simd::from_fn(|i| v[i].clamp(i16::MIN as i32, i16::MAX as i32) as i16)
}

// ---- slice versions ---------------------------------------------------------

macro_rules! pack_slice {
    ($(#[$m:meta])* $name:ident, $simd:ident, $src:ty, $dst:ty, $n:literal, $scalar:expr, $ex:literal) => {
        $(#[$m])*
        ///
        /// Processes full vectors with the SIMD routine and the tail with the
        /// same clamp in scalar form, so the result is identical for every
        /// length (including 0).
        ///
        /// # Panics
        /// If `src.len() != dst.len()`.
        ///
        /// # Example
        /// ```
        #[doc = concat!("use tpt_simd_saturate::", stringify!($name), ";")]
        #[doc = $ex]
        /// ```
        pub fn $name(src: &[$src], dst: &mut [$dst]) {
            assert_eq!(src.len(), dst.len(), "src and dst length mismatch");
            let mut s = src.chunks_exact($n);
            let mut d = dst.chunks_exact_mut($n);
            for (sc, dc) in (&mut s).zip(&mut d) {
                $simd(Simd::from_slice(sc)).copy_to_slice(dc);
            }
            let f: fn($src) -> $dst = $scalar;
            for (x, y) in s.remainder().iter().zip(d.into_remainder()) {
                *y = f(*x);
            }
        }
    };
}

pack_slice!(
    /// Saturating `i16` to `i8` over slices; see [`pack_i16_to_i8`].
    /// This is the home of the function that `tpt-simd-permute` re-exports.
    pack_i16_to_i8_slice, pack_i16_to_i8, i16, i8, 16,
    |x| x.clamp(i8::MIN as i16, i8::MAX as i16) as i8,
    "let src = [-500i16, -1, 0, 1, 500];\nlet mut dst = [0i8; 5];\npack_i16_to_i8_slice(&src, &mut dst);\nassert_eq!(dst, [-128, -1, 0, 1, 127]);"
);
pack_slice!(
    /// Saturating `i16` to `u8` (clamp to `0..=255`) over slices; see [`pack_i16_to_u8`].
    pack_i16_to_u8_slice, pack_i16_to_u8, i16, u8, 16,
    |x| x.clamp(0, u8::MAX as i16) as u8,
    "let src = [-500i16, -1, 0, 1, 500];\nlet mut dst = [9u8; 5];\npack_i16_to_u8_slice(&src, &mut dst);\nassert_eq!(dst, [0, 0, 0, 1, 255]);"
);
pack_slice!(
    /// Saturating `i32` to `i16` over slices; see [`pack_i32_to_i16`].
    pack_i32_to_i16_slice, pack_i32_to_i16, i32, i16, 8,
    |x| x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
    "let src = [-70000i32, 5, 70000];\nlet mut dst = [0i16; 3];\npack_i32_to_i16_slice(&src, &mut dst);\nassert_eq!(dst, [-32768, 5, 32767]);"
);

#[cfg(test)]
mod tests;
