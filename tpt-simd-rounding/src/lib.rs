//! SIMD rounding operations on 8 `f32` lanes.
//!
//! Implemented:
//! * [`round_f32`] — round to nearest, **ties away from zero** (like C `roundf`).
//! * [`floor_f32`], [`ceil_f32`], [`trunc_f32`] — directed rounding.
//! * [`round_ties_even_f32`] — round to nearest, **ties to even** (like `rintf`
//!   in the default rounding mode, and `roundps`/`vrndn`).
//! * [`round_to_nearest_even_i32`] — ties-to-even, converted to `i32`.
//! * [`round_with_bias_i32`] — `floor(x + bias)` converted to `i32`.
//!
//! ## Tie-breaking summary
//!
//! | function                      | tie rule         | `0.5` | `1.5` | `2.5` | `-0.5` | `-2.5` |
//! |-------------------------------|------------------|-------|-------|-------|--------|--------|
//! | [`round_f32`]                 | away from zero   | 1     | 2     | 3     | -1     | -3     |
//! | [`round_ties_even_f32`]       | to even          | 0     | 2     | 2     | -0     | -2     |
//! | [`round_to_nearest_even_i32`] | to even          | 0     | 2     | 2     | 0      | -2     |
//! | [`round_with_bias_i32`] (0.5) | toward +infinity | 1     | 2     | 3     | 0      | -2     |
//!
//! ## Special values
//!
//! Float-returning functions preserve NaN, infinities and the sign of zero
//! (`ceil(-0.3) == -0.0`). Integer-returning functions saturate out-of-range
//! values to `i32::MIN`/`i32::MAX` and map NaN to 0 (Rust `as` semantics).
//!
//! ## Backends
//!
//! On x86-64 with AVX enabled at compile time (`-C target-feature=+avx2` or
//! `target-cpu=native`), `floor`/`ceil`/`trunc`/ties-even use `vroundps`.
//! Everywhere else (including plain SSE2, which has no `roundps`; that
//! arrived with SSE4.1) a branch-free bit-manipulation fallback is used that
//! LLVM vectorises with SSE2 instructions only. It needs no `libm`; `libm`
//! is only used by the tests as the reference. NEON `vrnd*` is not
//! hand-written: the fallback is left to LLVM. The `scalar-only` feature
//! forces the fallback.
#![no_std]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

use tpt_simd_core::Simd;

#[allow(dead_code)]
const SIGN: u32 = 0x8000_0000;
#[allow(dead_code)]
const TWO23: f32 = 8_388_608.0;

/// Truncate toward zero using only integer bit tricks (no `roundps`).
#[inline(always)]
#[allow(dead_code)] // unused when the AVX path is compiled in
fn trunc_lane(x: f32) -> f32 {
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    if exp >= 23 {
        // Already integral, or inf/NaN.
        x
    } else if exp < 0 {
        f32::from_bits(bits & SIGN)
    } else {
        f32::from_bits(bits & !(0x007f_ffffu32 >> exp))
    }
}

#[inline(always)]
#[allow(dead_code)] // unused when the AVX path is compiled in
fn floor_lane(x: f32) -> f32 {
    let t = trunc_lane(x);
    if x < t { t - 1.0 } else { t }
}

#[inline(always)]
#[allow(dead_code)] // unused when the AVX path is compiled in
fn ceil_lane(x: f32) -> f32 {
    let t = trunc_lane(x);
    if x > t { t + 1.0 } else { t }
}

#[inline(always)]
fn round_away_lane(x: f32) -> f32 {
    let t = trunc_lane(x);
    // `x - t` is exact for |x| < 2^23; for larger |x|, t == x and the
    // difference is 0.
    if (x - t).abs() >= 0.5 {
        t + 1.0f32.copysign(x)
    } else {
        t
    }
}

#[inline(always)]
#[allow(dead_code)] // unused when the AVX path is compiled in
fn rint_lane(x: f32) -> f32 {
    let a = x.abs();
    if a < TWO23 {
        ((a + TWO23) - TWO23).copysign(x)
    } else {
        x
    }
}

#[cfg(all(
    not(feature = "scalar-only"),
    target_arch = "x86_64",
    target_feature = "avx"
))]
#[inline]
fn round_avx<const MODE: i32>(a: Simd<f32, 8>) -> Simd<f32, 8> {
    use core::arch::x86_64::*;
    let src = a.to_array();
    let mut out = [0.0f32; 8];
    // SAFETY: only compiled when `avx` is enabled at compile time; the
    // pointers come from `[f32; 8]` arrays (32 bytes), accessed unaligned.
    unsafe {
        let v = _mm256_loadu_ps(src.as_ptr());
        _mm256_storeu_ps(out.as_mut_ptr(), _mm256_round_ps::<MODE>(v));
    }
    Simd::from_array(out)
}

/// True when the `vroundps` fast path is compiled in.
pub const ROUNDING_USES_INTRINSICS: bool = cfg!(all(
    not(feature = "scalar-only"),
    target_arch = "x86_64",
    target_feature = "avx"
));

/// Round each lane to the nearest integer, **ties away from zero**
/// (`0.5 -> 1`, `-0.5 -> -1`, `2.5 -> 3`).
///
/// Use [`round_ties_even_f32`] for banker's rounding. NaN, infinities and
/// the sign of zero are preserved.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::F32x8;
/// use tpt_simd_rounding::round_f32;
/// let r = round_f32(F32x8::from_array([0.5, 1.5, 2.5, -0.5, -2.5, 0.49999997, 2.4, -2.6]));
/// assert_eq!(r.to_array(), [1.0, 2.0, 3.0, -1.0, -3.0, 0.0, 2.0, -3.0]);
/// ```
#[inline]
pub fn round_f32(a: Simd<f32, 8>) -> Simd<f32, 8> {
    a.map(round_away_lane)
}

/// Round each lane toward negative infinity.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::F32x8;
/// use tpt_simd_rounding::floor_f32;
/// let r = floor_f32(F32x8::from_array([1.5, -1.5, 0.2, -0.2, 3.0, -3.0, 1e10, -1e10]));
/// assert_eq!(r.to_array(), [1.0, -2.0, 0.0, -1.0, 3.0, -3.0, 1e10, -1e10]);
/// ```
#[inline]
pub fn floor_f32(a: Simd<f32, 8>) -> Simd<f32, 8> {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx"
    ))]
    {
        round_avx::<0x09>(a) // _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx"
    )))]
    {
        a.map(floor_lane)
    }
}

/// Round each lane toward positive infinity.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::F32x8;
/// use tpt_simd_rounding::ceil_f32;
/// let r = ceil_f32(F32x8::from_array([1.5, -1.5, 0.2, -0.2, 3.0, -3.0, 0.0, 7.0001]));
/// assert_eq!(r.to_array(), [2.0, -1.0, 1.0, -0.0, 3.0, -3.0, 0.0, 8.0]);
/// ```
#[inline]
pub fn ceil_f32(a: Simd<f32, 8>) -> Simd<f32, 8> {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx"
    ))]
    {
        round_avx::<0x0A>(a) // _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx"
    )))]
    {
        a.map(ceil_lane)
    }
}

/// Round each lane toward zero.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::F32x8;
/// use tpt_simd_rounding::trunc_f32;
/// let r = trunc_f32(F32x8::from_array([1.9, -1.9, 0.9, -0.9, 3.0, -3.0, 0.0, 8.5]));
/// assert_eq!(r.to_array(), [1.0, -1.0, 0.0, -0.0, 3.0, -3.0, 0.0, 8.0]);
/// ```
#[inline]
pub fn trunc_f32(a: Simd<f32, 8>) -> Simd<f32, 8> {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx"
    ))]
    {
        round_avx::<0x0B>(a) // _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx"
    )))]
    {
        a.map(trunc_lane)
    }
}

/// Round each lane to the nearest integer, **ties to even**
/// (`0.5 -> 0`, `1.5 -> 2`, `2.5 -> 2`).
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::F32x8;
/// use tpt_simd_rounding::round_ties_even_f32;
/// let r = round_ties_even_f32(F32x8::from_array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.6, -2.4]));
/// assert_eq!(r.to_array(), [0.0, 2.0, 2.0, -0.0, -2.0, -2.0, 3.0, -2.0]);
/// ```
#[inline]
pub fn round_ties_even_f32(a: Simd<f32, 8>) -> Simd<f32, 8> {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx"
    ))]
    {
        round_avx::<0x08>(a) // _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx"
    )))]
    {
        a.map(rint_lane)
    }
}

/// Round each lane to the nearest integer with **ties to even** and convert
/// to `i32`. Out-of-range values saturate, NaN becomes 0.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::F32x8;
/// use tpt_simd_rounding::round_to_nearest_even_i32;
/// let r = round_to_nearest_even_i32(F32x8::from_array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 1e20, f32::NAN]));
/// assert_eq!(r.to_array(), [0, 2, 2, 0, -2, -2, i32::MAX, 0]);
/// ```
#[inline]
pub fn round_to_nearest_even_i32(a: Simd<f32, 8>) -> Simd<i32, 8> {
    round_ties_even_f32(a).map(|x| x as i32)
}

/// Convert `floor(x + bias)` to `i32` for each lane.
///
/// `bias = 0.5` is round-half-up (ties toward +infinity) and `0.0` is
/// floor. The addition is a single `f32` operation, so the result follows
/// `f32` arithmetic exactly (e.g. `0.49999997 + 0.5` rounds to `1.0`, giving
/// 1). Out-of-range results saturate, NaN becomes 0.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::F32x8;
/// use tpt_simd_rounding::round_with_bias_i32;
/// let x = F32x8::from_array([0.5, 1.5, -0.5, -1.5, 2.4, -2.4, 0.0, 7.0]);
/// assert_eq!(round_with_bias_i32(x, 0.5).to_array(), [1, 2, 0, -1, 2, -2, 0, 7]);
/// assert_eq!(round_with_bias_i32(x, 0.0).to_array(), [0, 1, -1, -2, 2, -3, 0, 7]);
/// ```
#[inline]
pub fn round_with_bias_i32(a: Simd<f32, 8>, bias: f32) -> Simd<i32, 8> {
    floor_f32(a + Simd::splat(bias)).map(|x| x as i32)
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use proptest::prelude::*;
    use tpt_simd_testutil::f32_with_specials;

    fn same(x: f32, y: f32) -> bool {
        x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
    }

    #[test]
    fn edge_cases_match_libm() {
        for set in [
            [
                0.5,
                -0.5,
                8_388_607.5,
                -8_388_607.5,
                0.49999997,
                f32::INFINITY,
                -0.0,
                f32::NAN,
            ],
            [
                1.5,
                2.5,
                -1.5,
                -2.5,
                8_388_608.0,
                1e30,
                f32::NEG_INFINITY,
                1.0e-40,
            ],
            [-0.3, 0.3, -0.7, 0.7, 16_777_216.0, -16_777_216.0, 3.0, -3.0],
        ] {
            let v = Simd::from_array(set);
            for l in 0..8 {
                assert!(
                    same(floor_f32(v)[l], libm::floorf(set[l])),
                    "floor {}",
                    set[l]
                );
                assert!(same(ceil_f32(v)[l], libm::ceilf(set[l])), "ceil {}", set[l]);
                assert!(
                    same(trunc_f32(v)[l], libm::truncf(set[l])),
                    "trunc {}",
                    set[l]
                );
                assert!(
                    same(round_f32(v)[l], libm::roundf(set[l])),
                    "round {}",
                    set[l]
                );
                assert!(
                    same(round_ties_even_f32(v)[l], libm::rintf(set[l])),
                    "rint {}",
                    set[l]
                );
            }
        }
    }

    #[test]
    fn tie_breaking() {
        let v = Simd::from_array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5]);
        assert_eq!(
            round_f32(v).to_array(),
            [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0]
        );
        assert_eq!(
            round_to_nearest_even_i32(v).to_array(),
            [0, 2, 2, 4, 0, -2, -2, -4]
        );
        assert_eq!(
            round_with_bias_i32(v, 0.5).to_array(),
            [1, 2, 3, 4, 0, -1, -2, -3]
        );
    }

    #[test]
    fn int_saturation() {
        let v = Simd::from_array([
            3e9,
            -3e9,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.0,
            1.0,
            -1.0,
        ]);
        assert_eq!(
            round_to_nearest_even_i32(v).to_array(),
            [i32::MAX, i32::MIN, i32::MAX, i32::MIN, 0, 0, 1, -1]
        );
    }

    proptest! {
        #[test]
        fn matches_libm(v in prop::collection::vec(f32_with_specials(), 8)) {
            let a: [f32; 8] = v.try_into().unwrap();
            let s = Simd::from_array(a);
            for l in 0..8 {
                prop_assert!(same(floor_f32(s)[l], libm::floorf(a[l])));
                prop_assert!(same(ceil_f32(s)[l], libm::ceilf(a[l])));
                prop_assert!(same(trunc_f32(s)[l], libm::truncf(a[l])));
                prop_assert!(same(round_f32(s)[l], libm::roundf(a[l])));
                prop_assert!(same(round_ties_even_f32(s)[l], libm::rintf(a[l])));
                prop_assert_eq!(round_to_nearest_even_i32(s)[l], libm::rintf(a[l]) as i32);
                prop_assert_eq!(round_with_bias_i32(s, 0.5)[l], libm::floorf(a[l] + 0.5) as i32);
            }
        }

        #[test]
        fn matches_libm_all_bit_patterns(bits in prop::collection::vec(any::<u32>(), 8)) {
            let a: [f32; 8] = core::array::from_fn(|i| f32::from_bits(bits[i]));
            let s = Simd::from_array(a);
            for l in 0..8 {
                prop_assert!(same(floor_f32(s)[l], libm::floorf(a[l])));
                prop_assert!(same(ceil_f32(s)[l], libm::ceilf(a[l])));
                prop_assert!(same(trunc_f32(s)[l], libm::truncf(a[l])));
                prop_assert!(same(round_f32(s)[l], libm::roundf(a[l])));
                prop_assert!(same(round_ties_even_f32(s)[l], libm::rintf(a[l])));
            }
        }
    }
}
