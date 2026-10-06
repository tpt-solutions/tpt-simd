//! SIMD shift and rotate variants on 8 `i32` lanes.
//!
//! Implemented (uniform amount, `u32`):
//! * [`shift_left_i32`], [`shift_right_logical_i32`],
//!   [`shift_right_arithmetic_i32`], [`rotate_left_i32`],
//!   [`shift_with_rounding_i32`].
//!
//! Per-lane variable amounts (`Simd<u32, 8>`): the same names with a `_var`
//! suffix, e.g. [`shift_left_var_i32`].
//!
//! ## Out-of-range shift policy (amount `>= 32`)
//!
//! The amount is an unsigned `u32`, so there are no negative amounts. For
//! amounts `>= 32` (ADR 0002, and identical to AVX2 `vpsllvd`/`vpsrlvd`/
//! `vpsravd`, so the fast path needs no fix-up):
//!
//! | operation                    | amount `>= 32`                         |
//! |------------------------------|----------------------------------------|
//! | `shift_left`                 | `0`                                    |
//! | `shift_right_logical`        | `0`                                    |
//! | `shift_right_arithmetic`     | sign fill (`0` or `-1`)                |
//! | `shift_with_rounding`        | `0` (the value is below half an ulp)   |
//! | `rotate_left`                | amount taken **modulo 32**             |
//!
//! No function panics and none has undefined behaviour for any amount.
#![no_std]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

use tpt_simd_core::Simd;

/// True when the AVX2 variable-shift fast path is compiled in.
pub const SHIFT_VAR_USES_INTRINSICS: bool = cfg!(all(
    not(feature = "scalar-only"),
    target_arch = "x86_64",
    target_feature = "avx2"
));

#[inline(always)]
fn shl(x: i32, n: u32) -> i32 {
    if n >= 32 { 0 } else { x << n }
}

#[inline(always)]
fn shr_logical(x: i32, n: u32) -> i32 {
    if n >= 32 { 0 } else { ((x as u32) >> n) as i32 }
}

#[inline(always)]
fn shr_arith(x: i32, n: u32) -> i32 {
    x >> n.min(31)
}

#[inline(always)]
fn shr_round(x: i32, n: u32) -> i32 {
    match n {
        0 => x,
        1..=31 => ((i64::from(x) + (1i64 << (n - 1))) >> n) as i32,
        _ => 0,
    }
}

/// Shift every lane left by `amount` bits. `amount >= 32` gives 0.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_shift::shift_left_i32;
/// assert_eq!(shift_left_i32(I32x8::splat(3), 4), I32x8::splat(48));
/// assert_eq!(shift_left_i32(I32x8::splat(3), 32), I32x8::splat(0));
/// ```
#[inline]
pub fn shift_left_i32(a: Simd<i32, 8>, amount: u32) -> Simd<i32, 8> {
    a.map(|x| shl(x, amount))
}

/// Logical (zero-filling) shift right of every lane. `amount >= 32` gives 0.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_shift::shift_right_logical_i32;
/// assert_eq!(shift_right_logical_i32(I32x8::splat(-1), 28), I32x8::splat(15));
/// assert_eq!(shift_right_logical_i32(I32x8::splat(-1), 40), I32x8::splat(0));
/// ```
#[inline]
pub fn shift_right_logical_i32(a: Simd<i32, 8>, amount: u32) -> Simd<i32, 8> {
    a.map(|x| shr_logical(x, amount))
}

/// Arithmetic (sign-extending) shift right of every lane. `amount >= 32`
/// gives the sign fill (`0` for non-negative lanes, `-1` for negative).
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_shift::shift_right_arithmetic_i32;
/// assert_eq!(shift_right_arithmetic_i32(I32x8::splat(-16), 2), I32x8::splat(-4));
/// assert_eq!(shift_right_arithmetic_i32(I32x8::splat(-16), 99), I32x8::splat(-1));
/// assert_eq!(shift_right_arithmetic_i32(I32x8::splat(16), 99), I32x8::splat(0));
/// ```
#[inline]
pub fn shift_right_arithmetic_i32(a: Simd<i32, 8>, amount: u32) -> Simd<i32, 8> {
    a.map(|x| shr_arith(x, amount))
}

/// Rotate every lane left by `amount % 32` bits (rotation is periodic, so
/// there is no out-of-range case; `amount = 32` is the identity).
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_shift::rotate_left_i32;
/// assert_eq!(rotate_left_i32(I32x8::splat(i32::MIN), 1), I32x8::splat(1));
/// assert_eq!(rotate_left_i32(I32x8::splat(0x1234), 32), I32x8::splat(0x1234));
/// ```
#[inline]
pub fn rotate_left_i32(a: Simd<i32, 8>, amount: u32) -> Simd<i32, 8> {
    a.map(|x| x.rotate_left(amount % 32))
}

/// Arithmetic shift right by `amount` with **round-half-up** (ties toward
/// +infinity): `(x + (1 << (amount-1))) >> amount`, evaluated in 64 bits so
/// it never overflows. `amount == 0` is the identity; `amount >= 32` gives 0
/// (every `i32` divided by `2^32` or more rounds to 0).
///
/// This is the usual fixed-point "descale" step.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::I32x8;
/// use tpt_simd_shift::shift_with_rounding_i32;
/// let r = shift_with_rounding_i32(I32x8::from_array([5, 6, 7, -5, -6, -7, i32::MAX, i32::MIN]), 2);
/// assert_eq!(r.to_array(), [1, 2, 2, -1, -1, -2, 1 << 29, i32::MIN >> 2]);
/// ```
#[inline]
pub fn shift_with_rounding_i32(a: Simd<i32, 8>, amount: u32) -> Simd<i32, 8> {
    a.map(|x| shr_round(x, amount))
}

#[cfg(all(
    not(feature = "scalar-only"),
    target_arch = "x86_64",
    target_feature = "avx2"
))]
macro_rules! var_avx2 {
    ($a:expr, $n:expr, $op:ident) => {{
        use core::arch::x86_64::*;
        let (a, n) = ($a.to_array(), $n.to_array());
        let mut out = [0i32; 8];
        // SAFETY: only compiled when `avx2` is enabled at compile time.
        // Pointers come from 32-byte arrays and are accessed unaligned.
        unsafe {
            let va = _mm256_loadu_si256(a.as_ptr().cast());
            let vn = _mm256_loadu_si256(n.as_ptr().cast());
            _mm256_storeu_si256(out.as_mut_ptr().cast(), $op(va, vn));
        }
        Simd::from_array(out)
    }};
}

/// Per-lane left shift: lane `i` is shifted by `amounts[i]`. Amounts `>= 32`
/// give 0 (matches `vpsllvd`).
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::{I32x8, U32x8};
/// use tpt_simd_shift::shift_left_var_i32;
/// let r = shift_left_var_i32(I32x8::splat(1), U32x8::from_array([0, 1, 2, 31, 32, 33, 100, u32::MAX]));
/// assert_eq!(r.to_array(), [1, 2, 4, i32::MIN, 0, 0, 0, 0]);
/// ```
#[inline]
pub fn shift_left_var_i32(a: Simd<i32, 8>, amounts: Simd<u32, 8>) -> Simd<i32, 8> {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    ))]
    {
        var_avx2!(a, amounts, _mm256_sllv_epi32)
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    )))]
    {
        shift_left_var_i32_portable(a, amounts)
    }
}

/// Portable reference for [`shift_left_var_i32`].
#[inline]
pub fn shift_left_var_i32_portable(a: Simd<i32, 8>, amounts: Simd<u32, 8>) -> Simd<i32, 8> {
    Simd::from_fn(|i| shl(a[i], amounts[i]))
}

/// Per-lane logical right shift. Amounts `>= 32` give 0 (matches `vpsrlvd`).
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::{I32x8, U32x8};
/// use tpt_simd_shift::shift_right_logical_var_i32;
/// let r = shift_right_logical_var_i32(I32x8::splat(-1), U32x8::from_array([0, 1, 31, 32, 33, 0, 0, 0]));
/// assert_eq!(r.to_array(), [-1, i32::MAX, 1, 0, 0, -1, -1, -1]);
/// ```
#[inline]
pub fn shift_right_logical_var_i32(a: Simd<i32, 8>, amounts: Simd<u32, 8>) -> Simd<i32, 8> {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    ))]
    {
        var_avx2!(a, amounts, _mm256_srlv_epi32)
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    )))]
    {
        shift_right_logical_var_i32_portable(a, amounts)
    }
}

/// Portable reference for [`shift_right_logical_var_i32`].
#[inline]
pub fn shift_right_logical_var_i32_portable(
    a: Simd<i32, 8>,
    amounts: Simd<u32, 8>,
) -> Simd<i32, 8> {
    Simd::from_fn(|i| shr_logical(a[i], amounts[i]))
}

/// Per-lane arithmetic right shift. Amounts `>= 32` give the sign fill
/// (matches `vpsravd`).
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::{I32x8, U32x8};
/// use tpt_simd_shift::shift_right_arithmetic_var_i32;
/// let r = shift_right_arithmetic_var_i32(I32x8::splat(-8), U32x8::from_array([0, 1, 3, 4, 31, 32, 1000, 2]));
/// assert_eq!(r.to_array(), [-8, -4, -1, -1, -1, -1, -1, -2]);
/// ```
#[inline]
pub fn shift_right_arithmetic_var_i32(a: Simd<i32, 8>, amounts: Simd<u32, 8>) -> Simd<i32, 8> {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    ))]
    {
        var_avx2!(a, amounts, _mm256_srav_epi32)
    }
    #[cfg(not(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "avx2"
    )))]
    {
        shift_right_arithmetic_var_i32_portable(a, amounts)
    }
}

/// Portable reference for [`shift_right_arithmetic_var_i32`].
#[inline]
pub fn shift_right_arithmetic_var_i32_portable(
    a: Simd<i32, 8>,
    amounts: Simd<u32, 8>,
) -> Simd<i32, 8> {
    Simd::from_fn(|i| shr_arith(a[i], amounts[i]))
}

/// Per-lane rotate left by `amounts[i] % 32`.
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::{I32x8, U32x8};
/// use tpt_simd_shift::rotate_left_var_i32;
/// let r = rotate_left_var_i32(I32x8::splat(i32::MIN), U32x8::from_array([0, 1, 31, 32, 33, 0, 0, 0]));
/// assert_eq!(r.to_array(), [i32::MIN, 1, 1 << 30, i32::MIN, 1, i32::MIN, i32::MIN, i32::MIN]);
/// ```
#[inline]
pub fn rotate_left_var_i32(a: Simd<i32, 8>, amounts: Simd<u32, 8>) -> Simd<i32, 8> {
    Simd::from_fn(|i| a[i].rotate_left(amounts[i] % 32))
}

/// Per-lane [`shift_with_rounding_i32`] (round-half-up arithmetic right
/// shift; amount 0 is the identity, `>= 32` gives 0).
///
/// # Panics
/// Never.
///
/// ```
/// use tpt_simd_core::{I32x8, U32x8};
/// use tpt_simd_shift::shift_with_rounding_var_i32;
/// let r = shift_with_rounding_var_i32(I32x8::splat(7), U32x8::from_array([0, 1, 2, 3, 31, 32, 40, 4]));
/// assert_eq!(r.to_array(), [7, 4, 2, 1, 0, 0, 0, 0]);
/// ```
#[inline]
pub fn shift_with_rounding_var_i32(a: Simd<i32, 8>, amounts: Simd<u32, 8>) -> Simd<i32, 8> {
    Simd::from_fn(|i| shr_round(a[i], amounts[i]))
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use proptest::prelude::*;
    use tpt_simd_testutil::i32_edgy;

    fn arr(v: std::vec::Vec<i32>) -> Simd<i32, 8> {
        Simd::from_slice(&v)
    }

    fn amts() -> impl Strategy<Value = u32> {
        prop_oneof![0u32..=34, any::<u32>()]
    }

    #[test]
    fn out_of_range_policy() {
        let neg = Simd::<i32, 8>::splat(-5);
        let pos = Simd::<i32, 8>::splat(5);
        for n in [32u32, 33, 63, 64, 1000, u32::MAX] {
            assert_eq!(shift_left_i32(pos, n), Simd::splat(0));
            assert_eq!(shift_right_logical_i32(neg, n), Simd::splat(0));
            assert_eq!(shift_right_arithmetic_i32(neg, n), Simd::splat(-1));
            assert_eq!(shift_right_arithmetic_i32(pos, n), Simd::splat(0));
            assert_eq!(shift_with_rounding_i32(neg, n), Simd::splat(0));
            assert_eq!(rotate_left_i32(pos, n), rotate_left_i32(pos, n % 32));
        }
        // 31 is still in range.
        assert_eq!(shift_left_i32(Simd::splat(1), 31), Simd::splat(i32::MIN));
        assert_eq!(
            shift_right_arithmetic_i32(Simd::splat(i32::MIN), 31),
            Simd::splat(-1)
        );
        assert_eq!(
            shift_right_logical_i32(Simd::splat(i32::MIN), 31),
            Simd::splat(1)
        );
    }

    #[test]
    fn rounding_ties() {
        // Ties go toward +infinity.
        let r = shift_with_rounding_i32(arr(std::vec![1, 3, -1, -3, 2, -2, 0, 4]), 1);
        assert_eq!(r.to_array(), [1, 2, 0, -1, 1, -1, 0, 2]);
        let r = shift_with_rounding_i32(Simd::splat(i32::MAX), 1);
        assert_eq!(r, Simd::splat(1 << 30));
        assert_eq!(
            shift_with_rounding_i32(Simd::splat(i32::MIN), 31),
            Simd::splat(-1)
        );
        assert_eq!(
            shift_with_rounding_i32(Simd::splat(i32::MIN), 32),
            Simd::splat(0)
        );
        assert_eq!(shift_with_rounding_i32(Simd::splat(77), 0), Simd::splat(77));
    }

    proptest! {
        #[test]
        fn uniform_matches_scalar(v in prop::collection::vec(i32_edgy(), 8), n in amts()) {
            let a = arr(v.clone());
            let (sl, sr, sa, rl, rr) = (
                shift_left_i32(a, n),
                shift_right_logical_i32(a, n),
                shift_right_arithmetic_i32(a, n),
                rotate_left_i32(a, n),
                shift_with_rounding_i32(a, n),
            );
            for l in 0..8 {
                let x = v[l];
                let wide = i128::from(x);
                prop_assert_eq!(sl[l], if n >= 32 { 0 } else { (wide << n) as i32 });
                prop_assert_eq!(sr[l], if n >= 32 { 0 } else { ((x as u32 as u64) >> n) as i32 });
                prop_assert_eq!(sa[l], if n >= 32 { -i32::from(wide < 0) } else { (wide >> n) as i32 });
                prop_assert_eq!(rl[l], x.rotate_left(n % 32));
                let expect = if n >= 32 { 0 } else if n == 0 { x } else { ((wide + (1i128 << (n - 1))) >> n) as i32 };
                prop_assert_eq!(rr[l], expect);
            }
        }

        #[test]
        fn variable_matches_uniform(
            v in prop::collection::vec(i32_edgy(), 8),
            ns in prop::collection::vec(amts(), 8),
        ) {
            let a = arr(v);
            let n = Simd::<u32, 8>::from_slice(&ns);
            let sl = shift_left_var_i32(a, n);
            let sr = shift_right_logical_var_i32(a, n);
            let sa = shift_right_arithmetic_var_i32(a, n);
            let rl = rotate_left_var_i32(a, n);
            let rr = shift_with_rounding_var_i32(a, n);
            prop_assert_eq!(sl, shift_left_var_i32_portable(a, n));
            prop_assert_eq!(sr, shift_right_logical_var_i32_portable(a, n));
            prop_assert_eq!(sa, shift_right_arithmetic_var_i32_portable(a, n));
            for l in 0..8 {
                let one = Simd::<i32, 8>::splat(a[l]);
                prop_assert_eq!(sl[l], shift_left_i32(one, ns[l])[0]);
                prop_assert_eq!(sr[l], shift_right_logical_i32(one, ns[l])[0]);
                prop_assert_eq!(sa[l], shift_right_arithmetic_i32(one, ns[l])[0]);
                prop_assert_eq!(rl[l], rotate_left_i32(one, ns[l])[0]);
                prop_assert_eq!(rr[l], shift_with_rounding_i32(one, ns[l])[0]);
            }
        }
    }
}
