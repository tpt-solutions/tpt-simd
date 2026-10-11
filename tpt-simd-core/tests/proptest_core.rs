//! Property tests for the core vector types against scalar references.
//!
//! The policies checked here are the ones in ADR 0002: integer `+ - *` wrap,
//! saturation is always explicit, float min/max ignore NaN operands, shifts of
//! 32 or more lanes produce 0 (sign fill for arithmetic right shifts), and
//! complex multiply is the unfused formula `(ac - bd) + (ad + bc)i`.

use proptest::prelude::*;
use tpt_simd_core::{ComplexSimd, Simd, SimdElement};
use tpt_simd_testutil::{f32_with_specials, i16_edgy, i32_edgy};

proptest! {
    #[test]
    fn f32_add_mul_match_scalar_bitwise(
        a in arr8(f32_with_specials()),
        b in arr8(f32_with_specials()),
    ) {
        let sum = (a + b).to_array();
        let prod = (a * b).to_array();
        let (aa, ba) = (a.to_array(), b.to_array());
        for i in 0..8 {
            prop_assert!(same_f32(sum[i], aa[i] + ba[i]));
            prop_assert!(same_f32(prod[i], aa[i] * ba[i]));
        }
    }

    #[test]
    fn i16_wrapping_ops_match_scalar(
        a in arr16(i16_edgy()),
        b in arr16(i16_edgy()),
    ) {
        let sum = (a + b).to_array();
        let prod = (a * b).to_array();
        let (aa, ba) = (a.to_array(), b.to_array());
        for i in 0..16 {
            prop_assert_eq!(sum[i], aa[i].wrapping_add(ba[i]));
            prop_assert_eq!(prod[i], aa[i].wrapping_mul(ba[i]));
        }
    }

    #[test]
    fn i16_saturating_ops_match_scalar(
        a in arr16(i16_edgy()),
        b in arr16(i16_edgy()),
    ) {
        let (add, sub, mul) = (a.sat_add(b).to_array(), a.sat_sub(b).to_array(), a.sat_mul(b).to_array());
        let (aa, ba) = (a.to_array(), b.to_array());
        for i in 0..16 {
            prop_assert_eq!(add[i], aa[i].saturating_add(ba[i]));
            prop_assert_eq!(sub[i], aa[i].saturating_sub(ba[i]));
            prop_assert_eq!(mul[i], aa[i].saturating_mul(ba[i]));
        }
    }

    #[test]
    fn i32_shifts_out_of_range_follow_policy(
        a in arr8(i32_edgy()),
        amount in 0u32..64,
    ) {
        let aa = a.to_array();
        let shl = a.shl_scalar(amount).to_array();
        let shr = a.shr_scalar(amount).to_array();
        let shrl = a.shr_logical(amount).to_array();
        for i in 0..8 {
            if amount >= 32 {
                prop_assert_eq!(shl[i], 0);
                prop_assert_eq!(shrl[i], 0);
                prop_assert_eq!(shr[i], if aa[i] < 0 { -1 } else { 0 });
            } else {
                prop_assert_eq!(shl[i], aa[i] << amount);
                prop_assert_eq!(shr[i], aa[i] >> amount);
                prop_assert_eq!(shrl[i], ((aa[i] as u32) >> amount) as i32);
            }
        }
    }

    #[test]
    fn f32_min_max_ignore_nan(
        a in arr8(f32_with_specials()),
        b in arr8(f32_with_specials()),
    ) {
        let (mn, mx) = (a.min(b).to_array(), a.max(b).to_array());
        let (aa, ba) = (a.to_array(), b.to_array());
        for i in 0..8 {
            let (x, y) = (aa[i], ba[i]);
            let (emn, emx) = match (x.is_nan(), y.is_nan()) {
                (true, true) => (f32::NAN, f32::NAN),
                (true, false) => (y, y),
                (false, true) => (x, x),
                (false, false) => (x.min(y), x.max(y)),
            };
            // Compared by value: the sign of a zero result is not specified.
            prop_assert!(mn[i] == emn || (mn[i].is_nan() && emn.is_nan()));
            prop_assert!(mx[i] == emx || (mx[i].is_nan() && emx.is_nan()));
        }
    }

    #[test]
    fn complex_mul_is_unfused_formula(
        xs in prop::collection::vec(f32_with_specials(), 32),
    ) {
        // One vector of 32 specials split into four operands: fewer, smaller
        // strategy arguments keep the debug-build test frame within the
        // default 2 MiB test-thread stack.
        let lane = |k: usize| -> Simd<f32, 8> { Simd::from_slice(&xs[8 * k..8 * k + 8]) };
        let (ar, ai, br, bi) = (lane(0), lane(1), lane(2), lane(3));
        let p = ComplexSimd::new(ar, ai).mul(ComplexSimd::new(br, bi));
        let (re, im) = (p.real.to_array(), p.imag.to_array());
        let (ar, ai, br, bi) = (ar.to_array(), ai.to_array(), br.to_array(), bi.to_array());
        for i in 0..8 {
            prop_assert!(same_f32(re[i], ar[i] * br[i] - ai[i] * bi[i]));
            prop_assert!(same_f32(im[i], ar[i] * bi[i] + ai[i] * br[i]));
        }
    }

    #[test]
    fn lane_order_round_trips(v in arr8(i32_edgy())) {
        let arr = v.to_array();
        prop_assert_eq!(Simd::<i32, 8>::from_array(arr).to_array(), arr);
        prop_assert_eq!(v.reverse().reverse().to_array(), arr);
        for (i, &x) in arr.iter().enumerate() {
            prop_assert_eq!(v.reverse().to_array()[7 - i], x);
        }
    }

    #[test]
    fn int_to_float_cast_matches_as(v in arr8(i32_edgy())) {
        let f = v.cast::<f32>().to_array();
        let arr = v.to_array();
        for i in 0..8 {
            prop_assert!(same_f32(f[i], arr[i] as f32));
        }
    }
}

/// Bitwise equality for `f32`, except that any two NaNs are equal. Rust does
/// not specify NaN payload or sign propagation, and Miri deliberately varies
/// it, so comparing NaN bits would test the interpreter rather than the code.
fn same_f32(got: f32, want: f32) -> bool {
    (got.is_nan() && want.is_nan()) || got.to_bits() == want.to_bits()
}

/// Lane vector of 8 values drawn from `elem`.
fn arr8<T: SimdElement>(elem: impl Strategy<Value = T>) -> impl Strategy<Value = Simd<T, 8>> {
    prop::array::uniform8(elem).prop_map(Simd::from_array)
}

/// Lane vector of 16 values drawn from `elem`.
fn arr16<T: SimdElement>(elem: impl Strategy<Value = T>) -> impl Strategy<Value = Simd<T, 16>> {
    prop::array::uniform16(elem).prop_map(Simd::from_array)
}
