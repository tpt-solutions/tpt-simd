extern crate std;

use crate::*;
use proptest::prelude::*;
use tpt_simd_core::{F32x8, I8x32, I16x16, I32x8, U8x32};
use tpt_simd_testutil::{f32_with_specials, i16_edgy, i32_edgy};

fn simd_of<T: tpt_simd_core::SimdElement, S: Strategy<Value = T>, const N: usize>(
    s: S,
) -> impl Strategy<Value = tpt_simd_core::Simd<T, N>> {
    proptest::collection::vec(s, N).prop_map(|v| tpt_simd_core::Simd::from_slice(&v))
}

/// Scalar model of the documented 8-lane tree.
fn tree8(l: [f32; 8], f: impl Fn(f32, f32) -> f32) -> f32 {
    let t = [f(l[0], l[4]), f(l[1], l[5]), f(l[2], l[6]), f(l[3], l[7])];
    let u = [f(t[0], t[2]), f(t[1], t[3])];
    f(u[0], u[1])
}

fn same(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

fn nan_min(l: &[f32]) -> f32 {
    l.iter()
        .copied()
        .filter(|x| !x.is_nan())
        .fold(f32::NAN, |a, b| if a.is_nan() || b < a { b } else { a })
}

fn nan_max(l: &[f32]) -> f32 {
    l.iter()
        .copied()
        .filter(|x| !x.is_nan())
        .fold(f32::NAN, |a, b| if a.is_nan() || b > a { b } else { a })
}

#[test]
fn sum_i32_basic_and_wrap() {
    assert_eq!(
        horizontal_sum_i32(I32x8::from_array([1, 2, 3, 4, 5, 6, 7, 8])),
        36
    );
    assert_eq!(
        horizontal_sum_i32(I32x8::splat(i32::MAX)),
        i32::MAX.wrapping_mul(8)
    );
    assert_eq!(horizontal_sum_i32(I32x8::zero()), 0);
}

#[test]
fn sum_f32_order_is_documented_tree() {
    // Values chosen so that left-to-right and tree order differ.
    let v = F32x8::from_array([1.0e8, 1.0, -1.0e8, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let expect = ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
    assert_eq!(horizontal_sum_f32(v).to_bits(), expect.to_bits());
}

#[test]
fn sum_f32_specials() {
    let mut a = [1.0f32; 8];
    a[3] = f32::INFINITY;
    assert_eq!(horizontal_sum_f32(F32x8::from_array(a)), f32::INFINITY);
    a[5] = f32::NEG_INFINITY;
    assert!(horizontal_sum_f32(F32x8::from_array(a)).is_nan());
    a[0] = f32::NAN;
    assert!(horizontal_sum_f32(F32x8::from_array(a)).is_nan());
    assert_eq!(horizontal_sum_f32(F32x8::splat(f32::MAX)), f32::INFINITY);
}

#[test]
fn sum_i16_extremes() {
    assert_eq!(horizontal_sum_i16(I16x16::splat(i16::MAX)), 16 * 32767);
    assert_eq!(horizontal_sum_i16(I16x16::splat(i16::MIN)), -16 * 32768);
}

#[test]
fn sum_i8_u8_extremes() {
    assert_eq!(horizontal_sum_u8(U8x32::splat(255)), 8160);
    assert_eq!(horizontal_sum_i8(I8x32::splat(-128)), -4096);
    assert_eq!(horizontal_sum_i8(I8x32::splat(127)), 4064);
}

#[test]
fn min_max_nan_policy() {
    let v = F32x8::from_array([f32::NAN, 3.0, 2.0, f32::NAN, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(horizontal_min_f32(v), 2.0);
    assert_eq!(horizontal_max_f32(v), 8.0);
    // NaN in the lanes that start a tree step.
    let w = F32x8::from_array([f32::NAN, f32::NAN, f32::NAN, f32::NAN, 1.0, 2.0, 3.0, 4.0]);
    assert_eq!(horizontal_min_f32(w), 1.0);
    assert_eq!(horizontal_max_f32(w), 4.0);
    assert!(horizontal_min_f32(F32x8::splat(f32::NAN)).is_nan());
    assert!(horizontal_max_f32(F32x8::splat(f32::NAN)).is_nan());
    let inf = F32x8::from_array([
        f32::INFINITY,
        1.0,
        f32::NEG_INFINITY,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]);
    assert_eq!(horizontal_min_f32(inf), f32::NEG_INFINITY);
    assert_eq!(horizontal_max_f32(inf), f32::INFINITY);
}

#[test]
fn int_min_max() {
    let v = I16x16::from_fn(|i| (i as i16 - 7) * 1000);
    assert_eq!(horizontal_max_i16(v), 8000);
    assert_eq!(horizontal_min_i16(v), -7000);
    assert_eq!(horizontal_max_i16(I16x16::splat(i16::MIN)), i16::MIN);
    assert_eq!(horizontal_min_i32(I32x8::splat(i32::MAX)), i32::MAX);
    assert_eq!(
        horizontal_max_i32(I32x8::from_array([i32::MIN, 0, 0, 0, 0, 0, 0, -1])),
        0
    );
}

#[test]
fn product_f32() {
    assert_eq!(horizontal_product_f32(F32x8::splat(2.0)), 256.0);
    assert_eq!(horizontal_product_f32(F32x8::splat(1.0e20)), f32::INFINITY);
    let mut a = [1.0f32; 8];
    a[2] = f32::NAN;
    assert!(horizontal_product_f32(F32x8::from_array(a)).is_nan());
    a[2] = 0.0;
    a[6] = f32::INFINITY;
    assert!(horizontal_product_f32(F32x8::from_array(a)).is_nan()); // 0 * inf
}

#[test]
fn generic_reductions_odd_lane_counts() {
    use tpt_simd_core::Simd;
    assert_eq!(reduce_sum(Simd::<i32, 1>::from_array([7])), 7);
    assert_eq!(reduce_sum(Simd::<i32, 5>::from_array([1, 2, 3, 4, 5])), 15);
    assert_eq!(reduce_product(Simd::<i32, 3>::from_array([2, 3, 4])), 24);
    assert_eq!(reduce_min(Simd::<i32, 7>::from_fn(|i| 10 - i as i32)), 4);
    assert_eq!(reduce_max(Simd::<u8, 6>::from_fn(|i| i as u8)), 5);
}

proptest! {
    #[test]
    fn prop_sum_i32(v in simd_of::<i32, _, 8>(i32_edgy())) {
        let r = v.to_array().iter().fold(0i32, |a, &b| a.wrapping_add(b));
        prop_assert_eq!(horizontal_sum_i32(v), r);
        prop_assert_eq!(reduce_sum(v), r);
    }

    #[test]
    fn prop_sum_f32(v in simd_of::<f32, _, 8>(f32_with_specials())) {
        let r = tree8(v.to_array(), |a, b| a + b);
        prop_assert!(same(horizontal_sum_f32(v), r));
        let p = tree8(v.to_array(), |a, b| a * b);
        prop_assert!(same(horizontal_product_f32(v), p));
    }

    #[test]
    fn prop_sum_i16(v in simd_of::<i16, _, 16>(i16_edgy())) {
        let r: i32 = v.to_array().iter().map(|&x| x as i32).sum();
        prop_assert_eq!(horizontal_sum_i16(v), r);
        prop_assert_eq!(horizontal_max_i16(v), *v.as_array().iter().max().unwrap());
        prop_assert_eq!(horizontal_min_i16(v), *v.as_array().iter().min().unwrap());
    }

    #[test]
    fn prop_i32_minmax(v in simd_of::<i32, _, 8>(i32_edgy())) {
        prop_assert_eq!(horizontal_max_i32(v), *v.as_array().iter().max().unwrap());
        prop_assert_eq!(horizontal_min_i32(v), *v.as_array().iter().min().unwrap());
    }

    #[test]
    fn prop_f32_minmax(v in simd_of::<f32, _, 8>(f32_with_specials())) {
        let (mn, mx) = (nan_min(v.as_array()), nan_max(v.as_array()));
        let (gn, gx) = (horizontal_min_f32(v), horizontal_max_f32(v));
        // Zero sign is unspecified; compare numerically.
        prop_assert!((gn.is_nan() && mn.is_nan()) || gn == mn);
        prop_assert!((gx.is_nan() && mx.is_nan()) || gx == mx);
    }

    #[test]
    fn prop_u8_i8(a in simd_of::<u8, _, 32>(any::<u8>()), b in simd_of::<i8, _, 32>(any::<i8>())) {
        prop_assert_eq!(horizontal_sum_u8(a), a.as_array().iter().map(|&x| x as u32).sum::<u32>());
        prop_assert_eq!(horizontal_sum_i8(b), b.as_array().iter().map(|&x| x as i32).sum::<i32>());
    }
}
