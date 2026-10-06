extern crate std;

use crate::*;
use proptest::collection::vec;
use proptest::prelude::*;
use std::vec::Vec;
use tpt_simd_core::{ComplexSimd, F32x8};
use tpt_simd_testutil::{f32_with_specials, i16_edgy};

fn same(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

fn ref_i16(a: &[i16], b: &[i16]) -> i32 {
    a.iter()
        .zip(b)
        .fold(0i32, |s, (&x, &y)| s.wrapping_add(x as i32 * y as i32))
}

/// Scalar model of the documented f32 order: 32 partial sums, element i -> i % 32,
/// then (acc0+acc1)+(acc2+acc3) lane-wise, then the 8-lane pairwise tree.
fn ref_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut p = [0.0f32; 32];
    for i in 0..a.len() {
        p[i % 32] += a[i] * b[i];
    }
    let mut l = [0.0f32; 8];
    for j in 0..8 {
        l[j] = (p[j] + p[8 + j]) + (p[16 + j] + p[24 + j]);
    }
    let t = [l[0] + l[4], l[1] + l[5], l[2] + l[6], l[3] + l[7]];
    let u = [t[0] + t[2], t[1] + t[3]];
    u[0] + u[1]
}

/// Scalar model of the documented saturating order: 8 lanes (i % 8), tree.
fn ref_sat(a: &[i16], b: &[i16]) -> i32 {
    let mut l = [0i32; 8];
    for i in 0..a.len() {
        l[i % 8] = l[i % 8].saturating_add(a[i] as i32 * b[i] as i32);
    }
    let t = [
        l[0].saturating_add(l[4]),
        l[1].saturating_add(l[5]),
        l[2].saturating_add(l[6]),
        l[3].saturating_add(l[7]),
    ];
    let u = [t[0].saturating_add(t[2]), t[1].saturating_add(t[3])];
    u[0].saturating_add(u[1])
}

#[test]
fn i16_all_tail_lengths() {
    for n in 0..=130 {
        let a: Vec<i16> = (0..n)
            .map(|i| (i as i16).wrapping_mul(997) ^ 0x1234)
            .collect();
        let b: Vec<i16> = (0..n).map(|i| (i as i16).wrapping_mul(-313)).collect();
        assert_eq!(dot_product_i16(&a, &b), ref_i16(&a, &b), "n={n}");
        assert_eq!(portable::dot_product_i16(&a, &b), ref_i16(&a, &b), "n={n}");
        assert_eq!(dot_product_saturating_i16(&a, &b), ref_sat(&a, &b), "n={n}");
    }
}

#[test]
fn i16_overflow_wraps() {
    let a = std::vec![i16::MIN; 100];
    let r = ref_i16(&a, &a);
    assert_eq!(dot_product_i16(&a, &a), r);
    assert_eq!(r, 100i32.wrapping_mul(1 << 30));
    assert_eq!(dot_product_i16(&[i16::MIN; 2], &[i16::MIN; 2]), i32::MIN);
}

#[test]
fn i16_saturates() {
    let a = std::vec![i16::MIN; 100];
    assert_eq!(dot_product_saturating_i16(&a, &a), i32::MAX);
    let b = std::vec![i16::MAX; 100];
    assert_eq!(dot_product_saturating_i16(&a, &b), i32::MIN);
    // Saturated lane stays clamped even if later products are negative.
    let mut x = std::vec![i16::MIN; 8 * 3];
    let mut y = std::vec![i16::MIN; 8 * 3];
    x.extend([i16::MAX; 8]);
    y.extend([i16::MIN; 8]);
    assert_eq!(dot_product_saturating_i16(&x, &y), ref_sat(&x, &y));
    assert_eq!(dot_product_saturating_i16(&[], &[]), 0);
}

#[test]
fn flac_example_from_spec() {
    let a: Vec<i16> = (0..256).collect();
    let b: Vec<i16> = (0..256).collect();
    assert_eq!(dot_product_i16(&a, &b), 5_559_680);
}

#[test]
fn f32_matches_documented_order_all_lengths() {
    for n in 0..=100 {
        let a: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.37 - 5.0) * 1e3 + 0.1)
            .collect();
        let b: Vec<f32> = (0..n).map(|i| 1.0 / (i as f32 + 1.0) - 0.2).collect();
        let r = ref_f32(&a, &b);
        assert!(same(dot_product_f32(&a, &b), r), "n={n}");
        assert!(same(portable::dot_product_f32(&a, &b), r), "n={n}");
    }
}

#[test]
fn f32_specials() {
    assert_eq!(dot_product_f32(&[], &[]), 0.0);
    assert!(dot_product_f32(&[f32::NAN, 1.0], &[1.0, 1.0]).is_nan());
    assert_eq!(dot_product_f32(&[f32::INFINITY], &[2.0]), f32::INFINITY);
    assert!(dot_product_f32(&[f32::INFINITY], &[0.0]).is_nan());
    assert_eq!(dot_product_f32(&[f32::MAX; 40], &[2.0; 40]), f32::INFINITY);
    let mut a = std::vec![1.0f32; 70];
    a[69] = f32::INFINITY; // lives in the partial tail chunk
    assert_eq!(dot_product_f32(&a, &std::vec![1.0; 70]), f32::INFINITY);
}

fn cplx(n: usize, seed: f32) -> Vec<ComplexSimd<f32, 8>> {
    (0..n)
        .map(|i| {
            ComplexSimd::new(
                F32x8::from_fn(|j| (i * 8 + j) as f32 * 0.25 + seed),
                F32x8::from_fn(|j| (j as f32 - i as f32) * 0.5 - seed),
            )
        })
        .collect()
}

#[test]
fn complex_matches_model() {
    for n in 0..=11 {
        let (a, b) = (cplx(n, 1.0), cplx(n, -2.0));
        let r = dot_product_complex_f32(&a, &b);
        let mut e = ComplexSimd::<f32, 8>::zero();
        for i in 0..n {
            e = e.add(a[i].mul(b[i]));
        }
        assert_eq!(
            r.real.to_array().map(f32::to_bits),
            e.real.to_array().map(f32::to_bits)
        );
        assert_eq!(
            r.imag.to_array().map(f32::to_bits),
            e.imag.to_array().map(f32::to_bits)
        );
    }
}

#[test]
fn complex_not_conjugated() {
    let i = ComplexSimd::new(F32x8::splat(0.0), F32x8::splat(1.0));
    let r = dot_product_complex_f32(&[i], &[i]); // i*i = -1 (conjugated would be +1)
    assert_eq!(r.real.to_array(), [-1.0; 8]);
    assert_eq!(r.imag.to_array(), [0.0; 8]);
    assert_eq!(dot_product_complex_f32(&[], &[]), ComplexSimd::zero());
}

#[test]
#[should_panic(expected = "dot_product_i16: length mismatch (3 vs 2)")]
fn i16_len_mismatch_panics() {
    dot_product_i16(&[1, 2, 3], &[1, 2]);
}

#[test]
#[should_panic(expected = "dot_product_f32: length mismatch")]
fn f32_len_mismatch_panics() {
    dot_product_f32(&[1.0], &[]);
}

#[test]
#[should_panic(expected = "dot_product_saturating_i16: length mismatch")]
fn sat_len_mismatch_panics() {
    dot_product_saturating_i16(&[1], &[]);
}

#[test]
#[should_panic(expected = "dot_product_complex_f32: length mismatch")]
fn complex_len_mismatch_panics() {
    dot_product_complex_f32(&[ComplexSimd::zero()], &[]);
}

fn pair<S: Strategy + Clone>(
    s: S,
    max: usize,
) -> impl Strategy<Value = (Vec<S::Value>, Vec<S::Value>)>
where
    S::Value: Clone + core::fmt::Debug,
{
    (0..=max).prop_flat_map(move |n| (vec(s.clone(), n), vec(s.clone(), n)))
}

proptest! {
    #[test]
    fn prop_i16((a, b) in pair(i16_edgy().boxed(), 300)) {
        prop_assert_eq!(dot_product_i16(&a, &b), ref_i16(&a, &b));
        prop_assert_eq!(portable::dot_product_i16(&a, &b), ref_i16(&a, &b));
        prop_assert_eq!(dot_product_saturating_i16(&a, &b), ref_sat(&a, &b));
    }

    #[test]
    fn prop_f32((a, b) in pair(f32_with_specials().boxed(), 300)) {
        let r = ref_f32(&a, &b);
        prop_assert!(same(dot_product_f32(&a, &b), r));
        prop_assert!(same(portable::dot_product_f32(&a, &b), r));
    }
}
