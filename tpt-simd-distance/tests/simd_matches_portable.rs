//! The public `f32` kernels must return the same bits as the portable lane
//! reference, whichever path this build compiled (AVX2 or portable).

use proptest::prelude::*;
use tpt_simd_distance::portable;
use tpt_simd_distance::{cosine_distance_f32, inner_product_f32, l2_squared_f32};

/// Same summation as the crate: whole-vector body in lanes, then the tail in order.
fn reduce8(acc: [f32; 8]) -> f32 {
    let lo = (acc[0] + acc[4]) + (acc[1] + acc[5]);
    let hi = (acc[2] + acc[6]) + (acc[3] + acc[7]);
    lo + hi
}

fn split(n: usize) -> usize {
    n - n % 8
}

fn ref_ip(a: &[f32], b: &[f32]) -> f32 {
    let body = split(a.len());
    let mut s = reduce8(portable::inner_product_lanes(&a[..body], &b[..body]));
    for (x, y) in a[body..].iter().zip(&b[body..]) {
        s += x * y;
    }
    s
}

fn ref_l2(a: &[f32], b: &[f32]) -> f32 {
    let body = split(a.len());
    let mut s = reduce8(portable::l2_squared_lanes(&a[..body], &b[..body]));
    for (x, y) in a[body..].iter().zip(&b[body..]) {
        let d = x - y;
        s += d * d;
    }
    s
}

fn ref_cosine(a: &[f32], b: &[f32]) -> f32 {
    let body = split(a.len());
    let (dot, na, nb) = portable::cosine_lanes(&a[..body], &b[..body]);
    let (mut d, mut sa, mut sb) = (reduce8(dot), reduce8(na), reduce8(nb));
    for (x, y) in a[body..].iter().zip(&b[body..]) {
        d += x * y;
        sa += x * x;
        sb += y * y;
    }
    if sa == 0.0 || sb == 0.0 {
        return 1.0;
    }
    1.0 - d / (libm::sqrtf(sa) * libm::sqrtf(sb))
}

fn pair() -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    (0usize..600).prop_flat_map(|n| {
        (
            prop::collection::vec(-1.0e3f32..1.0e3, n),
            prop::collection::vec(-1.0e3f32..1.0e3, n),
        )
    })
}

proptest! {
    #[test]
    fn inner_product_bits_match_portable((a, b) in pair()) {
        prop_assert_eq!(inner_product_f32(&a, &b).to_bits(), ref_ip(&a, &b).to_bits());
    }

    #[test]
    fn l2_squared_bits_match_portable((a, b) in pair()) {
        prop_assert_eq!(l2_squared_f32(&a, &b).to_bits(), ref_l2(&a, &b).to_bits());
    }

    #[test]
    fn cosine_bits_match_portable((a, b) in pair()) {
        prop_assert_eq!(cosine_distance_f32(&a, &b).to_bits(), ref_cosine(&a, &b).to_bits());
    }
}

#[test]
fn special_values_match_portable() {
    let specials = [
        0.0f32,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE,
        1.0e-40,
        -3.5,
        7.25,
        f32::MAX,
    ];
    for n in 0..=20 {
        let a: Vec<f32> = (0..n).map(|i| specials[i % specials.len()]).collect();
        let b: Vec<f32> = (0..n)
            .map(|i| specials[(i * 3 + 1) % specials.len()])
            .collect();
        assert_eq!(
            inner_product_f32(&a, &b).to_bits(),
            ref_ip(&a, &b).to_bits(),
            "ip n={n}"
        );
        assert_eq!(
            l2_squared_f32(&a, &b).to_bits(),
            ref_l2(&a, &b).to_bits(),
            "l2 n={n}"
        );
        assert_eq!(
            cosine_distance_f32(&a, &b).to_bits(),
            ref_cosine(&a, &b).to_bits(),
            "cos n={n}"
        );
    }
}
