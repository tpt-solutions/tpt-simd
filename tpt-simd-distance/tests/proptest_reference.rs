//! Property tests: distance kernels against per-element formulas.

use proptest::prelude::*;
use tpt_simd_distance::{
    cosine_distance_f32, hamming_distance_u64, inner_product_f32, inner_product_i8,
    inner_product_u8, l2_squared_f32, l2_squared_i8, l2_squared_u8,
};

fn f32_pair() -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    (0usize..300).prop_flat_map(|n| {
        (
            prop::collection::vec(-10.0f32..10.0, n),
            prop::collection::vec(-10.0f32..10.0, n),
        )
    })
}

proptest! {
    #[test]
    fn f32_kernels_match_formula((a, b) in f32_pair()) {
        let ip: f64 = a.iter().zip(&b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum();
        let l2: f64 = a.iter().zip(&b).map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2)).sum();
        let tol = |v: f64| 1e-3 * v.abs().max(1.0);
        prop_assert!((f64::from(inner_product_f32(&a, &b)) - ip).abs() <= tol(ip));
        prop_assert!((f64::from(l2_squared_f32(&a, &b)) - l2).abs() <= tol(l2));
    }

    #[test]
    fn cosine_distance_is_one_minus_cosine((a, b) in f32_pair()) {
        let dot: f64 = a.iter().zip(&b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum();
        let na: f64 = a.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
        if na > 1e-3 && nb > 1e-3 {
            let want = 1.0 - dot / (na * nb);
            prop_assert!((f64::from(cosine_distance_f32(&a, &b)) - want).abs() <= 1e-3);
        }
    }

    #[test]
    fn integer_kernels_are_exact(
        a in prop::collection::vec(any::<i8>(), 0..400),
        b in prop::collection::vec(any::<i8>(), 0..400),
    ) {
        let n = a.len().min(b.len());
        let (a, b) = (&a[..n], &b[..n]);
        let ip: i64 = a.iter().zip(b).map(|(&x, &y)| i64::from(x) * i64::from(y)).sum();
        let l2: i64 = a.iter().zip(b).map(|(&x, &y)| (i64::from(x) - i64::from(y)).pow(2)).sum();
        prop_assert_eq!(inner_product_i8(a, b), ip);
        prop_assert_eq!(l2_squared_i8(a, b), l2);

        let au: Vec<u8> = a.iter().map(|&x| x as u8).collect();
        let bu: Vec<u8> = b.iter().map(|&x| x as u8).collect();
        let ipu: i64 = au.iter().zip(&bu).map(|(&x, &y)| i64::from(x) * i64::from(y)).sum();
        let l2u: i64 = au.iter().zip(&bu).map(|(&x, &y)| (i64::from(x) - i64::from(y)).pow(2)).sum();
        prop_assert_eq!(inner_product_u8(&au, &bu), ipu);
        prop_assert_eq!(l2_squared_u8(&au, &bu), l2u);
    }

    #[test]
    fn hamming_counts_differing_bits(
        a in prop::collection::vec(any::<u64>(), 0..200),
        b in prop::collection::vec(any::<u64>(), 0..200),
    ) {
        let n = a.len().min(b.len());
        let want: u32 = a[..n].iter().zip(&b[..n]).map(|(x, y)| (x ^ y).count_ones()).sum();
        prop_assert_eq!(hamming_distance_u64(&a[..n], &b[..n]), want);
    }
}
