extern crate std;

use crate::*;
use proptest::prelude::*;
use std::vec::Vec;
use tpt_simd_testutil::f32_with_specials;

fn data_f32(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7919 % 2003) as f32 - 1000.0) * 0.37)
        .collect()
}
fn data_f64(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i * 7919 % 2003) as f64 - 1000.0) * 0.37)
        .collect()
}

fn ref_sum(xs: &[f32]) -> f64 {
    xs.iter().map(|&x| f64::from(x)).sum()
}
fn ref_abs(xs: &[f32]) -> f64 {
    xs.iter().map(|&x| f64::from(x).abs()).sum()
}

/// Tolerance relative to the sum of absolute values (see crate docs).
fn close32(got: f32, exact: f64, abs_sum: f64, n: usize) -> bool {
    let eps = f64::from(f32::EPSILON);
    (f64::from(got) - exact).abs() <= (n as f64 + 16.0) * eps * abs_sum + 1e-30
}

const LENS: [usize; 16] = [
    0, 1, 2, 7, 8, 9, 31, 32, 33, 63, 64, 65, 100, 1023, 1025, 5000,
];

#[test]
fn sums_match_reference_for_all_tails() {
    for n in LENS.into_iter().chain(0..140) {
        let x = data_f32(n);
        let (e, a) = (ref_sum(&x), ref_abs(&x));
        assert!(close32(sum_f32(&x), e, a, n), "sum n={n}");
        assert!(close32(pairwise_sum_f32(&x), e, a, n), "pairwise n={n}");
        assert!(close32(sum_compensated_f32(&x), e, a, 1), "comp n={n}");
        let y = data_f64(n);
        let e64: f64 = y.iter().sum();
        let a64: f64 = y.iter().map(|v| v.abs()).sum();
        let tol = (n as f64 + 16.0) * f64::EPSILON * a64 + 1e-300;
        assert!((sum_f64(&y) - e64).abs() <= tol);
        assert!((pairwise_sum_f64(&y) - e64).abs() <= tol);
        assert!((sum_compensated_f64(&y) - e64).abs() <= tol);
    }
}

#[test]
fn empty_behaviour() {
    assert_eq!(sum_f32(&[]), 0.0);
    assert_eq!(pairwise_sum_f64(&[]), 0.0);
    assert_eq!(sum_compensated_f32(&[]), 0.0);
    assert_eq!(sum_i32(&[]), 0);
    assert_eq!(mean_f32(&[]), None);
    assert_eq!(variance_f64(&[]), None);
    assert_eq!(variance_welford_f64(&[]), None);
    assert_eq!(sample_variance_f32(&[1.0]), None);
    assert_eq!(covariance_f32(&[], &[]), None);
    assert_eq!(sample_covariance_f32(&[1.0], &[1.0]), None);
    assert_eq!(min_f32(&[]), None);
    assert_eq!(max_f64(&[]), None);
    assert_eq!(argmin_f32(&[]), None);
    assert_eq!(argmax_f64(&[]), None);
    assert_eq!(min_i32(&[]), None);
    assert_eq!(argmax_i32(&[]), None);
    assert_eq!(sum_squares_f32(&[]), 0.0);
    assert_eq!(norm_f32(&[]), 0.0);
}

#[test]
fn compensated_beats_naive_on_cancellation() {
    // 1e7 + 1 - 1e7 repeated: the exact sum is the number of triples.
    let n = 3 * 1000;
    let x: Vec<f32> = (0..n)
        .map(|i| match i % 3 {
            0 => 1.0e8,
            1 => 1.0,
            _ => -1.0e8,
        })
        .collect();
    let exact = 1000.0f32;
    let comp = sum_compensated_f32(&x);
    let naive_loop: f32 = x.iter().sum();
    let simd = sum_f32(&x);
    assert_eq!(comp, exact);
    assert!((naive_loop - exact).abs() > 100.0, "naive {naive_loop}");
    assert!((simd - exact).abs() > 100.0, "simd {simd}");
    // f64 version
    let y: Vec<f64> = x.iter().map(|&v| f64::from(v) * 1.0e8).collect();
    let _ = y;
    let z: Vec<f64> = (0..n)
        .map(|i| match i % 3 {
            0 => 1.0e17,
            1 => 1.0,
            _ => -1.0e17,
        })
        .collect();
    assert_eq!(sum_compensated_f64(&z), 1000.0);
    assert!((sum_f64(&z) - 1000.0).abs() > 100.0);
}

#[test]
fn pairwise_beats_naive_on_long_constant_sum() {
    // 0.1f32 summed 4M times: left-to-right drifts badly, pairwise stays close.
    let x = std::vec![0.1f32; 4_000_000];
    let exact = 4_000_000.0 * f64::from(0.1f32);
    let naive: f32 = x.iter().sum();
    let pw = pairwise_sum_f32(&x);
    assert!((f64::from(pw) - exact).abs() < (f64::from(naive) - exact).abs());
    assert!((f64::from(pw) - exact).abs() / exact < 1e-6);
}

#[test]
fn nan_and_inf_sums() {
    for n in [1usize, 5, 8, 33, 100] {
        let mut x = data_f32(n);
        x[n / 2] = f32::NAN;
        assert!(sum_f32(&x).is_nan());
        assert!(pairwise_sum_f32(&x).is_nan());
        assert!(sum_compensated_f32(&x).is_nan());
        assert!(mean_f32(&x).unwrap().is_nan());
        assert!(variance_f32(&x).unwrap().is_nan());
        assert!(norm_f32(&x).is_nan());
        x[n / 2] = f32::INFINITY;
        assert_eq!(sum_f32(&x), f32::INFINITY);
        assert_eq!(sum_compensated_f32(&x), f32::INFINITY);
        assert_eq!(sum_squares_f32(&x), f32::INFINITY);
        assert_eq!(norm_f32(&x), f32::INFINITY);
        x[0] = f32::NEG_INFINITY;
        if n > 1 {
            assert!(sum_f32(&x).is_nan());
            assert!(sum_compensated_f32(&x).is_nan());
        }
    }
    // Finite inputs overflowing: compensated agrees with plain (inf).
    let big = [f32::MAX, f32::MAX, 1.0];
    assert_eq!(sum_compensated_f32(&big), f32::INFINITY);
}

#[test]
fn mean_variance_covariance_known_values() {
    let x = [2.0f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    assert_eq!(mean_f64(&x), Some(5.0));
    assert_eq!(variance_f64(&x), Some(4.0));
    assert_eq!(variance_welford_f64(&x), Some(4.0));
    assert!((sample_variance_f64(&x).unwrap() - 32.0 / 7.0).abs() < 1e-12);
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
    assert!((covariance_f64(&x, &y).unwrap() - 8.0).abs() < 1e-12);
    assert!((sample_covariance_f64(&x, &y).unwrap() - 64.0 / 7.0).abs() < 1e-12);
    assert_eq!(norm_f64(&[3.0, 4.0]), 5.0);
}

#[test]
fn variance_stable_with_large_mean() {
    // mean 1e6, spread 1: naive E[x^2]-E[x]^2 in f32 is garbage; two-pass is fine.
    let n = 1000;
    let x: Vec<f32> = (0..n).map(|i| 1.0e6 + (i % 4) as f32).collect();
    let v = variance_f32(&x).unwrap();
    let w = variance_welford_f32(&x).unwrap();
    assert!((v - 1.25).abs() < 0.05, "{v}");
    assert!((w - 1.25).abs() < 0.1, "{w}");
}

#[test]
#[should_panic(expected = "length mismatch")]
fn covariance_length_mismatch_panics() {
    let _ = covariance_f32(&[1.0], &[1.0, 2.0]);
}

fn ref_min(x: &[f32]) -> Option<f32> {
    if x.is_empty() {
        return None;
    }
    let mut r = f32::NAN;
    for &v in x {
        if r.is_nan() || v < r {
            r = v;
        }
    }
    Some(r)
}
fn ref_max(x: &[f32]) -> Option<f32> {
    if x.is_empty() {
        return None;
    }
    let mut r = f32::NAN;
    for &v in x {
        if r.is_nan() || v > r {
            r = v;
        }
    }
    Some(r)
}
fn eq_val(a: Option<f32>, b: Option<f32>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => (a.is_nan() && b.is_nan()) || a == b,
        _ => false,
    }
}

#[test]
fn extrema_nan_policy_and_indices() {
    for n in 1..140usize {
        let mut x = data_f32(n);
        assert!(eq_val(min_f32(&x), ref_min(&x)));
        assert!(eq_val(max_f32(&x), ref_max(&x)));
        // NaN ignored
        x[0] = f32::NAN;
        x[n - 1] = f32::NAN;
        assert!(eq_val(min_f32(&x), ref_min(&x)), "n={n}");
        assert!(eq_val(max_f32(&x), ref_max(&x)), "n={n}");
        let i = argmin_f32(&x);
        assert_eq!(i.is_none(), n <= 2, "n={n}");
        if let Some(i) = i {
            assert_eq!(x[i], ref_min(&x).unwrap());
            assert!(x[..i].iter().all(|&v| v.is_nan() || v > x[i]));
        }
        // all NaN
        let all = std::vec![f32::NAN; n];
        assert!(min_f32(&all).unwrap().is_nan());
        assert!(max_f32(&all).unwrap().is_nan());
        assert_eq!(argmin_f32(&all), None);
        assert_eq!(argmax_f32(&all), None);
    }
    // ties: first index wins
    let t = [3.0f32, 1.0, 1.0, 9.0, 9.0, 1.0, 9.0, 0.5, 0.5, 5.0, 1.0];
    assert_eq!(argmin_f32(&t), Some(7));
    assert_eq!(argmax_f32(&t), Some(3));
    let inf = [f32::INFINITY, f32::NEG_INFINITY, 0.0];
    assert_eq!(min_f32(&inf), Some(f32::NEG_INFINITY));
    assert_eq!(max_f32(&inf), Some(f32::INFINITY));
    assert_eq!(argmax_f64(&[1.0, 5.0, 5.0]), Some(1));
    assert_eq!(min_f64(&[2.0, f64::NAN, -1.0]), Some(-1.0));
}

#[test]
fn integer_kernels() {
    for n in 0..140usize {
        let x: Vec<i32> = (0..n)
            .map(|i| (i as i32).wrapping_mul(2_000_000_011).wrapping_sub(7))
            .collect();
        assert_eq!(sum_i32(&x), x.iter().fold(0i32, |a, &b| a.wrapping_add(b)));
        assert_eq!(
            sum_i32_wide(&x),
            x.iter().map(|&v| i64::from(v)).sum::<i64>()
        );
        assert_eq!(min_i32(&x), x.iter().copied().min());
        assert_eq!(max_i32(&x), x.iter().copied().max());
        if n > 0 {
            let mi = x.iter().copied().min().unwrap();
            assert_eq!(argmin_i32(&x), x.iter().position(|&v| v == mi));
            let ma = x.iter().copied().max().unwrap();
            assert_eq!(argmax_i32(&x), x.iter().position(|&v| v == ma));
        }
    }
    assert_eq!(sum_i32(&[i32::MAX, 1]), i32::MIN);
    assert_eq!(sum_i32_wide(&[i32::MAX, 1]), i64::from(i32::MAX) + 1);
    assert_eq!(min_i32(&[i32::MAX; 40]), Some(i32::MAX));
    assert_eq!(max_i32(&[i32::MIN; 40]), Some(i32::MIN));
}

#[test]
fn deterministic_across_calls() {
    let x = data_f32(1234);
    assert_eq!(sum_f32(&x).to_bits(), sum_f32(&x).to_bits());
}

proptest! {
    #[test]
    fn prop_sum_family(v in proptest::collection::vec(-1.0e4f32..1.0e4, 0..400)) {
        let (e, a, n) = (ref_sum(&v), ref_abs(&v), v.len());
        prop_assert!(close32(sum_f32(&v), e, a, n));
        prop_assert!(close32(pairwise_sum_f32(&v), e, a, n));
        prop_assert!(close32(sum_compensated_f32(&v), e, a, 1));
        let ss: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        prop_assert!(close32(sum_squares_f32(&v), ss, ss, n));
    }

    #[test]
    fn prop_moments(v in proptest::collection::vec(-100.0f64..100.0, 1..300)) {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        prop_assert!((mean_f64(&v).unwrap() - mean).abs() <= 1e-10);
        prop_assert!((variance_f64(&v).unwrap() - var).abs() <= 1e-8 * (1.0 + var));
        prop_assert!((variance_welford_f64(&v).unwrap() - var).abs() <= 1e-8 * (1.0 + var));
        let w: Vec<f64> = v.iter().rev().copied().collect();
        let cov = v.iter().zip(&w).map(|(x, y)| (x - mean) * (y - mean)).sum::<f64>() / n;
        prop_assert!((covariance_f64(&v, &w).unwrap() - cov).abs() <= 1e-8 * (1.0 + cov.abs()));
    }

    #[test]
    fn prop_extrema(v in proptest::collection::vec(f32_with_specials(), 0..300)) {
        prop_assert!(eq_val(min_f32(&v), ref_min(&v)));
        prop_assert!(eq_val(max_f32(&v), ref_max(&v)));
        let ai = argmax_f32(&v);
        match ref_max(&v) {
            Some(m) if !m.is_nan() => {
                let i = ai.unwrap();
                prop_assert!(v[i] == m);
                prop_assert!(v[..i].iter().all(|&x| x != m));
            }
            _ => prop_assert!(ai.is_none()),
        }
    }

    #[test]
    fn prop_specials_sum_matches_scalar_class(v in proptest::collection::vec(f32_with_specials(), 0..200)) {
        let scalar: f32 = v.iter().sum();
        let got = sum_f32(&v);
        // NaN-ness and infinity are order independent unless finite overflow occurs.
        if v.iter().any(|x| x.is_nan()) {
            prop_assert!(got.is_nan());
        }
        if v.iter().all(|x| x.is_finite()) && scalar.is_finite() && got.is_finite() {
            let a: f64 = v.iter().map(|&x| f64::from(x).abs()).sum();
            prop_assert!(f64::from(got - scalar).abs() <= (v.len() as f64 + 16.0) * f64::from(f32::EPSILON) * a * 2.0 + 1e-30);
        }
    }

    #[test]
    fn prop_i32(v in proptest::collection::vec(any::<i32>(), 0..300)) {
        prop_assert_eq!(sum_i32(&v), v.iter().fold(0i32, |a, &b| a.wrapping_add(b)));
        prop_assert_eq!(min_i32(&v), v.iter().copied().min());
        prop_assert_eq!(max_i32(&v), v.iter().copied().max());
    }
}
