//! f16 and bf16 element-wise arithmetic against exact or correctly rounded references.

use half::{bf16, f16};
use proptest::prelude::*;
use tpt_simd_half::{add_bf16, add_f16, mul_bf16, mul_f16};

/// Finite f16 values, so that the references are not NaN or infinity.
fn finite_f16() -> impl Strategy<Value = f16> {
    any::<u16>().prop_filter_map("finite", |bits| {
        let v = f16::from_bits(bits);
        v.is_finite().then_some(v)
    })
}

fn finite_bf16() -> impl Strategy<Value = bf16> {
    any::<u16>().prop_filter_map("finite", |bits| {
        let v = bf16::from_bits(bits);
        v.is_finite().then_some(v)
    })
}

/// Distance between two bf16 values, in units of the last place.
fn bf16_ulps(a: bf16, b: bf16) -> i32 {
    let key = |v: bf16| {
        let bits = i32::from(v.to_bits());
        if bits & 0x8000 != 0 {
            -(bits & 0x7fff)
        } else {
            bits
        }
    };
    (key(a) - key(b)).abs()
}

proptest! {
    #[test]
    fn f16_add_is_correctly_rounded(a in finite_f16(), b in finite_f16()) {
        let mut out = [f16::ZERO];
        add_f16(&[a], &[b], &mut out);
        // The exact sum fits in f64, so one rounding to f16 is the correct result.
        let exact = a.to_f64() + b.to_f64();
        prop_assert_eq!(out[0].to_bits(), f16::from_f64(exact).to_bits());
    }

    #[test]
    fn f16_mul_is_correctly_rounded(a in finite_f16(), b in finite_f16()) {
        let mut out = [f16::ZERO];
        mul_f16(&[a], &[b], &mut out);
        prop_assert_eq!(out[0].to_bits(), f16::from_f64(a.to_f64() * b.to_f64()).to_bits());
    }

    #[test]
    fn bf16_mul_is_correctly_rounded(a in finite_bf16(), b in finite_bf16()) {
        let mut out = [bf16::ZERO];
        mul_bf16(&[a], &[b], &mut out);
        // The exact product of two bf16 values fits in f64, so one rounding is correct.
        prop_assert_eq!(out[0].to_bits(), bf16::from_f64(a.to_f64() * b.to_f64()).to_bits());
    }

    #[test]
    fn bf16_add_is_within_one_ulp_of_exact(a in finite_bf16(), b in finite_bf16()) {
        let mut out = [bf16::ZERO];
        add_bf16(&[a], &[b], &mut out);
        let exact = bf16::from_f64(a.to_f64() + b.to_f64());
        if exact.is_finite() {
            prop_assert!(bf16_ulps(out[0], exact) <= 1, "{} + {}: {} vs {}", a, b, out[0], exact);
        }
    }

    #[test]
    fn slices_match_scalar_operations(
        xs in prop::collection::vec(finite_f16(), 0..100),
        ys in prop::collection::vec(finite_f16(), 0..100),
    ) {
        let n = xs.len().min(ys.len());
        let mut sum = vec![f16::ZERO; n];
        let mut prod = vec![f16::ZERO; n];
        add_f16(&xs[..n], &ys[..n], &mut sum);
        mul_f16(&xs[..n], &ys[..n], &mut prod);
        for i in 0..n {
            prop_assert_eq!(sum[i].to_bits(), f16::from_f64(xs[i].to_f64() + ys[i].to_f64()).to_bits());
            prop_assert_eq!(prod[i].to_bits(), f16::from_f64(xs[i].to_f64() * ys[i].to_f64()).to_bits());
        }
    }
}

#[test]
#[should_panic(expected = "length mismatch")]
fn mismatched_lengths_panic() {
    let a = [f16::ONE; 2];
    let b = [f16::ONE; 3];
    let mut out = [f16::ZERO; 2];
    add_f16(&a, &b, &mut out);
}
