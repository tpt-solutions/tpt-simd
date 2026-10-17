//! The prefix-sum kernels must match a plain scalar scan exactly, including
//! wrapping at the top of the range and every tail length. This runs against
//! whichever path the build compiled (AVX2 or scalar).

use proptest::prelude::*;
use tpt_simd_bits::{delta_decode_u32, prefix_sum_u32, prefix_sum_u64};

fn scalar_u32(v: &[u32]) -> Vec<u32> {
    let mut acc = 0u32;
    v.iter()
        .map(|&x| {
            acc = acc.wrapping_add(x);
            acc
        })
        .collect()
}

fn scalar_u64(v: &[u64]) -> Vec<u64> {
    let mut acc = 0u64;
    v.iter()
        .map(|&x| {
            acc = acc.wrapping_add(x);
            acc
        })
        .collect()
}

proptest! {
    #[test]
    fn prefix_sum_u32_matches_scalar(v in prop::collection::vec(any::<u32>(), 0..300)) {
        let mut got = v.clone();
        prefix_sum_u32(&mut got);
        prop_assert_eq!(got, scalar_u32(&v));
    }

    #[test]
    fn prefix_sum_u64_matches_scalar(v in prop::collection::vec(any::<u64>(), 0..300)) {
        let mut got = v.clone();
        prefix_sum_u64(&mut got);
        prop_assert_eq!(got, scalar_u64(&v));
    }

    #[test]
    fn delta_decode_inverts_encode(v in prop::collection::vec(any::<u32>(), 0..300)) {
        let mut enc = v.clone();
        for i in (1..enc.len()).rev() {
            enc[i] = enc[i].wrapping_sub(enc[i - 1]);
        }
        delta_decode_u32(&mut enc);
        prop_assert_eq!(enc, v);
    }
}

#[test]
fn every_length_up_to_40_with_max_values() {
    // All-max inputs wrap at every position, so any lane or carry mistake shows up.
    for n in 0..=40 {
        let v32: Vec<u32> = (0..n).map(|i| u32::MAX - (i as u32 % 3)).collect();
        let mut got32 = v32.clone();
        prefix_sum_u32(&mut got32);
        assert_eq!(got32, scalar_u32(&v32), "u32 n={n}");

        let v64: Vec<u64> = (0..n).map(|i| u64::MAX - (i as u64 % 5)).collect();
        let mut got64 = v64.clone();
        prefix_sum_u64(&mut got64);
        assert_eq!(got64, scalar_u64(&v64), "u64 n={n}");
    }
}
