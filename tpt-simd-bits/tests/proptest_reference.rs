//! Property tests: each kernel against a plain reference.

use proptest::prelude::*;
use tpt_simd_bits::{
    delta_decode_u32, delta_encode_u32, popcount_u64, prefix_sum_u32, prefix_sum_u64,
    zigzag_decode_slice_u32, zigzag_encode_slice_i32,
};

proptest! {
    #[test]
    fn popcount_matches_count_ones(words in prop::collection::vec(any::<u64>(), 0..300)) {
        let expected: u64 = words.iter().map(|w| u64::from(w.count_ones())).sum();
        prop_assert_eq!(popcount_u64(&words), expected);
    }

    #[test]
    fn delta_round_trips(values in prop::collection::vec(any::<u32>(), 0..300)) {
        let mut v = values.clone();
        delta_encode_u32(&mut v);
        delta_decode_u32(&mut v);
        prop_assert_eq!(v, values);
    }

    #[test]
    fn delta_encode_matches_wrapping_difference(values in prop::collection::vec(any::<u32>(), 1..300)) {
        let mut v = values.clone();
        delta_encode_u32(&mut v);
        prop_assert_eq!(v[0], values[0]);
        for i in 1..values.len() {
            prop_assert_eq!(v[i], values[i].wrapping_sub(values[i - 1]));
        }
    }

    #[test]
    fn prefix_sum_u32_matches_scan(values in prop::collection::vec(any::<u32>(), 0..300)) {
        let mut got = values.clone();
        prefix_sum_u32(&mut got);
        let mut acc = 0u32;
        let want: Vec<u32> = values.iter().map(|&x| { acc = acc.wrapping_add(x); acc }).collect();
        prop_assert_eq!(got, want);
    }

    #[test]
    fn prefix_sum_u64_matches_scan(values in prop::collection::vec(any::<u64>(), 0..300)) {
        let mut got = values.clone();
        prefix_sum_u64(&mut got);
        let mut acc = 0u64;
        let want: Vec<u64> = values.iter().map(|&x| { acc = acc.wrapping_add(x); acc }).collect();
        prop_assert_eq!(got, want);
    }

    #[test]
    fn zigzag_matches_formula_and_round_trips(values in prop::collection::vec(any::<i32>(), 0..300)) {
        let mut encoded = vec![0u32; values.len()];
        zigzag_encode_slice_i32(&values, &mut encoded);
        for (&x, &z) in values.iter().zip(&encoded) {
            prop_assert_eq!(z, ((x << 1) ^ (x >> 31)) as u32);
        }
        let mut decoded = vec![0i32; values.len()];
        zigzag_decode_slice_u32(&encoded, &mut decoded);
        prop_assert_eq!(decoded, values);
    }
}
