//! Property tests: each sort kernel against `std`.

use proptest::prelude::*;
use tpt_simd_sort::{
    argsort, partition, select_nth, sort_by, sort_u32, sort_unstable, top_k_smallest,
};

proptest! {
    #[test]
    fn sort_unstable_matches_std(mut v in prop::collection::vec(any::<i32>(), 0..600)) {
        let mut want = v.clone();
        want.sort_unstable();
        sort_unstable(&mut v);
        prop_assert_eq!(v, want);
    }

    #[test]
    fn sort_u32_matches_std(v in prop::collection::vec(any::<u32>(), 0..600)) {
        let mut got = v.clone();
        let mut scratch = vec![0u32; got.len()];
        sort_u32(&mut got, &mut scratch);
        let mut want = v;
        want.sort_unstable();
        prop_assert_eq!(got, want);
    }

    #[test]
    fn sort_by_orders_descending(mut v in prop::collection::vec(any::<i16>(), 0..600)) {
        sort_by(&mut v, |a, b| a > b);
        prop_assert!(v.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn select_nth_places_the_kth_order_statistic(
        v in prop::collection::vec(any::<i32>(), 1..400),
        k_seed in any::<usize>(),
    ) {
        let k = k_seed % v.len();
        let mut sorted = v.clone();
        sorted.sort_unstable();
        let mut got = v;
        select_nth(&mut got, k);
        prop_assert_eq!(got[k], sorted[k]);
    }

    #[test]
    fn top_k_smallest_returns_the_k_smallest(
        v in prop::collection::vec(any::<i32>(), 0..400),
        k_seed in any::<usize>(),
    ) {
        let k = if v.is_empty() { 0 } else { k_seed % (v.len() + 1) };
        let mut sorted = v.clone();
        sorted.sort_unstable();
        let mut got = v;
        top_k_smallest(&mut got, k);
        let mut head = got[..k].to_vec();
        head.sort_unstable();
        prop_assert_eq!(head, sorted[..k].to_vec());
    }

    #[test]
    fn argsort_is_a_permutation_that_sorts_the_keys(keys in prop::collection::vec(any::<i32>(), 0..400)) {
        let mut out = vec![0usize; keys.len()];
        argsort(&keys, &mut out);
        let mut seen = out.clone();
        seen.sort_unstable();
        prop_assert_eq!(seen, (0..keys.len()).collect::<Vec<_>>());
        prop_assert!(out.windows(2).all(|w| keys[w[0]] <= keys[w[1]]));
    }

    #[test]
    fn partition_groups_by_predicate(mut v in prop::collection::vec(any::<u16>(), 0..400)) {
        let mut before: Vec<u16> = v.clone();
        let mid = partition(&mut v, |x| x % 2 == 0);
        prop_assert!(v[..mid].iter().all(|x| x % 2 == 0));
        prop_assert!(v[mid..].iter().all(|x| x % 2 != 0));
        before.sort_unstable();
        v.sort_unstable();
        prop_assert_eq!(v, before);
    }
}
