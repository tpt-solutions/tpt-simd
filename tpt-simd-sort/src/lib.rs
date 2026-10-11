//! Unstable sort, quickselect, argsort, partition and top-k.
//!
//! | function | notes |
//! |---|---|
//! | [`sort_unstable`], [`sort_by`] | introsort, 3-way partition, heapsort fallback |
//! | [`select_nth`], [`top_k_smallest`] | quickselect with the same fallback |
//! | [`argsort`] | sorts indices by keys; not stable |
//! | [`partition`] | in-place, unstable, returns the count of matches |
//!
//! Worst case is O(n log n) for every sort and select function. Equal elements
//! are handled in one pass by the 3-way partition, so many duplicates are cheap.
//!
//! ## Example
//!
//! ```
//! use tpt_simd_sort::{select_nth, sort_unstable};
//!
//! let mut v = [5, 1, 4, 1, 3];
//! sort_unstable(&mut v);
//! assert_eq!(v, [1, 1, 3, 4, 5]);
//!
//! let mut w = [9, 2, 7, 4];
//! select_nth(&mut w, 1);
//! assert_eq!(w[1], 4);
//! ```
#![no_std]
#![deny(unsafe_code)]
#![warn(missing_docs)]

/// Below this length, insertion sort is used.
const INSERTION_LIMIT: usize = 16;

/// Sorts `v` in ascending order. Not stable.
pub fn sort_unstable<T: Ord + Copy>(v: &mut [T]) {
    sort_by(v, |a, b| a < b);
}

/// Sorts `v` so that `less(a, b)` orders elements. Not stable.
///
/// `less` must be a strict weak ordering; otherwise the result is unspecified
/// (but the call still terminates and does not read out of bounds).
pub fn sort_by<T: Copy, F: Fn(&T, &T) -> bool>(v: &mut [T], less: F) {
    let depth = depth_limit(v.len());
    introsort(v, &less, depth);
}

/// Rearranges `v` so that `v[k]` is the element that would be there if `v`
/// were sorted, with smaller-or-equal elements before it and greater-or-equal
/// elements after it. Returns nothing; read `v[k]` afterwards.
///
/// # Panics
///
/// Panics if `k >= v.len()`.
pub fn select_nth<T: Ord + Copy>(v: &mut [T], k: usize) {
    assert!(
        k < v.len(),
        "select_nth: index {k} out of range for length {}",
        v.len()
    );
    let depth = depth_limit(v.len());
    select(v, k, &|a: &T, b: &T| a < b, depth);
}

/// Moves the `k` smallest elements to the front of `v`, sorted ascending.
/// The rest of `v` is left in unspecified order. If `k >= v.len()`, the
/// whole slice is sorted.
pub fn top_k_smallest<T: Ord + Copy>(v: &mut [T], k: usize) {
    if k == 0 || v.is_empty() {
        return;
    }
    if k < v.len() {
        select_nth(v, k - 1);
        sort_unstable(&mut v[..k]);
    } else {
        sort_unstable(v);
    }
}

/// Writes into `out` the indices that would sort `keys` ascending.
///
/// # Panics
///
/// Panics if `out.len() != keys.len()`.
pub fn argsort<T: Ord>(keys: &[T], out: &mut [usize]) {
    assert!(
        out.len() == keys.len(),
        "argsort: output length {} does not match key length {}",
        out.len(),
        keys.len()
    );
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = i;
    }
    sort_by(out, |&a, &b| keys[a] < keys[b]);
}

/// Moves elements for which `pred` is true to the front of `v`. Returns the
/// number of such elements. Order within each group is unspecified.
pub fn partition<T, F: Fn(&T) -> bool>(v: &mut [T], pred: F) -> usize {
    let mut lo = 0;
    let mut hi = v.len();
    while lo < hi {
        if pred(&v[lo]) {
            lo += 1;
        } else {
            hi -= 1;
            v.swap(lo, hi);
        }
    }
    lo
}

fn depth_limit(n: usize) -> u32 {
    // 2 * floor(log2(n)) + 1 levels before falling back to heapsort.
    2 * (usize::BITS - n.leading_zeros()) + 1
}

fn introsort<T: Copy, F: Fn(&T, &T) -> bool>(mut v: &mut [T], less: &F, mut depth: u32) {
    loop {
        if v.len() <= INSERTION_LIMIT {
            insertion_sort(v, less);
            return;
        }
        if depth == 0 {
            heapsort(v, less);
            return;
        }
        depth -= 1;
        let (lt, gt) = three_way_partition(v, less);
        let (left, rest) = v.split_at_mut(lt);
        let right = &mut rest[gt - lt..];
        // Recurse into the smaller side, loop on the larger one.
        if left.len() < right.len() {
            introsort(left, less, depth);
            v = right;
        } else {
            introsort(right, less, depth);
            v = left;
        }
    }
}

fn select<T: Copy, F: Fn(&T, &T) -> bool>(mut v: &mut [T], mut k: usize, less: &F, mut depth: u32) {
    loop {
        if v.len() <= INSERTION_LIMIT {
            insertion_sort(v, less);
            return;
        }
        if depth == 0 {
            heapsort(v, less);
            return;
        }
        depth -= 1;
        let (lt, gt) = three_way_partition(v, less);
        if k < lt {
            v = &mut v[..lt];
        } else if k < gt {
            // Equal to the pivot: already in its final place.
            return;
        } else {
            k -= gt;
            v = &mut v[gt..];
        }
    }
}

/// Partitions `v` into `[< pivot | == pivot | > pivot]` and returns the
/// boundaries `(lt, gt)`. The pivot is the median of three samples.
fn three_way_partition<T: Copy, F: Fn(&T, &T) -> bool>(v: &mut [T], less: &F) -> (usize, usize) {
    let n = v.len();
    let mid = n / 2;
    let pivot = median_of_three(v[0], v[mid], v[n - 1], less);
    let mut lt = 0;
    let mut i = 0;
    let mut gt = n;
    while i < gt {
        if less(&v[i], &pivot) {
            v.swap(lt, i);
            lt += 1;
            i += 1;
        } else if less(&pivot, &v[i]) {
            gt -= 1;
            v.swap(i, gt);
        } else {
            i += 1;
        }
    }
    (lt, gt)
}

fn median_of_three<T: Copy, F: Fn(&T, &T) -> bool>(a: T, b: T, c: T, less: &F) -> T {
    if less(&a, &b) {
        if less(&b, &c) {
            b
        } else if less(&a, &c) {
            c
        } else {
            a
        }
    } else if less(&a, &c) {
        a
    } else if less(&b, &c) {
        c
    } else {
        b
    }
}

fn insertion_sort<T: Copy, F: Fn(&T, &T) -> bool>(v: &mut [T], less: &F) {
    for i in 1..v.len() {
        let x = v[i];
        let mut j = i;
        while j > 0 && less(&x, &v[j - 1]) {
            v[j] = v[j - 1];
            j -= 1;
        }
        v[j] = x;
    }
}

fn heapsort<T: Copy, F: Fn(&T, &T) -> bool>(v: &mut [T], less: &F) {
    let n = v.len();
    for start in (0..n / 2).rev() {
        sift_down(v, start, n, less);
    }
    for end in (1..n).rev() {
        v.swap(0, end);
        sift_down(v, 0, end, less);
    }
}

fn sift_down<T: Copy, F: Fn(&T, &T) -> bool>(v: &mut [T], mut root: usize, end: usize, less: &F) {
    loop {
        let mut child = 2 * root + 1;
        if child >= end {
            return;
        }
        if child + 1 < end && less(&v[child], &v[child + 1]) {
            child += 1;
        }
        if !less(&v[root], &v[child]) {
            return;
        }
        v.swap(root, child);
        root = child;
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec;
    use std::vec::Vec;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }
    }

    fn inputs() -> Vec<Vec<u32>> {
        let mut rng = Lcg(5);
        let mut out = Vec::new();
        for n in (0..60).chain([100, 257, 1000, 4096]) {
            out.push((0..n).map(|_| rng.next() as u32).collect());
            out.push((0..n).map(|_| (rng.next() % 4) as u32).collect()); // many duplicates
            out.push((0..n as u32).collect()); // sorted
            out.push((0..n as u32).rev().collect()); // reversed
            out.push(vec![7u32; n]); // all equal
        }
        out
    }

    #[test]
    fn sort_matches_std() {
        for mut v in inputs() {
            let mut want = v.clone();
            want.sort_unstable();
            sort_unstable(&mut v);
            assert_eq!(v, want);
        }
    }

    #[test]
    fn sort_by_descending() {
        let mut v = [3, 1, 4, 1, 5, 9, 2, 6];
        sort_by(&mut v, |a, b| a > b);
        assert_eq!(v, [9, 6, 5, 4, 3, 2, 1, 1]);
    }

    #[test]
    fn select_nth_matches_sorted_order() {
        for v in inputs() {
            if v.is_empty() {
                continue;
            }
            let mut sorted = v.clone();
            sorted.sort_unstable();
            for k in [0, v.len() / 3, v.len() / 2, v.len() - 1] {
                let mut w = v.clone();
                select_nth(&mut w, k);
                assert_eq!(w[k], sorted[k], "len={} k={k}", v.len());
                assert!(w[..k].iter().all(|x| *x <= w[k]));
                assert!(w[k + 1..].iter().all(|x| *x >= w[k]));
            }
        }
    }

    #[test]
    fn top_k_returns_k_smallest_sorted() {
        let v: Vec<u32> = (0..500u32).map(|i| (i * 7919) % 500).collect();
        for k in [0usize, 1, 10, 499, 500, 600] {
            let mut w = v.clone();
            top_k_smallest(&mut w, k);
            let take = k.min(v.len());
            let mut sorted = v.clone();
            sorted.sort_unstable();
            assert_eq!(&w[..take], &sorted[..take], "k={k}");
        }
    }

    #[test]
    fn argsort_orders_keys() {
        let keys = [30u32, 10, 20, 10, 40];
        let mut idx = [0usize; 5];
        argsort(&keys, &mut idx);
        let got: Vec<u32> = idx.iter().map(|&i| keys[i]).collect();
        assert_eq!(got, [10, 10, 20, 30, 40]);
        let mut seen = idx;
        seen.sort_unstable();
        assert_eq!(seen, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn partition_groups_matches() {
        let mut v: Vec<u32> = (0..101).collect();
        let n = partition(&mut v, |x| x % 3 == 0);
        assert_eq!(n, 34);
        assert!(v[..n].iter().all(|x| x % 3 == 0));
        assert!(v[n..].iter().all(|x| x % 3 != 0));
        let mut e: [u32; 0] = [];
        assert_eq!(partition(&mut e, |_| true), 0);
    }

    #[test]
    #[should_panic(expected = "select_nth: index 3 out of range for length 3")]
    fn select_nth_out_of_range_panics() {
        select_nth(&mut [1u32, 2, 3], 3);
    }

    #[test]
    #[should_panic(expected = "argsort: output length 2 does not match key length 3")]
    fn argsort_length_mismatch_panics() {
        argsort(&[1u32, 2, 3], &mut [0usize; 2]);
    }
}
