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
    // `pred` is the pivot of the partition that produced `v` as its upper side, if any.
    // Elements of `v` are all at least `pred`, so if the next pivot is not greater than `pred`
    // the input has many equal keys, and those are split off first (as in pdqsort).
    let mut pred: Option<T> = None;
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
        let pivot = choose_pivot(v, less);
        if let Some(p) = pred.filter(|p| !less(p, &pivot)) {
            let equal = partition_branchless(v, |x| !less(&p, x));
            v = &mut v[equal..];
            pred = None;
            continue;
        }
        let lt = partition_branchless(v, |x| less(x, &pivot));
        let (left, right) = v.split_at_mut(lt);
        // Recurse into the smaller side, loop on the larger one.
        if left.len() < right.len() {
            introsort(left, less, depth);
            v = right;
            pred = Some(pivot);
        } else {
            introsort(right, less, depth);
            v = left;
            pred = None;
        }
    }
}

/// Median of three samples, or the ninther (median of three medians of three) for long slices.
fn choose_pivot<T: Copy, F: Fn(&T, &T) -> bool>(v: &[T], less: &F) -> T {
    let n = v.len();
    let mid = n / 2;
    if n < 128 {
        return median_of_three(v[0], v[mid], v[n - 1], less);
    }
    let step = n / 8;
    let a = median_of_three(v[0], v[step], v[2 * step], less);
    let b = median_of_three(v[mid - step], v[mid], v[mid + step], less);
    let c = median_of_three(v[n - 1 - 2 * step], v[n - 1 - step], v[n - 1], less);
    median_of_three(a, b, c, less)
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
    // Two branch-free passes: first move the elements below the pivot to the front, then move
    // the elements equal to the pivot (not above it) to the front of the rest.
    let lt = partition_branchless(v, |x| less(x, &pivot));
    let eq = partition_branchless(&mut v[lt..], |x| !less(&pivot, x));
    (lt, lt + eq)
}

/// Lomuto partition with no data-dependent branch: every element is swapped into place
/// unconditionally, so random inputs do not pay a misprediction per element. Returns the
/// number of elements for which `pred` holds, which are moved to the front.
fn partition_branchless<T: Copy, P: Fn(&T) -> bool>(v: &mut [T], pred: P) -> usize {
    let mut lo = 0;
    for i in 0..v.len() {
        let hit = pred(&v[i]);
        v.swap(lo, i);
        lo += usize::from(hit);
    }
    lo
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

/// Sorts `v` in ascending order with an LSD radix sort: one histogram pass over all four
/// bytes, then up to four stable scatter passes, skipping any byte that is constant.
///
/// `scratch` is caller-provided working space (the crate does not allocate). It must be at
/// least `v.len()` long, and its contents are unspecified afterwards.
///
/// # Panics
///
/// Panics if `scratch.len() < v.len()`.
///
/// # Examples
///
/// ```
/// let mut v = [5u32, 1, 4_000_000_000, 3, 1];
/// let mut scratch = [0u32; 5];
/// tpt_simd_sort::sort_u32(&mut v, &mut scratch);
/// assert_eq!(v, [1, 1, 3, 5, 4_000_000_000]);
/// ```
pub fn sort_u32(v: &mut [u32], scratch: &mut [u32]) {
    let n = v.len();
    assert!(
        scratch.len() >= n,
        "sort_u32: scratch length {} is shorter than input length {n}",
        scratch.len()
    );
    if n < 2 {
        return;
    }
    let scratch = &mut scratch[..n];

    let mut counts = [[0usize; 256]; 4];
    for &x in v.iter() {
        for (digit, column) in counts.iter_mut().enumerate() {
            column[((x >> (8 * digit)) & 0xFF) as usize] += 1;
        }
    }

    let mut in_v = true;
    for (digit, column) in counts.iter().enumerate() {
        // A digit that is the same for every element does not change the order.
        if column.contains(&n) {
            continue;
        }
        let shift = 8 * digit;
        let mut offsets = [0usize; 256];
        let mut sum = 0;
        for (offset, &count) in offsets.iter_mut().zip(column.iter()) {
            *offset = sum;
            sum += count;
        }
        if in_v {
            radix_scatter(v, scratch, shift, &mut offsets);
        } else {
            radix_scatter(scratch, v, shift, &mut offsets);
        }
        in_v = !in_v;
    }
    if !in_v {
        v.copy_from_slice(scratch);
    }
}

/// One stable counting-sort pass on the byte at `shift`: writes `src` into `dst` in order of
/// that byte, using `offsets` as the running write position of each bucket.
fn radix_scatter(src: &[u32], dst: &mut [u32], shift: usize, offsets: &mut [usize; 256]) {
    for &x in src {
        let bucket = ((x >> shift) & 0xFF) as usize;
        dst[offsets[bucket]] = x;
        offsets[bucket] += 1;
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

#[cfg(test)]
mod radix_tests {
    extern crate std;
    use super::sort_u32;
    use std::vec::Vec;

    fn check(input: &[u32]) {
        let mut want = input.to_vec();
        want.sort_unstable();
        let mut got = input.to_vec();
        let mut scratch = std::vec![0u32; input.len()];
        sort_u32(&mut got, &mut scratch);
        assert_eq!(got, want, "len {}", input.len());
    }

    #[test]
    fn radix_matches_std_over_sizes_and_shapes() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for n in (0..70).chain([127, 128, 129, 1000, 4096, 65_537]) {
            let random: Vec<u32> = (0..n).map(|_| next() as u32).collect();
            check(&random);
            // Few distinct values, so most digits are constant or repeated.
            let few: Vec<u32> = (0..n).map(|_| (next() % 5) as u32 * 0x0101_0101).collect();
            check(&few);
            // Values that differ only in the top byte: three digits are constant.
            let top: Vec<u32> = (0..n).map(|_| (next() as u32) & 0xFF00_0000).collect();
            check(&top);
        }
        check(&[]);
        check(&[u32::MAX, 0, u32::MAX, 0]);
        let sorted: Vec<u32> = (0..5000).collect();
        check(&sorted);
        let reversed: Vec<u32> = (0..5000).rev().collect();
        check(&reversed);
        check(&std::vec![7u32; 5000]);
    }

    #[test]
    #[should_panic(expected = "sort_u32: scratch length 2 is shorter than input length 3")]
    fn short_scratch_panics() {
        sort_u32(&mut [1u32, 2, 3], &mut [0u32; 2]);
    }
}
