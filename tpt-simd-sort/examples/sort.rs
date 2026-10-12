//! Sorting, argsort, quickselect and top-k over plain slices.
//!
//! Run with: `cargo run -p tpt-simd-sort --example sort`

use tpt_simd_sort::{argsort, partition, select_nth, sort_unstable, top_k_smallest};

fn main() {
    let mut v = [5, 1, 4, 1, 3];
    sort_unstable(&mut v);
    assert_eq!(v, [1, 1, 3, 4, 5]);

    // Quickselect: put the element of rank k in place without a full sort.
    let mut w = [9, 2, 7, 4];
    select_nth(&mut w, 1);
    assert_eq!(w[1], 4);

    // Sort indices by key, leaving the data untouched.
    let keys = [30, 10, 20];
    let mut order = [0usize, 1, 2];
    argsort(&keys, &mut order);
    assert_eq!(order, [1, 2, 0]);

    // Top-k: the k smallest values move to the front, sorted ascending.
    let mut data = [8, 3, 9, 1, 7];
    top_k_smallest(&mut data, 2);
    assert_eq!(&data[..2], &[1, 3]);

    // Partition: move matching elements to the front, return how many matched.
    let mut p = [1, 2, 3, 4, 5, 6];
    let n = partition(&mut p, |x| x % 2 == 0);
    assert_eq!(n, 3);

    println!("sort: all checks passed");
}
