//! Lane selection by index: gather elements within a vector or from a slice.
//!
//! Run with: `cargo run -p tpt-simd-select --example select`

use tpt_simd_core::Simd;
use tpt_simd_select::{select_from_slice_i32, select_i32, try_select_from_slice_i32};

fn main() {
    // Index-based lane selection; only the low 3 bits of each index matter.
    let v = Simd::from_array([10, 11, 12, 13, 14, 15, 16, 17]);
    let i = Simd::from_array([7, 0, 0, 1, 8, 9, 3, 3]);
    assert_eq!(
        select_i32(v, i).to_array(),
        [17, 10, 10, 11, 10, 11, 13, 13]
    );

    // Selection from a slice.
    let data = [5, 6, 7, 8];
    let i = Simd::from_array([3, 2, 1, 0, 0, 1, 2, 3]);
    assert_eq!(
        select_from_slice_i32(&data, i).to_array(),
        [8, 7, 6, 5, 5, 6, 7, 8]
    );

    // Non-panicking form returns None when the slice is too short.
    assert!(try_select_from_slice_i32(&[1, 2], Simd::splat(2)).is_none());

    println!("select: all checks passed");
}
