//! Comparisons that produce lane masks, and counting over slices.
//!
//! Run with: `cargo run -p tpt-simd-compare --example compare`

use tpt_simd_compare::*;
use tpt_simd_core::F32x8;

fn main() {
    let a = F32x8::from_array([1.0, f32::NAN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = F32x8::splat(3.0);
    let m = cmp_gt_f32(a, b);
    assert_eq!(mask_to_bitmask(m), 0b1111_1000);
    assert!(mask_any(m) && !mask_all(m));
    assert_eq!(mask_count(cmp_ne_f32(a, a)), 1); // only the NaN lane

    // Slice-level count: how many elements exceed a threshold.
    let data = [1, 5, 9, 2, 7];
    assert_eq!(count_gt_i32(&data, 4), 3);

    println!("compare: all checks passed");
}
