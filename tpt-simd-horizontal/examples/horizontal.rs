//! Horizontal reductions: collapse one vector's lanes into a single value.
//!
//! Run with: `cargo run -p tpt-simd-horizontal --example horizontal`

use tpt_simd_core::{F32x8, I32x8, Simd};
use tpt_simd_horizontal::{horizontal_min_f32, horizontal_sum_f32, horizontal_sum_i32, reduce_sum};

fn main() {
    assert_eq!(
        horizontal_sum_i32(I32x8::from_array([1, 2, 3, 4, 5, 6, 7, 8])),
        36
    );
    assert_eq!(horizontal_sum_i32(I32x8::splat(i32::MAX)), -8); // integer sums wrap
    assert_eq!(horizontal_sum_f32(F32x8::from_array([1.0; 8])), 8.0);

    // NaN lanes are ignored by min/max.
    let v = F32x8::from_array([f32::NAN, 3.0, 2.0, f32::NAN, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(horizontal_min_f32(v), 2.0);

    // Generic over element type and lane count.
    assert_eq!(reduce_sum(Simd::<i32, 4>::from_array([1, 2, 3, 4])), 10);

    println!("horizontal: all checks passed");
}
