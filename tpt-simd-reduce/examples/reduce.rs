//! Slice reductions: sums, statistics and argmin / argmax.
//!
//! Run with: `cargo run -p tpt-simd-reduce --example reduce`

use tpt_simd_reduce::*;

fn main() {
    let xs = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(sum_f32(&xs), 10.0);
    assert_eq!(mean_f32(&xs), Some(2.5));
    assert_eq!(variance_f32(&xs), Some(1.25));
    assert_eq!(argmax_f32(&[1.0, f32::NAN, 7.0, 3.0]), Some(2));
    assert_eq!(min_f32(&[]), None);

    // Compensated summation survives catastrophic cancellation.
    let big = [1.0e8f32, 1.0, -1.0e8];
    assert_eq!(sum_f32(&big), 0.0);
    assert_eq!(sum_compensated_f32(&big), 1.0);

    println!("reduce: all checks passed");
}
