//! Vectorised f32 transcendental functions with documented error bounds.
//!
//! Run with: `cargo run -p tpt-simd-math --example math`

use tpt_simd_math::{Simd, exp_f32, exp_f32x8};

fn main() {
    let x = [0.0f32, 1.0, -1.0, 10.0, 0.5, 2.0, -20.0, 88.0, 3.0];
    let mut y = [0.0f32; 9];
    exp_f32(&x, &mut y);
    assert_eq!(y[0], 1.0);
    assert!((y[1] - core::f32::consts::E).abs() < 1e-6);

    // Eight lanes at once.
    let v = exp_f32x8(Simd::splat(0.0));
    assert_eq!(v.to_array(), [1.0; 8]);

    println!("math: all checks passed");
}
