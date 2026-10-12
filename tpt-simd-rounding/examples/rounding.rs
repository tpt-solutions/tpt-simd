//! Rounding modes for f32 lanes, and conversion to integers.
//!
//! Run with: `cargo run -p tpt-simd-rounding --example rounding`

use tpt_simd_core::F32x8;
use tpt_simd_rounding::{round_f32, round_ties_even_f32, round_to_nearest_even_i32};

fn main() {
    // round_f32: halves round away from zero.
    let x = F32x8::from_array([0.5, 1.5, 2.5, -0.5, -2.5, 0.49999997, 2.4, -2.6]);
    assert_eq!(
        round_f32(x).to_array(),
        [1.0, 2.0, 3.0, -1.0, -3.0, 0.0, 2.0, -3.0]
    );

    // round_ties_even_f32: halves round to the even neighbour (banker's rounding).
    let y = F32x8::from_array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.6, -2.4]);
    assert_eq!(
        round_ties_even_f32(y).to_array(),
        [0.0, 2.0, 2.0, -0.0, -2.0, -2.0, 3.0, -2.0]
    );

    // Integer conversion saturates; NaN becomes 0.
    let z = F32x8::from_array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 1e20, f32::NAN]);
    assert_eq!(
        round_to_nearest_even_i32(z).to_array(),
        [0, 2, 2, 0, -2, -2, i32::MAX, 0]
    );

    println!("rounding: all checks passed");
}
