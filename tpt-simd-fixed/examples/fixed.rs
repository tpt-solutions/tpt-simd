//! Type-safe Q-format fixed-point arithmetic on 8 lanes.
//!
//! Run with: `cargo run -p tpt-simd-fixed --example fixed`

use tpt_simd_core::{F32x8, I32x8};
use tpt_simd_fixed::Fixed;

// Q16.16 in an i32: 16 integer bits, 16 fractional bits.
type Q = Fixed<i32, 16, 16, 8>;

fn main() {
    let a: Q = Fixed::from_f32(F32x8::splat(1.5));
    let b: Q = Fixed::from_f32(F32x8::splat(2.5));
    let result = a * b; // 1.5 * 2.5 = 3.75, rounded correctly
    assert_eq!(result.to_f32(), F32x8::splat(3.75));

    // Overflow is explicit: saturating ops clamp, checked ops return None.
    let big = Q::from_raw(I32x8::splat(1 << 30)); // 16384.0
    assert_eq!(big.saturating_mul(big), Q::max_value());
    assert!(Q::one().checked_div(Q::splat_raw(0)).is_none());

    println!("fixed: all checks passed");
}
