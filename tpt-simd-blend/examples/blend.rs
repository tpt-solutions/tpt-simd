//! Branch-free selection: pick lanes from two vectors without a branch.
//!
//! Run with: `cargo run -p tpt-simd-blend --example blend`

use tpt_simd_blend::{blend_gt_f32, blend_i32};
use tpt_simd_core::{F32x8, I32x8, SimdMask};

fn main() {
    // Lanes 0..4 take `a`, lanes 4..8 take `b`.
    let m = SimdMask::<i32, 8>::from_bitmask(0b0000_1111);
    let r = blend_i32(m, I32x8::splat(1), I32x8::splat(2));
    assert_eq!(r.to_array(), [1, 1, 1, 1, 2, 2, 2, 2]);

    // `if x > 2.0 { 1.0 } else { 0.0 }` per lane. NaN compares false and selects `b`.
    let x = F32x8::from_array([1.0, 3.0, f32::NAN, 0.0, 5.0, 5.0, 9.0, -1.0]);
    let r = blend_gt_f32(x, F32x8::splat(2.0), F32x8::splat(1.0), F32x8::splat(0.0));
    assert_eq!(r.to_array(), [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]);

    println!("blend: all checks passed");
}
