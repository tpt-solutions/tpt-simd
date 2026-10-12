//! Shifts and rotates, including rounding descale for fixed-point code.
//!
//! Run with: `cargo run -p tpt-simd-shift --example shift`

use tpt_simd_core::{I32x8, U32x8};
use tpt_simd_shift::{rotate_left_var_i32, shift_left_i32, shift_with_rounding_i32};

fn main() {
    assert_eq!(shift_left_i32(I32x8::splat(3), 4), I32x8::splat(48));
    assert_eq!(shift_left_i32(I32x8::splat(3), 32), I32x8::splat(0)); // out of range

    // Round-half-up descale: (x + (1 << (n-1))) >> n, evaluated in 64 bits.
    let r = shift_with_rounding_i32(
        I32x8::from_array([5, 6, 7, -5, -6, -7, i32::MAX, i32::MIN]),
        2,
    );
    assert_eq!(r.to_array(), [1, 2, 2, -1, -1, -2, 1 << 29, i32::MIN >> 2]);

    // Per-lane rotation amounts; rotation is modulo 32.
    let r = rotate_left_var_i32(
        I32x8::splat(i32::MIN),
        U32x8::from_array([0, 1, 31, 32, 33, 0, 0, 0]),
    );
    assert_eq!(
        r.to_array(),
        [
            i32::MIN,
            1,
            1 << 30,
            i32::MIN,
            1,
            i32::MIN,
            i32::MIN,
            i32::MIN
        ]
    );

    println!("shift: all checks passed");
}
