//! Non-contiguous stores: scatter lanes to arbitrary indices.
//!
//! Run with: `cargo run -p tpt-simd-scatter --example scatter`

use tpt_simd_core::I32x8;
use tpt_simd_scatter::{OutOfBounds, scatter_checked_i32, try_scatter_i32};

fn main() {
    // Checked scatter. With duplicate indices the last lane wins.
    let mut d = [0i32; 3];
    scatter_checked_i32(
        &mut d,
        I32x8::from_array([0, 1, 2, 2, 2, 1, 0, 0]),
        I32x8::from_array([1, 2, 3, 4, 5, 6, 7, 8]),
    );
    assert_eq!(d, [8, 6, 5]);

    // Non-panicking form: nothing is written if any index is out of range.
    let mut d = [0i32; 4];
    let r = try_scatter_i32(
        &mut d,
        I32x8::from_array([0, 1, 2, 3, 4, 0, 0, 0]),
        I32x8::splat(9),
    );
    assert_eq!(r, Err(OutOfBounds { lane: 4, index: 4 }));
    assert_eq!(d, [0; 4]);

    println!("scatter: all checks passed");
}
