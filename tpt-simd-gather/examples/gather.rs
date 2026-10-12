//! Non-contiguous loads: gather lanes from arbitrary indices.
//!
//! Run with: `cargo run -p tpt-simd-gather --example gather`

use tpt_simd_core::I32x8;
use tpt_simd_gather::{gather_checked_i32, gather_i32, try_gather_i32};

fn main() {
    // Checked gather: panics on an out-of-range index.
    let data = [5, 6, 7];
    let r = gather_checked_i32(&data, I32x8::from_array([0, 1, 2, 2, 1, 0, 0, 1]));
    assert_eq!(r.to_array(), [5, 6, 7, 7, 6, 5, 5, 6]);

    // Non-panicking form: any index outside 0..len yields None.
    assert!(try_gather_i32(&data, I32x8::splat(3)).is_none());
    assert!(try_gather_i32(&data, I32x8::splat(-1)).is_none());

    // Unchecked gather for hot loops where the caller has already validated indices.
    let table = [0, 10, 20, 30, 40, 50, 60, 70, 80];
    // SAFETY: every index is in bounds of `table`.
    let r = unsafe { gather_i32(table.as_ptr(), I32x8::from_array([8, 0, 4, 4, 1, 2, 3, 5])) };
    assert_eq!(r.to_array(), [80, 0, 40, 40, 10, 20, 30, 50]);

    println!("gather: all checks passed");
}
