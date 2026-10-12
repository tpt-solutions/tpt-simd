//! Aligned storage for SIMD loads and stores.
//!
//! Run with: `cargo run -p tpt-simd-aligned --example aligned`

use tpt_simd_aligned::{
    Aligned32, AlignedBuf, is_aligned, load_aligned, store_aligned, try_load_aligned,
};
use tpt_simd_core::Simd;

fn main() {
    // Stack or static data forced onto a 32-byte boundary.
    let mut buf = Aligned32([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let v = load_aligned::<f32, 8>(&buf.0);
    assert_eq!(v[7], 8.0);
    store_aligned(Simd::<f32, 8>::splat(2.5), &mut buf.0);
    assert_eq!(buf.0, [2.5; 8]);

    // A misaligned slice is rejected by the try_ form instead of panicking.
    assert!(try_load_aligned::<f32, 8>(&buf.0[1..]).is_none());

    // Heap buffer with 64-byte base alignment and densely packed elements.
    let mut b = AlignedBuf::<f32, 64>::filled(100, 1.5);
    assert!(is_aligned(b.as_ptr(), 64));
    b[3] = 2.0;
    assert_eq!(b.len(), 100);

    println!("aligned: all checks passed");
}
