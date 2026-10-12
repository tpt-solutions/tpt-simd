//! Saturating arithmetic and saturating packs, for pixels and audio.
//!
//! Run with: `cargo run -p tpt-simd-saturate --example saturate`

use tpt_simd_core::{I8x16, I16x16};
use tpt_simd_saturate::{pack_i16_to_i8, pack_i16_to_u8_slice, saturating_add_i16};

fn main() {
    // Adding past i16::MAX clamps instead of wrapping.
    let a = I16x16::splat(i16::MAX);
    assert_eq!(saturating_add_i16(a, I16x16::splat(1)), a);

    // Narrow 16-bit values to 8-bit with clamping.
    let v = I16x16::from_fn(|i| (i as i16 - 8) * 100);
    let p: I8x16 = pack_i16_to_i8(v);
    assert_eq!(p[0], -128); // -800 clamps
    assert_eq!(p[8], 0);
    assert_eq!(p[15], 127); // 700 clamps

    // Slice form: i16 samples to unsigned 8-bit pixels.
    let src = [-500i16, -1, 0, 1, 500];
    let mut dst = [9u8; 5];
    pack_i16_to_u8_slice(&src, &mut dst);
    assert_eq!(dst, [0, 0, 0, 1, 255]);

    println!("saturate: all checks passed");
}
