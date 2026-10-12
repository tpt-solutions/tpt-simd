//! Lane permutation, stereo interleaving and 8x8 transposes.
//!
//! Run with: `cargo run -p tpt-simd-permute --example permute`

use tpt_simd_core::{F32x8, Simd};
use tpt_simd_permute::{interleave_stereo_i16, permute_f32, transpose_8x8_i16};

fn main() {
    // Reverse the lanes of a vector.
    let v = F32x8::from_fn(|i| i as f32);
    let r = permute_f32(v, Simd::from_array([7, 6, 5, 4, 3, 2, 1, 0]));
    assert_eq!(r.to_array(), [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);

    // Interleave two mono channels into packed stereo.
    let mut out = [0i16; 6];
    interleave_stereo_i16(&[1, 2, 3], &[-1, -2, -3], &mut out);
    assert_eq!(out, [1, -1, 2, -2, 3, -3]);

    // In-place 8x8 transpose.
    let mut m = [[0i16; 8]; 8];
    m[1][2] = 7;
    transpose_8x8_i16(&mut m);
    assert_eq!(m[2][1], 7);

    println!("permute: all checks passed");
}
