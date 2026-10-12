//! Small fixed-size matrices: multiply, invert, with compile-time dimensions.
//!
//! Run with: `cargo run -p tpt-simd-matrix --example matrix`

use tpt_simd_matrix::{mat3x3_inverse_f32, mat4x4_mul_f32, mat8x8_mul_i16};

fn main() {
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let m = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ];
    assert_eq!(mat4x4_mul_f32(m, identity), m);

    // 8x8 i16 product with saturation.
    let mut a = [[0i16; 8]; 8];
    for k in 0..8 {
        a[k][k] = 1;
    }
    let b = [[300i16; 8]; 8];
    assert_eq!(mat8x8_mul_i16(a, b), b);
    assert_eq!(mat8x8_mul_i16(b, b)[0][0], i16::MAX); // 8 * 300 * 300 saturates

    // Inverse of a diagonal 3x3; a singular matrix returns None.
    let d = [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 8.0]];
    let inv = mat3x3_inverse_f32(d).unwrap();
    assert_eq!(inv, [[0.5, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.125]]);
    assert!(mat3x3_inverse_f32([[1.0; 3]; 3]).is_none());

    println!("matrix: all checks passed");
}
