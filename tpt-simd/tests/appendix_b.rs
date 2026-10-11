//! Appendix B of `spec.txt`, checked against the real API.
//!
//! Each example is the spec's snippet with the imports resolved through the
//! umbrella crate and the result asserted, so the documented outputs are
//! verified rather than just compiled.

use tpt_simd::butterfly::butterfly_f32;
use tpt_simd::dot::dot_product_i16;
use tpt_simd::fixed::Fixed;
use tpt_simd::permute::transpose_8x8_i16;
use tpt_simd::{ComplexSimd, F32x8};

/// B.1 Complex multiplication (FFT).
#[test]
fn b1_complex_multiplication() {
    let a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0));
    let b = ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0));
    let result = a * b; // (1+2i)(3+4i) = -5 + 10i

    assert_eq!(result.real.to_array(), [-5.0; 8]);
    assert_eq!(result.imag.to_array(), [10.0; 8]);
}

/// B.2 Fixed-point multiplication (motion compensation).
#[test]
fn b2_fixed_point_multiplication() {
    type Q = Fixed<i32, 16, 16, 8>;
    let a = Q::from_f32(F32x8::splat(1.5));
    let b = Q::from_f32(F32x8::splat(2.5));
    let result = a * b; // 1.5 * 2.5 = 3.75

    assert_eq!(result.to_f32().to_array(), [3.75; 8]);
    // 3.75 in Q16.16 is 3.75 * 2^16 = 245760.
    assert_eq!(result.raw().to_array(), [245_760; 8]);
}

/// B.3 Dot product (FLAC LPC).
#[test]
fn b3_dot_product() {
    let a: Vec<i16> = (0..256).collect();
    let b: Vec<i16> = (0..256).collect();
    let result = dot_product_i16(&a, &b);

    // Sum of k^2 for k = 0..=255.
    let expected: i64 = (0..256i64).map(|k| k * k).sum();
    assert_eq!(i64::from(result), expected);
}

/// B.4 8x8 transpose (video DCT).
#[test]
fn b4_transpose_8x8() {
    let mut block: [[i16; 8]; 8] = [[0; 8]; 8];
    for (r, row) in block.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = (r * 8 + c) as i16;
        }
    }
    transpose_8x8_i16(&mut block);

    for (r, row) in block.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            assert_eq!(v, (c * 8 + r) as i16, "row {r}, column {c}");
        }
    }
}

/// B.5 FFT butterfly.
#[test]
fn b5_butterfly() {
    let mut a = F32x8::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let mut b = F32x8::from_array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    butterfly_f32(&mut a, &mut b);

    assert_eq!(a.to_array(), [9.0; 8]);
    assert_eq!(b.to_array(), [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]);
}
