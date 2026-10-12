//! Dot products for integer and float signals, including the saturating and complex variants.
//!
//! Run with: `cargo run -p tpt-simd-dot --example dot`

use tpt_simd_core::{ComplexSimd, F32x8};
use tpt_simd_dot::{
    dot_product_complex_f32, dot_product_f32, dot_product_i16, dot_product_saturating_i16,
};

fn main() {
    assert_eq!(dot_product_i16(&[1, 2, 3], &[4, 5, 6]), 32);
    assert_eq!(dot_product_i16(&[], &[]), 0);
    // (-32768)^2 * 2 = 2^31 wraps to i32::MIN.
    assert_eq!(dot_product_i16(&[i16::MIN; 2], &[i16::MIN; 2]), i32::MIN);

    // The saturating variant clamps instead of wrapping.
    let big = [i16::MIN; 64];
    assert_eq!(dot_product_saturating_i16(&big, &big), i32::MAX);

    assert_eq!(dot_product_f32(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);

    // Complex, lane-wise and not conjugated: (1+2i)(3+4i) = -5+10i.
    let a = [ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0))];
    let b = [ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0))];
    let r = dot_product_complex_f32(&a, &b);
    assert_eq!(r.real.to_array(), [-5.0; 8]);
    assert_eq!(r.imag.to_array(), [10.0; 8]);

    println!("dot: all checks passed");
}
