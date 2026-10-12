//! Multiply variants: high-half, and complex multiply.
//!
//! Run with: `cargo run -p tpt-simd-mul --example mul`

use tpt_simd_core::{F32x8, I16x16};
use tpt_simd_mul::{ComplexSimd, complex_mul_f32, mul_hi_i16};

fn main() {
    // High 16 bits of each 32-bit product: (-32768)^2 = 2^30, high half 0x4000.
    let r = mul_hi_i16(I16x16::splat(i16::MIN), I16x16::splat(i16::MIN));
    assert_eq!(r, I16x16::splat(0x4000));

    // (1+2i)(3+4i) = -5 + 10i.
    let a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0));
    let b = ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0));
    let r = complex_mul_f32(a, b);
    assert_eq!(r.real.to_array(), [-5.0; 8]);
    assert_eq!(r.imag.to_array(), [10.0; 8]);

    println!("mul: all checks passed");
}
