//! FFT butterflies on 8 lanes at a time, real and complex.
//!
//! Run with: `cargo run -p tpt-simd-butterfly --example butterfly`

use tpt_simd_butterfly::{butterfly_f32, butterfly_with_twiddle_complex_f32};
use tpt_simd_core::{ComplexSimd, F32x8};

fn main() {
    // Real butterfly: (a, b) -> (a + b, a - b), eight pairs at once.
    let mut a = F32x8::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let mut b = F32x8::from_array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    butterfly_f32(&mut a, &mut b);
    assert_eq!(a.to_array(), [9.0; 8]);
    assert_eq!(b.to_array(), [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]);

    // Complex butterfly with a twiddle factor w = i: b*w = i*i = -1.
    let mut a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(0.0));
    let mut b = ComplexSimd::new(F32x8::splat(0.0), F32x8::splat(1.0)); // i
    let w = ComplexSimd::new(F32x8::splat(0.0), F32x8::splat(1.0)); // i
    butterfly_with_twiddle_complex_f32(&mut a, &mut b, w);
    assert_eq!(a.real, F32x8::splat(0.0));
    assert_eq!(b.real, F32x8::splat(2.0));

    println!("butterfly: all checks passed");
}
