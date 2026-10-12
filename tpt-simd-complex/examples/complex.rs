//! Complex arithmetic on 8 lanes, FFT butterflies, and split/interleaved layouts.
//!
//! Run with: `cargo run -p tpt-simd-complex --example complex`

use tpt_simd_complex::{ComplexSimd, ComplexSimdF32Ext, interleaved_to_split};
use tpt_simd_core::F32x8;

fn main() {
    // (1+2i)(3+4i) = -5 + 10i, in all 8 lanes at once.
    let a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0));
    let b = ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0));
    let result = a * b;
    assert_eq!(result.real.to_array(), [-5.0; 8]);
    assert_eq!(result.imag.to_array(), [10.0; 8]);

    // Radix-2 butterfly with twiddle w = i.
    let mut a = ComplexSimd::<f32, 8>::splat(1.0, 0.0);
    let mut b = ComplexSimd::<f32, 8>::splat(0.0, 1.0);
    ComplexSimd::fft_butterfly(&mut a, &mut b, ComplexSimd::splat(0.0, 1.0));
    assert_eq!((a.real[0], a.imag[0]), (0.0, 0.0));
    assert_eq!((b.real[0], b.imag[0]), (2.0, 0.0));

    // Interleaved (re, im, re, im, ...) -> split arrays.
    let (mut re, mut im) = ([0.0; 3], [0.0; 3]);
    interleaved_to_split(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &mut re, &mut im);
    assert_eq!(re, [1.0, 3.0, 5.0]);
    assert_eq!(im, [2.0, 4.0, 6.0]);

    println!("complex: all checks passed");
}
