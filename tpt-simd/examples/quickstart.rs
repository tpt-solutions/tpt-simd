//! A tour of the umbrella crate: vector arithmetic, complex multiply, a dot
//! product, a FIR filter and a reduction. Run with:
//!
//! ```text
//! cargo run -p tpt-simd --example quickstart
//! ```

use tpt_simd::convolve::{fir_filter_i16, fir_filter_scalar_i16};
use tpt_simd::dot::dot_product_f32;
use tpt_simd::reduce::sum_f32;
use tpt_simd::{ComplexSimd, F32x8};

fn main() {
    // 1. Eight lanes at once. Complex multiply: (1+2i)(3+4i) = -5 + 10i.
    let a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0));
    let b = ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0));
    let r = a * b;
    println!(
        "complex product, lane 0: {} + {}i",
        r.real.to_array()[0],
        r.imag.to_array()[0]
    );

    // 2. Slice kernels take plain slices; tails are handled for you.
    let x: Vec<f32> = (0..1001).map(|i| i as f32 * 0.5).collect();
    let y: Vec<f32> = vec![2.0; x.len()];
    let dot = dot_product_f32(&x, &y);
    let expected: f32 = x.iter().map(|v| v * 2.0).sum();
    println!("dot product: {dot} (scalar sum {expected})");
    println!("sum:         {}", sum_f32(&x));

    // 3. Causal Q15 FIR filter on i16 samples (output has the input's length).
    // The SIMD and scalar paths must agree bit for bit.
    let samples: Vec<i16> = (0..64).map(|i| (i * 37 % 200) as i16).collect();
    let taps: [i16; 5] = [4096, 8192, 12288, 8192, 4096];
    let mut fast = vec![0i16; samples.len()];
    let mut reference = vec![0i16; samples.len()];
    fir_filter_i16(&samples, &taps, &mut fast);
    fir_filter_scalar_i16(&samples, &taps, &mut reference);
    assert_eq!(fast, reference);
    println!("fir filter: {} outputs, first = {}", fast.len(), fast[0]);
}
