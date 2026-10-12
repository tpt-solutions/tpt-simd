//! Convolution and Q15 FIR filtering.
//!
//! Run with: `cargo run -p tpt-simd-convolve --example convolve`

use tpt_simd_convolve::{convolve_1d_f32, convolve_1d_output_len, fir_filter_i16};

fn main() {
    // Full 1-D convolution: output length is input + kernel - 1.
    let input = [1.0f32, 2.0, 3.0];
    let kernel = [1.0f32, 1.0];
    let mut out = [0.0f32; 4];
    assert_eq!(convolve_1d_output_len(input.len(), kernel.len()), 4);
    convolve_1d_f32(&input, &kernel, &mut out);
    assert_eq!(out, [1.0, 3.0, 5.0, 3.0]);

    // Q15 FIR: y[n] = 0.5 * x[n] + 0.5 * x[n-1], with 16384 = 0.5 in Q15.
    let mut y = [0i16; 4];
    fir_filter_i16(&[100, 200, 300, 400], &[16384, 16384], &mut y);
    assert_eq!(y, [50, 150, 250, 350]);

    println!("convolve: all checks passed");
}
