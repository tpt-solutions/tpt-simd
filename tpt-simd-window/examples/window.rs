//! Window functions for spectral analysis: Hann and Kaiser.
//!
//! Run with: `cargo run -p tpt-simd-window --example window`

use tpt_simd_window::{apply_window_f32, hanning_window_into_f32, kaiser_window_into_f32};

fn main() {
    // Hann window: symmetric, with zero endpoints.
    let mut w = [0.0f32; 8];
    hanning_window_into_f32(&mut w);
    assert!(w[0].abs() < 1e-5 && w[7].abs() < 1e-5);

    // Apply it to a frame before an FFT.
    let mut frame = [1.0f32; 8];
    apply_window_f32(&mut frame, &w);
    assert_eq!(frame, w);

    // Kaiser window with beta = 8, written into a caller buffer.
    let mut k = [0.0f32; 64];
    kaiser_window_into_f32(&mut k, 8.0);
    // Kaiser endpoints are 1 / I0(beta), about 0.002 for beta = 8; the peak is 1.
    assert!(k[0] < 0.01);
    assert!(k.iter().any(|&x| x > 0.99));

    println!("window: all checks passed");
}
