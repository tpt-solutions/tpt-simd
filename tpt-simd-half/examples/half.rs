//! Half-precision conversion and Q8_0 quantisation, as used for model weights.
//!
//! Run with: `cargo run -p tpt-simd-half --example half`

use tpt_simd_half::{BlockQ8_0, dequantize_q8_0, f16, f32_to_f16_slice, quantize_q8_0};

fn main() {
    // Slice conversion to f16.
    let src = [1.0f32, 0.5, -2.0];
    let mut half = [f16::ZERO; 3];
    f32_to_f16_slice(&src, &mut half);
    assert_eq!(half[2].to_f32(), -2.0);

    // One Q8_0 block: 32 values, so the input length must be a multiple of 32.
    let values = [1.0f32; 32];
    let mut blocks = [BlockQ8_0 {
        scale: f16::ZERO,
        q: [0; 32],
    }; 1];
    quantize_q8_0(&values, &mut blocks);
    let mut restored = [0.0f32; 32];
    dequantize_q8_0(&blocks, &mut restored);
    assert!((restored[0] - 1.0).abs() < 0.01);

    println!("half: all checks passed");
}
