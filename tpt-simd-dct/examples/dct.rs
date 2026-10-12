//! DCT and MDCT for codec-sized blocks, with perfect reconstruction.
//!
//! Run with: `cargo run -p tpt-simd-dct --example dct`

use tpt_simd_dct::{Dct, Mdct, sine_window};

fn main() {
    // Orthonormal DCT-II of a constant block: all energy lands in coefficient 0.
    let dct = Dct::new(4);
    let mut coeffs = [0.0f32; 4];
    dct.forward(&[1.0; 4], &mut coeffs);
    assert!((coeffs[0] - 2.0).abs() < 1e-6);
    assert!(coeffs[1..].iter().all(|c| c.abs() < 1e-6));

    // The inverse recovers the input.
    let mut back = [0.0f32; 4];
    dct.inverse(&coeffs, &mut back);
    assert!(back.iter().all(|x| (x - 1.0).abs() < 1e-5));

    // MDCT with a sine window: hop N, overlap-add the IMDCT outputs.
    let n = 8;
    let mdct = Mdct::new(n);
    let window = sine_window(2 * n);
    let signal: Vec<f32> = (0..4 * n).map(|i| (i as f32 * 0.3).sin()).collect();
    let mut output = vec![0.0f32; signal.len() + 2 * n];
    for frame in 0..(signal.len() / n - 1) {
        let start = frame * n;
        let block: Vec<f32> = (0..2 * n).map(|i| signal[start + i] * window[i]).collect();
        let mut coeffs = vec![0.0f32; n];
        mdct.forward(&block, &mut coeffs);
        let mut synth = vec![0.0f32; 2 * n];
        mdct.inverse(&coeffs, &mut synth);
        for i in 0..2 * n {
            output[start + i] += synth[i] * window[i];
        }
    }
    // Samples covered by two frames are reconstructed (away from the edges).
    for i in n..signal.len() - n {
        assert!((output[i] - signal[i]).abs() < 1e-4, "sample {i}");
    }

    println!("dct: all checks passed");
}
