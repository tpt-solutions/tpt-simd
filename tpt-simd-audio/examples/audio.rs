//! Audio kernels on `f32` samples: mixing, levels, filtering and resampling.
//!
//! Run with: `cargo run -p tpt-simd-audio --example audio`

use tpt_simd_audio::{Biquad, deinterleave, fir, interleave, mix_into, peak, resample_sinc, rms};

fn main() {
    // Mix a source into a bus at half gain.
    let mut bus = [0.0f32; 4];
    mix_into(&mut bus, &[1.0, 1.0, 1.0, 1.0], 0.5);
    assert_eq!(bus, [0.5; 4]);

    // Stereo: planar <-> interleaved.
    let left = [1.0f32, 2.0];
    let right = [-1.0f32, -2.0];
    let mut packed = [0.0f32; 4];
    interleave(&[&left, &right], &mut packed);
    assert_eq!(packed, [1.0, -1.0, 2.0, -2.0]);
    let (mut l, mut r) = ([0.0f32; 2], [0.0f32; 2]);
    deinterleave(&packed, &mut [&mut l, &mut r]);
    assert_eq!(l, left);
    assert_eq!(r, right);

    // Level meters.
    assert_eq!(peak(&packed), 2.0);
    assert!(rms(&packed) > 0.0);

    // Causal FIR with a two-tap moving average (zero history before the start).
    let mut y = [0.0f32; 4];
    fir(&[0.5, 0.5], &[1.0, 1.0, 1.0, 1.0], &mut y);
    assert_eq!(y, [0.5, 1.0, 1.0, 1.0]);

    // Low-pass biquad at 1 kHz for a 48 kHz stream, applied in place.
    let mut x = [1.0f32, 0.0, 0.0, 0.0];
    let mut lp = Biquad::lowpass(48_000.0, 1_000.0, core::f64::consts::FRAC_1_SQRT_2);
    lp.process(&mut x);
    assert!(peak(&x) > 0.0);

    // Resample 48 kHz -> 24 kHz (windowed-sinc, Blackman window).
    let input: Vec<f32> = (0..480).map(|n| (n as f32 * 0.05).sin()).collect();
    let mut out = vec![0.0f32; 240];
    let written = resample_sinc(&input, 48_000.0, 24_000.0, 32, &mut out);
    assert_eq!(written, 240);
    println!("resampled {} -> {} samples", input.len(), written);

    println!("audio: all checks passed");
}
