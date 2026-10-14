//! Property test: MDCT perfect reconstruction, as documented on the crate.
//!
//! Windowed frames of length 2N, hop N, with the same sine window at analysis and synthesis,
//! overlap-added, reproduce the input away from the first and last hop.

use proptest::prelude::*;
use tpt_simd_dct::{Mdct, sine_window};

proptest! {
    #[test]
    fn mdct_overlap_add_reconstructs_the_middle(
        half_n in 1usize..=5,
        frames in 2usize..=6,
        seed in prop::collection::vec(-1.0f32..1.0, 0..400),
    ) {
        let n = 1usize << half_n;
        let len = n * (frames + 1);
        let x: Vec<f32> = (0..len)
            .map(|i| seed.get(i).copied().unwrap_or(0.5 * (i as f32).sin()))
            .collect();

        let window = sine_window(2 * n);
        let mdct = Mdct::new(n);
        let mut out = vec![0f32; len];
        for f in 0..frames {
            let start = f * n;
            let block: Vec<f32> = (0..2 * n).map(|i| x[start + i] * window[i]).collect();
            let mut coeffs = vec![0f32; n];
            mdct.forward(&block, &mut coeffs);
            let mut synth = vec![0f32; 2 * n];
            mdct.inverse(&coeffs, &mut synth);
            for i in 0..2 * n {
                out[start + i] += synth[i] * window[i];
            }
        }

        // Samples covered by two frames are [n, frames * n).
        for i in n..frames * n {
            prop_assert!(
                (out[i] - x[i]).abs() <= 1e-3,
                "sample {}: {} vs {}",
                i,
                out[i],
                x[i]
            );
        }
    }
}
