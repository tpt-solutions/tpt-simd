//! Property tests: audio kernels against per-sample definitions.

use proptest::prelude::*;
use tpt_simd_audio::{deinterleave, fir, interleave, mix_into, peak, rms};

fn samples(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1.0f32..1.0, 0..max_len)
}

proptest! {
    #[test]
    fn mix_into_adds_scaled_source(dst in samples(300), gain in -2.0f32..2.0) {
        let src: Vec<f32> = dst.iter().map(|x| x * 0.5 + 0.25).collect();
        let mut out = dst.clone();
        mix_into(&mut out, &src, gain);
        for i in 0..dst.len() {
            prop_assert!((out[i] - (dst[i] + gain * src[i])).abs() <= 1e-5);
        }
    }

    #[test]
    fn interleave_round_trips(frames in 0usize..200, channels in 1usize..=4, seed in any::<u32>()) {
        let data: Vec<Vec<f32>> = (0..channels)
            .map(|c| (0..frames).map(|i| (i as u32 ^ seed ^ (c as u32) << 16) as f32).collect())
            .collect();
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let mut inter = vec![0f32; frames * channels];
        interleave(&refs, &mut inter);
        for i in 0..frames {
            for c in 0..channels {
                prop_assert_eq!(inter[i * channels + c], data[c][i]);
            }
        }
        let mut back: Vec<Vec<f32>> = vec![vec![0f32; frames]; channels];
        let mut outs: Vec<&mut [f32]> = back.iter_mut().map(|v| v.as_mut_slice()).collect();
        deinterleave(&inter, &mut outs);
        prop_assert_eq!(back, data);
    }

    #[test]
    fn peak_is_max_abs(x in samples(300)) {
        let want = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        prop_assert_eq!(peak(&x), want);
    }

    #[test]
    fn rms_matches_formula(x in samples(300)) {
        if x.is_empty() {
            prop_assert_eq!(rms(&x), 0.0);
        } else {
            let want = (x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
            prop_assert!((f64::from(rms(&x)) - want).abs() <= 1e-5);
        }
    }

    #[test]
    fn fir_matches_causal_convolution(taps in samples(12), x in samples(300)) {
        let mut y = vec![0f32; x.len()];
        fir(&taps, &x, &mut y);
        for n in 0..x.len() {
            let want: f64 = taps
                .iter()
                .enumerate()
                .filter(|&(k, _)| k <= n)
                .map(|(k, &t)| f64::from(t) * f64::from(x[n - k]))
                .sum();
            prop_assert!((f64::from(y[n]) - want).abs() <= 1e-4 * want.abs().max(1.0));
        }
    }
}
