//! Channel-batched biquads over interleaved frames against per-channel processing.

use proptest::prelude::*;
use tpt_simd_audio::{Biquad, deinterleave, interleave, process_interleaved};

fn bank(channels: usize, f0: f64) -> Vec<Biquad> {
    (0..channels)
        .map(|c| Biquad::lowpass(48_000.0, f0 + 150.0 * c as f64, 0.6 + 0.2 * c as f64))
        .collect()
}

proptest! {
    #[test]
    fn interleaved_equals_each_channel_alone(
        channels in 1usize..=4,
        frames in 0usize..200,
        seed in any::<u32>(),
        split in any::<usize>(),
    ) {
        let data: Vec<Vec<f32>> = (0..channels)
            .map(|c| {
                (0..frames)
                    .map(|i| ((i as u32).wrapping_mul(2654435761).wrapping_add(seed ^ c as u32) % 2001) as f32 / 1000.0 - 1.0)
                    .collect()
            })
            .collect();

        // Reference: each channel through its own filter.
        let mut reference = data.clone();
        for (c, ch) in reference.iter_mut().enumerate() {
            let mut f = bank(channels, 800.0)[c];
            f.process(ch);
        }

        // Batched, with the buffer split into two calls at a frame boundary.
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let mut inter = vec![0f32; frames * channels];
        interleave(&refs, &mut inter);
        let mut filters = bank(channels, 800.0);
        let cut = if frames == 0 { 0 } else { split % (frames + 1) } * channels;
        let (a, b) = inter.split_at_mut(cut);
        process_interleaved(&mut filters, a);
        process_interleaved(&mut filters, b);

        let mut out: Vec<Vec<f32>> = vec![vec![0f32; frames]; channels];
        let mut outs: Vec<&mut [f32]> = out.iter_mut().map(|v| v.as_mut_slice()).collect();
        deinterleave(&inter, &mut outs);
        for c in 0..channels {
            for i in 0..frames {
                prop_assert_eq!(out[c][i].to_bits(), reference[c][i].to_bits());
            }
        }
    }
}

#[test]
#[should_panic(expected = "whole number of frames")]
fn rejects_a_partial_frame() {
    let mut filters = bank(2, 1000.0);
    let mut samples = [0.0f32; 5];
    process_interleaved(&mut filters, &mut samples);
}
